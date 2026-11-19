"""Paired crowded-basket validation of feedback near the existing release guard."""

import argparse
from itertools import pairwise
from math import degrees, radians
import os
from pathlib import Path

import numpy as np

from stage4_fixture import require, validate_study
from actual_physics_simulator import instrument, load_native
from common import SOUP, read, sha, write
from near_table_placement_pair import REFINED_ROTATION_TOL_DEG, REFINED_XY_TOL_M, placement_error
from placement_holdout import audit_arm, inspect_forks, read_fork, read_index, run_arm
from placement_holdout_supplement import collect, load_reference_sources, make_plan as reference_plan
import placement_preopen_holdout as preopen
from soup_placement_feasibility_v2 import Z_TOL, pinned, read_lines


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "placement_orientation_feedback_protocol.md"
SELECTOR_SHA = "c2958454288fe4d64dad71ba8917de6c54e86abf5d5e770724ec15e22d2d38f1"
PRIOR_SHA = "8a995b4e2efff6d67a06bd3d085291f3649f9d30f96cc3ac827db028f731846f"
DEV_FAILURES = (935632, 935635)
DEV_POSITIVE = 935634
DEV = (*DEV_FAILURES, DEV_POSITIVE)
RESERVE = tuple(range(935636, 935642))
WAIT_XYZ_STEPS = 20
CAP_DEG = 3.
QUOTA = 4


def plan_for(primary, previous):
    prior = read(pinned(previous / "summary.json", PRIOR_SHA))
    seen = {r["seed"] for r in prior["rows"]}
    reference = reference_plan(primary)
    both, = [g for g in reference["groups"] if g["name"] == "both"]
    candidates = [row for row in both["candidates"] if row["seed"] in RESERVE]
    require(tuple(row["seed"] for row in candidates) == RESERVE and not seen.intersection(RESERVE),
            "The independent fork pool has changed")
    pinned(HERE / "placement_preopen_holdout.py", SELECTOR_SHA)
    return {**reference, "schema": "v2-release-pose-feedback/1", "protocol_sha256": sha(PROTOCOL),
            "script_sha256": sha(Path(__file__)), "selector_sha256": SELECTOR_SHA,
            "development_summary_sha256": PRIOR_SHA, "consecutive_xyz_steps": WAIT_XYZ_STEPS,
            "correction_cap_deg": CAP_DEG, "groups": [{**both, "candidates": candidates, "quota": QUOTA}]}


def position_ok(error):
    return error["xy_error_m"] <= REFINED_XY_TOL_M and error["abs_z_error_m"] <= Z_TOL


def ideal_site(sample, soup_reference):
    site = np.asarray(sample["site_rotation"])
    soup = np.asarray(sample["body_poses"][SOUP]["rotation"])
    return soup_reference @ (site.T @ soup).T


class OrientationFeedback:
    def __init__(self, target):
        self.target = target
        self.streak = 0
        self.events = []

    def __call__(self, sample, nominal, guard):
        if not self.events:
            error = placement_error(sample, self.target, guard)
            blocked = position_ok(error) and error["soup_rotation_error_deg"] > REFINED_ROTATION_TOL_DEG
            self.streak = self.streak + 1 if blocked else 0
            if self.streak < WAIT_XYZ_STEPS:
                return nominal
        from scipy.spatial.transform import Rotation

        basis = np.asarray(nominal)
        delta = Rotation.from_matrix(ideal_site(sample, guard) @ basis.T).as_rotvec()
        size = float(np.linalg.norm(delta))
        cap = radians(CAP_DEG)
        correction = delta if size <= cap else delta * (cap / size)
        command = Rotation.from_rotvec(correction).as_matrix() @ basis
        self.events.append({"state_step": sample["step"], "commanded_site_rotation": command.tolist()})
        return command


def rotation_angle(a, b):
    relative = a @ b.T
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]]) / 2.
    sin = np.linalg.norm(skew)
    cos = (np.trace(relative) - 1.) / 2.
    return float(np.arctan2(sin, cos)), skew


def check_feedback(folder, trace, events, target):
    nominal = np.asarray(trace[0]["site_rotation"])
    guard = np.asarray(trace[0]["body_poses"][SOUP]["rotation"])
    by_step = {row["state_step"]: row for row in events}
    require(len(by_step) == len(events), "Repeated feedback events")
    observed = set()
    active, streak = False, 0
    largest = 0.
    xyz_count, low = 0, None
    for before, after in pairwise(trace):
        if after["phase"] != "lower":
            continue
        error = placement_error(before, target, guard)
        if position_ok(error):
            value = error["soup_rotation_error_deg"]
            low = value if low is None else min(low, value)
            xyz_count += 1
        if not active:
            blocked = position_ok(error) and error["soup_rotation_error_deg"] > REFINED_ROTATION_TOL_DEG
            streak = streak + 1 if blocked else 0
            active = streak >= WAIT_XYZ_STEPS
        if not active:
            continue
        step = before["step"]
        require(step in by_step, "A commanded pose correction was not captured")
        observed.add(step)
        desired = np.asarray(by_step[step]["commanded_site_rotation"])
        ideal = ideal_site(before, guard)
        magnitude, _ = rotation_angle(desired, nominal)
        demand, _ = rotation_angle(ideal, nominal)
        residual, _ = rotation_angle(desired, ideal)
        bounded = min(radians(CAP_DEG), demand)
        require(abs(magnitude - bounded) < 1e-7 and abs(residual - (demand - bounded)) < 1e-7,
                "Feedback target does not follow the bounded orientation correction")
        largest = max(largest, degrees(magnitude))
        servo_angle, skew = rotation_angle(desired, np.asarray(before["site_rotation"]))
        sin = np.linalg.norm(skew)
        rotvec = skew * (servo_angle / sin) if sin > 1e-12 else skew
        require(servo_angle < np.pi / 2 and np.allclose(np.clip(rotvec / .5, -.85, .85),
                                                       after["action"][3:6], atol=1e-7, rtol=0),
                f"Saved servo command differs from the audited target in {folder.name}")
    require(observed == set(by_step), "Feedback log is not aligned to the lower-phase actions")
    return {"first_feedback_command_step": None if not events else events[0]["state_step"] + 1,
            "feedback_steps": len(events), "max_correction_deg": round(largest, 6),
            "precommand_lower_xyz_samples": xyz_count,
            "min_soup_error_on_precommand_lower_xyz_deg": None if low is None else round(low, 6)}


def choices_for(root, cases):
    forks, screen = (read_index(root, name, cases) for name in ("inspect.json", "preopen-screen.json"))
    return {seed: (read_fork(root, seed, forks), preopen.read_choice(root, seed, screen, forks)["selected"])
            for seed in cases}


def execute_pairs(root, bddl, cases, choices):
    root.mkdir(parents=True, exist_ok=True)
    native = instrument(load_native())
    native.ensure_libero_config()
    index = {}
    for seed, source in cases.items():
        expected, choice = choices[seed]
        if choice is None:
            index[str(seed)] = {}
            print("FEEDBACK_ABSTAIN", seed, flush=True)
            continue
        base = root / "pairs" / str(seed)
        fixed = run_arm(native, bddl, seed, source, base / "fixed", expected, choice)
        recovery = OrientationFeedback(choice["release_target_m"])
        corrected = run_arm(native, bddl, seed, source, base / "feedback", expected, choice, rotation_feedback=recovery)
        events = base / "feedback" / "orientation-feedback.json"
        write(events, recovery.events)
        index[str(seed)] = {"fixed": fixed, "feedback": corrected, "events": sha(events)}
        print("FEEDBACK_PAIR", seed,
              {arm: read(base / arm / "result.json")["outcome"]["stable_placement"] for arm in ("fixed", "feedback")},
              "feedback_steps", len(recovery.events), flush=True)
    write(root / "orientation-pairs.json", index)


def audit_pairs(root, cases, choices, previous=None):
    index = read_index(root, "orientation-pairs.json", cases)
    rows = []
    for seed, source in cases.items():
        expected, choice = choices[seed]
        result = index[str(seed)]
        if choice is None:
            require(not result, "Unexpected arms for an abstained target")
            rows.append({"seed": seed, "selected_offset_mm": None})
            continue
        require(set(result) == {"fixed", "feedback", "events"}, "Incomplete feedback pair")
        path = root / "pairs" / str(seed)
        target = choice["release_target_m"]
        arms = {arm: audit_arm(path/arm, result[arm], source, expected, target) for arm in ("fixed", "feedback")}
        fixed_trace, feedback_trace = (path / arm / "physics.jsonl" for arm in ("fixed", "feedback"))
        events = read(pinned(path / "feedback" / "orientation-feedback.json", result["events"]))
        corrected = read_lines(feedback_trace)
        stats = check_feedback(path, corrected, events, target)
        if events:
            before = events[0]["state_step"] - corrected[0]["step"] + 1
            require(corrected[:before] == read_lines(fixed_trace)[:before], "Preactivation state parity failed")
        else:
            require(sha(fixed_trace) == sha(feedback_trace), "Inactive feedback altered native control")
        if previous is not None:
            original = previous / "pairs" / str(seed) / "preopen"
            archived = read(pinned(original / "result.json", read(previous / "pair.json")[str(seed)]["preopen"]))
            require(sha(fixed_trace) == archived["file_sha256"]["physics.jsonl"],
                    "The development control no longer matches the archived arm")
        rows.append({"seed": seed, "selected_offset_mm": choice["offset_mm"], "arms": arms, **stats})
    actual = [row for row in rows if "arms" in row]
    score = [(r["arms"]["fixed"]["stable_placement"], r["arms"]["feedback"]["stable_placement"]) for r in actual]
    regressions = sum(bool(arm["reversals"] or arm["completed_regrasp_steps"])
                      for row in actual for arm in row["arms"].values())
    report = {"eligible": len(rows), "pairs": len(actual), "fixed_opened": sum(r["arms"]["fixed"]["opening_step"] is not None for r in actual),
              "feedback_opened": sum(r["arms"]["feedback"]["opening_step"] is not None for r in actual),
              "fixed_stable": sum(a for a, _ in score), "feedback_stable": sum(b for _, b in score),
              "wins": sum(b and not a for a, b in score), "losses": sum(a and not b for a, b in score),
              "ties": sum(a == b for a, b in score), "object_regression_arms": regressions,
              "activated_cases": sum(r["feedback_steps"] > 0 for r in actual)}
    data = {"schema": "v2-orientation-feedback-results/1", "total": report, "rows": rows}
    if previous is not None:
        dev = {r["seed"]: r for r in actual}
        require(set(dev) == set(DEV), "Incomplete development set")
        data["development_gate"] = (
            any(not dev[seed]["arms"]["fixed"]["stable_placement"] and dev[seed]["arms"]["feedback"]["stable_placement"]
                for seed in DEV_FAILURES)
            and dev[DEV_POSITIVE]["arms"]["fixed"]["stable_placement"]
            and dev[DEV_POSITIVE]["arms"]["feedback"]["stable_placement"] and regressions == 0
        )
    write(root / "summary.json", data)
    print("FEEDBACK_AUDIT_OK", report, "development_gate", data.get("development_gate"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "dev-audit", "collect", "inspect", "screen", "pair", "audit"))
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stage = args.stage
    plan = plan_for(args.primary, args.previous)
    if stage in ("dev", "collect", "inspect", "pair"):
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin a free render device")
        bddl, _ = validate_study(args.study.resolve())
    if stage in ("dev", "dev-audit"):
        old_plan = preopen.make_plan(args.primary, args.supplement)
        _, old_cases = load_reference_sources(args.previous, old_plan)
        all_choices = choices_for(args.previous, old_cases)
        cases = {seed: old_cases[seed] for seed in DEV}
        choices = {seed: all_choices[seed] for seed in DEV}
        if stage == "dev":
            execute_pairs(args.output, bddl, cases, choices)
        else:
            audit_pairs(args.output, cases, choices, args.previous)
        return
    if stage == "collect":
        collect(args.output, args.primary, bddl, plan)
        return
    _, cases = load_reference_sources(args.output, plan)
    if stage == "inspect":
        inspect_forks(args.output, bddl, cases)
    elif stage == "screen":
        bddl, _ = validate_study(args.study.resolve())
        preopen.screen(args.output, args.supplement, bddl, cases)
    else:
        choices = choices_for(args.output, cases)
        if stage == "pair":
            execute_pairs(args.output, bddl, cases, choices)
        else:
            audit_pairs(args.output, cases, choices)


if __name__ == "__main__":
    main()
