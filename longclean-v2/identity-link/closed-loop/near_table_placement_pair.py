"""Paired evaluator-only placement from the sealed step-450 soup grasp."""

import argparse
from itertools import pairwise
import json
from math import dist
import os
from pathlib import Path
import socket
from time import monotonic

import numpy as np

from stage4_fixture import require, validate_study
from actual_physics_simulator import instrument, load_native
from common import BUDGET, CHEESE, GOALS, SAUCE, SOUP, array_hash, read, sha, write
from soup_placement_feasibility_v2 import (
    archive_frame, fingerprint, osc_control, physical_sample, pinned, read_lines,
    record_preview, site_formula, target_error,
)
from target_follow_audit import observation

LAYOUT, FORK, STABLE_STEPS = 931400, 450, 20
SOURCE_SHA = "5520f09d85380af8b746577bf3a732882c96cdc5545c842010a87a8ea9e3dced"
EVALUATOR_SHA = "d8ef0f1f5df2525cdbb152ea1b62644bb533a6bd2d453e98e65e58c5d979fe7e"
INSTRUMENT_SHA = "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8"
PIXEL_ARRAYS = {"camera1", "camera2", "camera1_depth", "camera2_depth"}
COMPLETED = (CHEESE, SAUCE)
PHASES = {"lift": 35, "transfer": 90, "lower": 95, "open": 20, "retreat": 40, "settle": 40}
LATERAL_M, RELEASE_ABOVE_M, MAX_STABLE_SPREAD_M = .03, .025, .005
REFINED_XY_TOL_M, REFINED_ROTATION_TOL_DEG = .001, 1.


def load_source(root):
    summary = read(pinned(root / "summary.json", SOURCE_SHA))
    meta = summary["arms"]["assisted_grasp"]
    arm = root / "assisted_grasp"
    result = read(pinned(arm / "result.json", meta["result_sha256"]))
    require(all(result[key] == value for key, value in meta.items() if key != "result_sha256"),
            "Archived arm metadata disagree")
    actions = read_lines(pinned(arm / "actions.jsonl", result["actions_sha256"]))
    grasp = read_lines(pinned(arm / "grasp-audit.jsonl", result["grasp_audit_sha256"]))
    states = {row["step"]: row for row in grasp}
    handoff = result["handoff"]["fingerprint"]
    initial = {"step": 0, "raw_sha256": handoff["raw_sha256"], "bddl_sha256": handoff["bddl_sha256"],
               "state_sha256": grasp[0]["state_sha256"]}
    require(summary["layout"] == LAYOUT and summary["matched_policy_resume_step"] == handoff["step"] == FORK
            and summary["total_budget"] == BUDGET == actions[-1]["step"] and result["policy_seed"] == 48
            and result["handoff_soup_held"] and not result["official_success"], "Wrong sealed episode")
    require([row["step"] for row in grasp] == [0, *[row["step"] for row in actions]]
            and len(states) == len(grasp), "Malformed native boundary sequence")
    for previous, row in zip([initial, *actions], actions):
        require(previous["step"] == row["start_step"] and row["step"] - row["start_step"] == len(row["actions"])
                and row["state_sha256"] == states[row["step"]]["state_sha256"]
                and row["audit"]["goals"] == states[row["step"]]["goals"], "Unmatched action/grasp boundary")
    fork = next(row for row in actions if row["step"] == FORK)
    require(fingerprint(fork) == handoff and states[FORK]["held_by_target"] == {obj: obj == SOUP for obj in GOALS}
            and states[FORK]["goals"] == {obj: obj != SOUP for obj in GOALS}, "Wrong held-soup fork")
    for name in ("handoff", "terminal"):
        frame = result[name]
        path = pinned(arm / frame["file"], frame["sha256"])
        observation(path, frame["fingerprint"])
    return {"root": arm, "result": result, "actions": actions, "grasp": states, "initial": initial, "fork": fork}


def compare_native(actual, expected, pixel_mismatches, different_renderer):
    actual = dict(actual)
    if "observation_sha256" in expected:
        new = actual["observation_sha256"]
        old = expected["observation_sha256"]
        changed = sorted(key for key in old if new[key] != old[key])
        if changed:
            pixel_mismatches.append({"step": actual["step"], "arrays": changed})
        if different_renderer and set(changed) <= PIXEL_ARRAYS:
            actual["observation_sha256"] = old
    require(all(actual[key] == value for key, value in expected.items()),
            f"Archived physical state or native response differs at step {actual['step']}")


def anchor_pixels(source, label, archived, dest):
    frame = source["result"][label]
    if frame["fingerprint"]["observation_sha256"] == archived["observation_sha256"]:
        return {"exact": True}
    with np.load(source["root"] / frame["file"], allow_pickle=False) as old, np.load(dest / archived["file"], allow_pickle=False) as new:
        differences = {}
        for key in sorted(PIXEL_ARRAYS):
            delta = np.abs(old[key].astype(np.float64) - new[key])
            differences[key] = {"fraction_changed": float(np.mean(delta > 0)),
                                "mean_absolute_error": float(delta.mean()), "max_absolute_error": float(delta.max())}
    return {"exact": False, "pixel_differences": differences}


def record(scene, owners, phase, action, done):
    import mujoco

    sample = physical_sample(scene)
    sim = scene.env.env.sim
    support = dict.fromkeys((*COMPLETED, "basket_1"), 0.)
    force = np.zeros(6, dtype=np.float64)
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        first, second = owners.get(int(contact.geom1)), owners.get(int(contact.geom2))
        other = second if first == SOUP else first if second == SOUP else None
        if other in support:
            mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
            support[other] += float(force[0])
    require(array_hash(scene.env.get_sim_state()) == sample["state_sha256"], "Contact read changed state")
    return {**sample, "phase": phase, "action": list(action), "environment_done": bool(done),
            "soup_normal_force_N": support}


def outcome(history, completed=COMPLETED):
    release = next((right["step"] for left, right in pairwise(history) if left["held"][SOUP] and not right["held"][SOUP]), None)
    official = next((row["step"] for row in history if all(row["goals"].values())), None)
    regrasp = [row["step"] for row in history[1:] if any(row["held"][obj] for obj in completed)]
    reversals = [{"step": right["step"], "object": obj} for left, right in pairwise(history)
                 for obj in completed if left["goals"][obj] and not right["goals"][obj]]
    tail = history[-STABLE_STEPS:]
    points = [row["body_poses"][SOUP]["position_m"] for row in tail]
    spread = max(dist(first, second) for first in points for second in points)
    stable = (len(tail) == STABLE_STEPS and tail[-1]["step"] - tail[0]["step"] == STABLE_STEPS - 1
              and release is not None and tail[0]["step"] >= release
              and all(all(row["goals"][obj] for obj in (*completed, SOUP)) and not any(row["held"].values())
                      for row in tail)
              and spread <= MAX_STABLE_SPREAD_M and not reversals and not regrasp)
    return {"first_unheld_step": release, "first_official_success_step": official, "stable_all_in_unheld": stable,
            "tail_spread_m": spread, "terminal_step": history[-1]["step"], "terminal_goals": history[-1]["goals"],
            "terminal_held": history[-1]["held"], "reversals": reversals, "completed_regrasp_steps": regrasp,
            "max_soup_normal_force_N": {obj: max(row["soup_normal_force_N"][obj] for row in history)
                                        for obj in history[0]["soup_normal_force_N"]}}


def lateral_target(fork, region):
    center = np.asarray(region["position_m"])
    completed = [obj for obj in COMPLETED if fork["goals"][obj]]
    clearance = {side: min(dist((center[0] + dx, center[1]), fork["body_poses"][obj]["position_m"][:2])
                           for obj in completed) if completed else 0.
                 for side, dx in (("right", LATERAL_M), ("left", -LATERAL_M))}
    side = max(clearance, key=clearance.get)  # Insertion order favors positive world x on a tie.
    offset = LATERAL_M if side == "right" else -LATERAL_M
    site = region["objects"][SOUP]
    point = np.array([center[0] + offset, center[1], site["upper"][2] + RELEASE_ABOVE_M])
    require(np.all(point[:2] > site["lower"][:2]) and np.all(point[:2] < site["upper"][:2]),
            "Sealed lateral offset lies outside the region XY extent")
    return point, {"chosen_side": side, "target_m": point.tolist(), "center_separation_m": clearance}


def select_refined_case(candidates):
    eligible = [case for case in candidates
                if case["min_clearance_m"] >= .003 and case["dense"]["min_clearance_m"] > 0]
    return max(eligible, key=lambda item: (item["dense"]["min_clearance_m"], item["min_clearance_m"],
                                          -dist(item["offset_mm"], (0, 0))), default=None)


def load_refined_release(path, digest, source):
    report = read(pinned(path, digest))
    require(report["schema"] == "v2-near-table-fixed-refinement/1" and report["layout"] == LAYOUT
            and report["step"] == FORK and report["fork_state_sha256"] == source["fork"]["state_sha256"]
            and report["min_clearance_m"] == .003, "Refinement does not match the sealed placement fork")
    case = select_refined_case(report["refined_candidates"])
    require(case is not None, "No nominally clear fixed pose also cleared the sampled position errors")
    return {"screen_sha256": digest, "offset_mm": case["offset_mm"], "target_m": case["release_target_m"],
            "locked_soup_rotation": case["soup_rotation"], "nominal_clearance_m": case["min_clearance_m"],
            "jitter_clearance_m": case["dense"]["min_clearance_m"]}


def placement_error(sample, target, rotation_guard):
    soup = sample["body_poses"][SOUP]
    error = target_error(soup["position_m"], target)
    if rotation_guard is not None:
        relative = rotation_guard.T @ np.asarray(soup["rotation"])
        cosine = np.clip((np.trace(relative) - 1.) / 2., -1., 1.)
        angle = float(np.degrees(np.arccos(cosine)))
        error["soup_rotation_error_deg"] = angle
        error["within_tolerance"] = (error["within_tolerance"] and error["xy_error_m"] <= REFINED_XY_TOL_M
                                     and angle <= REFINED_ROTATION_TOL_DEG)
    return error


def execute_arm(simulator, bddl, source, dest, candidate, different_renderer, common_fork=None,
                refined_release=None, use_refined_target=False):
    dest.mkdir()
    (dest / "frames").mkdir()
    scene = simulator(bddl, dest, LAYOUT, False)
    try:
        inner = scene.env.env
        owners = {inner.sim.model.geom_name2id(geom): obj for obj in (*GOALS, "basket_1")
                  for geom in inner.objects_dict[obj].contact_geoms}
        initial = scene.capture()
        compare_native(initial, source["initial"], [], False)
        mismatches = []
        history, frames, stages = [], {}, []

        def snapshot(sample, captured=None):
            step = sample["step"]
            if step not in frames:
                captured = captured if captured is not None else scene.capture()
                frames[step] = archive_frame(scene, dest, sample, captured)
            return frames[step]

        def replay(original, stream):
            require(scene.frame == original["start_step"], "Mismatch at native action boundary")
            current = scene.advance(original["actions"])
            actual = {"start_step": original["start_step"], "target": original["target"],
                      "instruction": original["instruction"], "actions": original["actions"][:current["executed"]], **current}
            stream.write(json.dumps(actual, allow_nan=False) + "\n")
            compare_native(actual, original, mismatches, different_renderer)
            return current

        with (dest / "replayed-actions.jsonl").open("x", encoding="utf-8") as native_log:
            for row in source["actions"]:
                if row["step"] > FORK:
                    break
                captured = replay(row, native_log)
            fork = record(scene, owners, "fork", (), False)
            require(scene.frame == FORK and fork["held"] == source["grasp"][FORK]["held_by_target"], "Wrong replay fork")
            history.append(fork)
            handoff = snapshot(fork, captured)
            if common_fork is not None:
                require(fingerprint(handoff) == common_fork, "Arms do not share the same RGB-D and state fork")
            print(f"{dest.name} matched step {FORK}, historical pixel changes at {len(mismatches)} boundaries", flush=True)
            report = {"handoff": handoff, "archived_handoff_pixels": anchor_pixels(source, "handoff", handoff, dest)}

            with (dest / "physics.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
                trace.write(json.dumps(fork, allow_nan=False) + "\n")
                if candidate:
                    target, choice = lateral_target(fork, handoff["region"])
                    rotation_guard = None
                    if refined_release is not None:
                        require(refined_release["locked_soup_rotation"] == fork["body_poses"][SOUP]["rotation"],
                                "Refined release rotation differs from the matched grasp")
                        rotation_guard = np.asarray(refined_release["locked_soup_rotation"])
                    if use_refined_target:
                        target = np.asarray(refined_release["target_m"])
                        site = handoff["region"]["objects"][SOUP]
                        require(np.all(target[:2] > site["lower"][:2]) and np.all(target[:2] < site["upper"][:2])
                                and abs(target[2] - site["upper"][2] - RELEASE_ABOVE_M) < 1e-12,
                                "Refined target is outside the sealed release bounds")
                        choice = dict(refined_release)
                    write(dest / "placement-target.json", choice)
                    report.update(choice)
                    stages, stop, opening = run_placement(scene, owners, history, trace, snapshot, fork, target,
                                                          rotation_guard)
                    report.update(phases=stages, stop=stop, opening_error=opening)
                else:
                    step_native = scene.env.step

                    def measured_step(action):
                        result = step_native(action)
                        scene.obs = result[0]
                        sample = record(scene, owners, "archived_policy", action.tolist(), result[2])
                        sample["step"] = scene.frame + 1
                        history.append(sample)
                        trace.write(json.dumps(sample, allow_nan=False) + "\n")
                        return result

                    scene.env.step = measured_step
                    try:
                        for row in source["actions"]:
                            if row["step"] <= FORK:
                                continue
                            captured = replay(row, native_log)
                            if scene.frame % 10 == 0 or scene.frame == BUDGET:
                                snapshot(history[-1], captured)
                    finally:
                        scene.env.step = step_native

            final = snapshot(history[-1])
            if not candidate:
                report["archived_terminal_pixels"] = anchor_pixels(source, "terminal", final, dest)
                require(read_lines(dest / "grasp-audit.jsonl") == list(source["grasp"].values()),
                        "Archived all-object grasp sidecar differs")
            else:
                historical_prefix = [row for step, row in source["grasp"].items() if step <= FORK]
                require(read_lines(dest / "grasp-audit.jsonl")[:len(historical_prefix)] == historical_prefix,
                        "Candidate grasp prefix differs")
        require([row["step"] for row in history] == list(range(FORK, scene.frame + 1)), "Trace lost a control step")
        report.update(outcome(history), frames=list(frames.values()), preview=record_preview(final, dest),
                      archived_pixel_mismatches=mismatches,
                      file_sha256={name: sha(dest / name) for name in ("replayed-actions.jsonl", "grasp-audit.jsonl", "physics.jsonl")})
        write(dest / "result.json", report)
        return report
    finally:
        scene.close()


def run_placement(scene, owners, history, trace, snapshot, fork, target, rotation_guard=None,
                  rotation_feedback=None):
    require(rotation_feedback is None or rotation_guard is not None, "Pose feedback requires a locked soup orientation")
    start = np.asarray(fork["body_poses"][SOUP]["position_m"])
    high = max(.70, start[2], target[2] + .10)
    rotation = fork["site_rotation"]
    completed = tuple(obj for obj in COMPLETED if fork["goals"][obj])
    budget_end = fork["step"] + sum(PHASES.values())
    require(budget_end <= BUDGET, "The full placement budget must fit after the fork")
    stop, opening = None, None
    corrected = False
    phases = []

    def advance(phase, command):
        nonlocal stop
        before = history[-1]
        action = np.asarray(command, dtype=np.float32)
        require(scene.frame < budget_end, "Candidate exceeded sealed budget")
        try:
            scene.obs, _, done, _ = scene.env.step(action)
        except ValueError as error:
            if all(before["goals"].values()) and "terminated episode" in str(error).lower():
                stop = "native_environment_refused_diagnostic_tail"
                return
            raise
        scene.frame += 1
        sample = record(scene, owners, phase, action.tolist(), done)
        history.append(sample)
        trace.write(json.dumps(sample, allow_nan=False) + "\n")
        if (scene.frame % 10 == 0 or before["held"][SOUP] != sample["held"][SOUP]
                or before["goals"] != sample["goals"]):
            snapshot(sample)
        if any(not sample["goals"][obj] or sample["held"][obj] for obj in completed):
            stop = "completed_object_goal_or_grasp_changed"
        elif done and not all(sample["goals"].values()):
            stop = "environment_done_without_all_goals"

    for phase, waypoint in (("lift", np.r_[start[:2], high]), ("transfer", np.r_[target[:2], high]), ("lower", target)):
        began = scene.frame
        guard = rotation_guard if phase == "lower" else None
        for _ in range(PHASES[phase]):
            current = history[-1]
            soup = np.asarray(current["body_poses"][SOUP]["position_m"])
            if stop or placement_error(current, waypoint, guard)["within_tolerance"]:
                break
            offset = np.asarray(current["site_xyz_m"]) - soup
            desired = rotation
            if phase == "lower" and rotation_feedback is not None:
                desired = rotation_feedback(current, rotation, rotation_guard)
                corrected = corrected or not np.array_equal(desired, rotation)
            action = osc_control(current, waypoint + offset, desired, 1, False)
            if phase == "lower":
                action[2] = max(-.12, action[2])
            advance(phase, action)
            if stop is None and not history[-1]["held"][SOUP]:
                stop = f"soup_grasp_lost_during_{phase}"
        error = placement_error(history[-1], waypoint, guard)
        phases.append({"phase": phase, "start_step": began, "end_step": scene.frame, "target_m": waypoint.tolist(), **error})
        if stop or not error["within_tolerance"]:
            return phases, stop or f"{phase}_target_not_reached", opening

    opening = {"state_step": scene.frame, "command_step": scene.frame + 1, **error}
    open_at = np.array(history[-1]["site_xyz_m"])
    retreat_at = open_at + [0., 0., .09]
    release_rotation = history[-1]["site_rotation"] if corrected else rotation
    snapshot(history[-1])
    for phase, waypoint in (("open", open_at), ("retreat", retreat_at), ("settle", None)):
        began = scene.frame
        for _ in range(PHASES[phase]):
            if stop:
                break
            action = [0.] * 6 + [-1.] if waypoint is None else osc_control(history[-1], waypoint, release_rotation, -1., False)
            advance(phase, action)
        phases.append({"phase": phase, "start_step": began, "end_step": scene.frame})
        if stop:
            break
    return phases, stop or "settling_budget_finished", opening


def audit(root, source, refined_release=None):
    summary = read(root / "summary.json")
    expected_arms = {"baseline", "placement"} if refined_release is None else {"baseline", "lateral", "placement"}
    require(summary.get("refined_release") == refined_release and set(summary["arms"]) == expected_arms,
            "Audit inputs do not match the recorded arms or release plan")
    for arm, expected in summary["arms"].items():
        folder = root / arm
        result = read(pinned(folder / "result.json", expected["result_sha256"]))
        for name, digest in result["file_sha256"].items():
            pinned(folder / name, digest)
        history = read_lines(folder / "physics.jsonl")
        replay = read_lines(folder / "replayed-actions.jsonl")
        mismatches = []
        for actual, expected_row in zip(replay, (r for r in source["actions"] if r["step"] <= replay[-1]["step"]), strict=True):
            compare_native(actual, expected_row, mismatches, summary["different_renderer_allowed"])
        require(mismatches == result["archived_pixel_mismatches"], "Historical pixel differences disagree")
        require(replay[-1]["step"] == (BUDGET if arm == "baseline" else FORK), "Incomplete archived replay")
        require([row["step"] for row in history] == list(range(FORK, result["terminal_step"] + 1)), "Missing physical control steps")
        require(fingerprint(result["handoff"]) == summary["common_fork"], "Common fork changed")
        require(all(result[k] == v for k, v in outcome(history).items()), "Reported outcome differs from trace")
        state_by_step = {row["step"]: row for row in history}
        for frame in result["frames"]:
            observation(pinned(folder / frame["file"], frame["sha256"]), frame)
            sample = state_by_step[frame["step"]]
            require(frame["state_sha256"] == sample["state_sha256"], "Frame and trace state disagree")
            region = frame["region"]
            for obj, goal in sample["goals"].items():
                placement = site_formula(region["position_m"], region["rotation"], region["size"],
                                         sample["body_poses"][obj]["position_m"])
                require(placement["strictly_inside"] == goal, "Official predicate differs from archived geometry")
        pinned(folder / result["preview"]["file"], result["preview"]["sha256"])
        if arm != "baseline":
            if refined_release is not None:
                if arm == "placement":
                    require(all(result[key] == value for key, value in refined_release.items()), "Refined release differs")
                else:
                    _, expected_choice = lateral_target(history[0], result["handoff"]["region"])
                    require(all(result[key] == value for key, value in expected_choice.items()), "Lateral control differs")
            opening = result["opening_error"]
            if opening is not None:
                previous = state_by_step[opening["state_step"]]
                require(previous["held"][SOUP] and opening["within_tolerance"]
                        and state_by_step[opening["command_step"]]["action"][-1] == -1., "Opening guard differs")
                if refined_release is not None:
                    guard = np.asarray(refined_release["locked_soup_rotation"])
                    measured = placement_error(previous, result["target_m"], guard)
                    error_fields = ("xyz_error_m", "xy_error_m", "abs_z_error_m", "soup_rotation_error_deg")
                    require(measured["within_tolerance"] and all(np.allclose(opening[key], measured[key], atol=1e-9, rtol=0)
                                                                  for key in error_fields),
                            "Opening does not meet the refined placement tolerances")
            else:
                require(all(row["action"][-1] == 1. for row in history[1:]), "Unexpected release command")
    print(f"CPU audit passed for {len(expected_arms)} paired arms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "audit"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--card", type=int)
    parser.add_argument("--different-renderer", action="store_true")
    parser.add_argument("--refined-screen", type=Path)
    parser.add_argument("--refined-screen-sha256")
    args = parser.parse_args()
    bddl, _ = validate_study(args.study.resolve())
    source = load_source(args.source.resolve())
    require(bool(args.refined_screen) == bool(args.refined_screen_sha256)
            and not (args.refined_screen and args.different_renderer),
            "Refined placement requires its SHA-256 and strict renderer parity")
    refined_release = None
    if args.refined_screen is not None:
        refined_release = load_refined_release(args.refined_screen, args.refined_screen_sha256, source)
    here = Path(__file__).resolve().parent
    pinned(here / "soup_placement_feasibility_v2.py", EVALUATOR_SHA)
    pinned(here / "actual_physics_simulator.py", INSTRUMENT_SHA)
    if args.mode == "check":
        print(json.dumps({"source_sha256": SOURCE_SHA, "layout": LAYOUT, "fork": source["grasp"][FORK],
                          "actions": len(source["actions"]), "budget": BUDGET, "policy_seed": 48,
                          "refined_release": refined_release}, ensure_ascii=False))
        return
    output = args.output.resolve()
    if args.mode == "audit":
        audit(output, source, refined_release)
        return
    require(args.card is not None and os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(args.card)
            and os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"),
            "Missing pinned CUDA and EGL devices")
    started = monotonic()
    output.mkdir(parents=True, exist_ok=False)
    native = instrument(load_native())
    native.ensure_libero_config()
    baseline = execute_arm(native.Simulator, bddl, source, output / "baseline", False, args.different_renderer)
    shared = fingerprint(baseline["handoff"])
    reports = {"baseline": baseline}
    if refined_release is not None:
        reports["lateral"] = execute_arm(native.Simulator, bddl, source, output / "lateral", True, False, shared,
                                         refined_release=refined_release)
    reports["placement"] = execute_arm(native.Simulator, bddl, source, output / "placement", True,
                                        args.different_renderer, shared, refined_release=refined_release,
                                        use_refined_target=refined_release is not None)
    kind = "v2-near-table-placement-pair/1" if refined_release is None else "v2-near-table-refined-pair/1"
    summary = {"schema": kind, "layout": LAYOUT, "fork_step": FORK,
          "source_sha256": SOURCE_SHA, "runner_sha256": sha(Path(__file__)), "common_fork": shared,
          "different_renderer_allowed": args.different_renderer, "host": socket.gethostname(), "gpu_index": args.card,
          "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "elapsed_seconds": monotonic() - started,
          "arms": {key: {"result_sha256": sha(output / key / "result.json"), **outcome(read_lines(output / key / "physics.jsonl")),
                          "pixel_mismatch_boundaries": len(report["archived_pixel_mismatches"]),
                          "opening_error": report.get("opening_error"), "stop": report.get("stop")}
                   for key, report in reports.items()}}
    if refined_release is not None:
        summary["refined_release"] = refined_release
    write(output / "summary.json", summary)


if __name__ == "__main__":
    main()
