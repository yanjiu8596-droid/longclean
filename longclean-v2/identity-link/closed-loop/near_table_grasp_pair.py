"""Test the archived RGB-D grasp skill at the last clear view of a failed pickup."""

import argparse
from math import dist
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import BUDGET, CHEESE, SAUCE, SOUP, V1, read, sha, write
from actual_physics_worker import worker_class
from basket_retreat_pair import resume
from basket_stall_probe import archive, replay_row, replay_rows, verify_sources
from dense_online import physical_sample, reason_at_terminal
from dense_online_diagnostic import checked, jsonl, load_arm
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
LAYOUT, FORK, RESUME = 931400, 300, 450
HOLDOUT_SHA = "b3d8f538150a80809df5479957fcf379bf1dae13ab80becd017f9dcc8a2655ca"
BLOCKERS_SHA = "7241f01190c9be24b923012dd0ea2a8d45592d64983917efdfe49d896ead034c"
REFERENCE_SHA = "6efa8f8fc6627d7173e4ef7e1dd54901df2b37cd5cd7c5aaea01b7689a723568"
TIMING_SHA = "673fdf159ae4044bbaef16f2923691d4023a8ef785d96fbfd0d6e2cc63f48a68"
PLACEMENT_BODIES = (CHEESE, SAUCE, "basket_1")
SITE_FROM_SENSOR = Rotation.from_euler("z", -90., degrees=True).as_matrix()


def pose(path):
    with np.load(path, allow_pickle=False) as frame:
        return (frame["eef_pos"], Rotation.from_quat(frame["eef_quat"]).as_matrix() @ SITE_FROM_SENSOR,
                abs(frame["gripper_qpos"]).sum().item())


def prepare(api, root):
    report = read(checked(root, "summary.json", HOLDOUT_SHA))
    blockers = read(checked(root, "blockers.json", BLOCKERS_SHA))
    require(blockers["source_sha256"] == HOLDOUT_SHA, "Diagnostic source changed")
    case, = [r for r in report["rows"] if r["layout_seed"] == LAYOUT]
    require(case["status"] == "same_path_no_trigger" and not case["baseline"]["official_success"],
            "Expected the archived near-table failure")
    source = load_arm(root, {**case["baseline"], "layout_seed": LAYOUT, "arm": "adaptive1"})
    folder = root / "shadow" / str(LAYOUT)
    locate = api.load_locator()
    estimates, fingerprints = [], []
    for step in (290, FORK):
        row, = [r for r in read(folder / "samples.json") if r["step"] == step]
        saved = row["observation"]
        frame = checked(folder, saved["file"], saved["sha256"])
        observation(frame, saved["fingerprint"])
        xyz, rotation, gap = pose(frame)
        native = source["endpoint"][step]
        require(saved["fingerprint"] == api.fingerprint(native) and np.max(abs(rotation - native["audit"]["site_rotation"])) < 1e-6
                and np.array_equal(xyz, native["audit"]["site_xyz"]), "Sensor orientation cannot be mapped to the native controller site")
        estimate = locate(frame)
        require(estimate is not None and row["target"] == SOUP and gap >= .075, "No clear open-gripper frame for assisted pickup")
        estimates.append(estimate)
        fingerprints.append(saved)
    require(dist(estimates[0]["xyz_m"], estimates[1]["xyz_m"]) <= .01, "Source pose estimates disagree")
    require(not any(source["physical"][FORK]["held_by_target"].values()) and
            source["physical"][FORK]["goals"] == {CHEESE: True, SAUCE: True, SOUP: False}, "Wrong source grasp state")
    reference = read(checked(V1, "grasp-reference.json", REFERENCE_SHA))
    timing = read(checked(V1, "trajectory.json", TIMING_SHA))
    require(16 + sum(timing["frames"].values()) == RESUME - FORK, "Archived timing changed")
    return source, estimate, reference, timing, fingerprints


def control_sequence(xyz, estimate, reference, timing):
    contact = np.asarray(estimate["xyz_m"]) + reference["grasp_xyz_offset_m"] + timing["contact_target_adjustment_m"]
    above = contact + [0., 0., timing["approach_height_m"]]
    lifted = contact + [0., 0., timing["lift_height_m"]]
    retreat = xyz.copy()
    retreat[2] = max(.74, above[2], xyz[2])
    frames = timing["frames"]
    return [("retreat", xyz, retreat, 16, -1.), ("approach", above, above, frames["approach"], -1.),
            ("descend", above, contact, frames["descend"], -1.), ("settle", contact, contact, frames["settle"], -1.),
            ("close", contact, contact, frames["close"], 1.), ("lift", contact, lifted, frames["lift"], 1.),
            ("hold", lifted, lifted, frames["hold"], 1.)]


def servo(position, rotation, target, goal_rotation, grip, timing, reference):
    bounds = reference["native_limits"]
    scale = (np.asarray(bounds["out_max"]) - bounds["out_min"]) / 2
    angle = Rotation.from_matrix(goal_rotation @ rotation.T).as_rotvec()
    delta = np.r_[timing["position_gain"] * (target - position), timing["rotation_gain"] * angle] / scale
    limit = timing["normalized_limit"]
    return np.r_[np.clip(delta, -limit, limit), grip].astype(np.float32).tolist()


def run_branch(api, policy, bddl, source, estimate, reference, timing, expected, branch, output):
    worker_cls = worker_class(api)
    prefix = replay_rows(source, FORK)
    seed = source["result"]["policy_seed"]
    script_rows, stop = [], None
    with worker_cls(bddl, output, LAYOUT) as worker, (output / "grasp-audit.jsonl").open(encoding="utf-8") as grasp:
        initial = physical_sample(grasp, worker.current)
        require(initial == source["physical"][0] and api.fingerprint(worker.current) == source["result"]["initial"],
                "Changed initial state")
        for row in prefix:
            replay_row(worker, row)
            require(physical_sample(grasp, worker.current) == source["physical"][row["step"]], "Changed archived grasp evidence")
        fork = archive(api, worker, output, "fork.npz")
        require(fork["fingerprint"] == expected["fingerprint"], "Development states are not paired")
        start_xyz, start_rotation, _ = pose(worker.npz)
        before = worker.current["audit"]
        shifts = {key: 0. for key in PLACEMENT_BODIES}
        rotation = np.asarray(reference["base_rotation_matrix"])
        phases = control_sequence(start_xyz, estimate, reference, timing)
        latest = source["physical"][FORK]
        for phase, head, tail, duration, grip in phases:
            for i in range(duration):
                xyz, attitude, _ = pose(worker.npz)
                point = head + (tail - head) * ((i + 1) / duration)
                wanted = start_rotation if phase == "retreat" else rotation
                action = servo(xyz, attitude, point, wanted, grip, timing, reference) if branch == "assisted_grasp" else [0.] * 6 + [-1.]
                worker.step([action], SOUP, f"grasp study {branch} {phase}")
                latest = physical_sample(grasp, worker.current)
                for key in PLACEMENT_BODIES:
                    shifts[key] = max(shifts[key], dist(before["positions"][key], worker.current["audit"]["positions"][key]))
                script_rows.append({"step": latest["step"], "phase": phase, "goals": latest["goals"],
                                    "soup_held": latest["held_by_target"][SOUP]})
                if (not latest["goals"][CHEESE] or not latest["goals"][SAUCE] or any(latest["held_by_target"][k] for k in (CHEESE, SAUCE))
                        or max(shifts.values()) > .005 or worker.terminal()):
                    stop = "script_evaluator_stop"
                    break
            if stop:
                break
        handoff = archive(api, worker, output, "handoff.npz")
        handoff_held = latest["held_by_target"][SOUP]
        handoff_height = worker.current["audit"]["positions"][SOUP][2] - before["positions"][SOUP][2]
        chunks = []
        if stop is None:
            require(worker.current["step"] == RESUME, "Policy handoff was not matched in time")
            chunks, latest, _, _ = resume(api, worker, grasp, policy, seed)
        terminal = archive(api, worker, output, "terminal.npz")
        reason = stop or reason_at_terminal(worker.current)
        require(not grasp.readline(), "Extra physical sample")
    rows = jsonl(output / "actions.jsonl")
    physics = jsonl(output / "grasp-audit.jsonl")
    require(rows[:len(prefix)] == prefix and [r["step"] for r in physics] == [0, *[r["step"] for r in rows]]
            and rows[-1]["step"] <= BUDGET and physics[-1] == latest, "Malformed native record")
    for row, state in zip(rows, physics[1:], strict=True):
        require(row["state_sha256"] == state["state_sha256"] and row["audit"]["goals"] == state["goals"], "Changed scoring sample")
    goals = latest["goals"]
    reversals = [t for row in rows if row["step"] > FORK for t in row["transitions"]
                 if t["object"] in (CHEESE, SAUCE) and not t["in_basket"]]
    arm = {"branch": branch, "fork": fork, "handoff": handoff, "terminal": terminal, "terminal_reason": reason,
           "policy_seed": seed, "terminal_goals": goals, "terminal_held": latest["held_by_target"],
           "handoff_soup_held": handoff_held, "handoff_soup_height_change_m": handoff_height,
           "first_sampled_soup_grasp": next((r["step"] for r in physics if r["step"] > FORK and r["held_by_target"][SOUP]), None),
           "official_success": all(goals.values()), "success_all_unheld": all(goals.values()) and not any(latest["held_by_target"].values()),
           "completed_object_max_script_displacement_m": shifts, "completed_object_reversals": reversals,
           "actions_sha256": sha(output / "actions.jsonl"), "grasp_audit_sha256": sha(output / "grasp-audit.jsonl")}
    write(output / "result.json", {**arm, "script_samples": script_rows, "policy_chunks": chunks})
    return {**arm, "result_sha256": sha(output / "result.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=GPU, type=int, required=True)
    for key in ("study", "source", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    bddl, _ = validate_study(args.study.resolve())
    validate_gpu(args.card)
    verify_sources()
    import run as api

    source, estimate, reference, timing, evidence = prepare(api, args.source.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_info = {"schema": "v2-near-table-grasp-pair/1", "runner_sha256": sha(Path(__file__)), "layout": LAYOUT,
                "source_sha256": HOLDOUT_SHA, "blockers_sha256": BLOCKERS_SHA, "reference_sha256": REFERENCE_SHA,
                "timing_sha256": TIMING_SHA, "pose_estimate": estimate, "source_observations": evidence,
                "fork_step": FORK, "matched_policy_resume_step": RESUME, "total_budget": BUDGET}
    write(output / "plan.json", run_info)
    policy = api.Pi05(GPU[args.card])
    arms = {}
    for name in ("wait", "assisted_grasp"):
        arms[name] = run_branch(api, policy, bddl, source, estimate, reference, timing, evidence[-1], name, output / name)
        arm = arms[name]
        print("NEAR_TABLE_GRASP", name, arm["handoff_soup_held"], arm["terminal_reason"], arm["success_all_unheld"], flush=True)
    require(arms["wait"]["fork"]["fingerprint"] == arms["assisted_grasp"]["fork"]["fingerprint"], "Paired checkpoint changed")
    write(output / "summary.json", {**run_info, "arms": arms})


if __name__ == "__main__":
    main()
