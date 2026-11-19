"""Replay a frozen acquisition, screen its held pose, and audit the matched placement arms."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from stage4_fixture import require, validate_study
from actual_physics_simulator import instrument, load_native
from common import GOALS, SOUP, read, sha, write
from near_table_placement_pair import (
    COMPLETED, RELEASE_ABOVE_M, STABLE_STEPS, compare_native, lateral_target, outcome,
    placement_error, record, run_placement, select_refined_case,
)
from placement_holdout_collect import FORK_LIMIT, GRASP_LIFT_M, collection_plan
from soup_placement_feasibility_v2 import (
    archive_frame, fingerprint, pinned, read_lines, record_preview, site_formula,
)
from target_follow_audit import observation


FORK_FILE = "fork.json"
ARM_FILES = ("replayed-actions.jsonl", "grasp-audit.jsonl", "physics.jsonl")
DEV_REFINEMENT_SHA = "4fd7235b8fab5fbaa1145671b50c295e00ef92a84ec344bcee698dec08a05962"


def load_sources(root):
    plan_path = root / "collection-plan.json"
    plan = read(plan_path)
    require(plan == collection_plan(), "Collection plan changed")
    collection = read(root / "collection.json")
    require(collection["plan_sha256"] == sha(plan_path), "Collection references the wrong plan")
    by_seed, initial_states = {}, set()
    remaining = iter(collection["attempts"])
    for group in plan["cohorts"]:
        picked = 0
        for seed in group["seeds"]:
            entry = next(remaining, None)
            require(entry is not None and entry["layout_seed"] == seed and entry["cohort"] == group["name"],
                    "Collection skipped or reordered a frozen seed")
            folder = root / "sources" / str(seed)
            result = read(pinned(folder / "result.json", entry["result_sha256"]))
            require(all(result[key] == entry[key] for key in ("layout_seed", "cohort", "eligible", "reason"))
                    and entry["step"] == result["terminal"]["step"] and result["requested_basket"] == group["goals"],
                    f"Collection metadata differs for {seed}")
            for name, digest in result["file_sha256"].items():
                pinned(folder / name, digest)
            observation(folder / "initial.npz", result["initial"])
            observation(folder / "observation.npz", result["terminal"])
            raw = result["initial"]["state_sha256"]
            require(raw not in initial_states, "Repeated initial state")
            initial_states.add(raw)
            actions = read_lines(folder / "actions.jsonl")
            grasp = read_lines(folder / "grasp-audit.jsonl")
            require([row["step"] for row in grasp] == [0, *[row["step"] for row in actions]]
                    and grasp[-1] == result["physical"]
                    and grasp[0]["state_sha256"] == raw
                    and grasp[-1]["state_sha256"] == result["terminal"]["state_sha256"],
                    "Collection native boundary sequence differs")
            previous = 0
            for action, truth in zip(actions, grasp[1:], strict=True):
                require(action["start_step"] == previous and action["step"] - previous == len(action["actions"])
                        and action["state_sha256"] == truth["state_sha256"]
                        and action["audit"]["goals"] == truth["goals"], "Unmatched action interval in source")
                previous = action["step"]
            if result["eligible"]:
                require(previous == result["terminal"]["step"] <= FORK_LIMIT and result["reason"] is None
                        and result["soup_lift_m"] >= GRASP_LIFT_M and grasp[-1]["held_by_target"] == {obj: obj == SOUP for obj in GOALS}
                        and grasp[-1]["goals"] == {obj: obj in group["goals"] for obj in GOALS},
                        "Held state does not satisfy the frozen inclusion rule")
                by_seed[seed] = {"result": result, "actions": actions, "grasp": grasp}
                picked += 1
            if picked == plan["quota"]:
                break
    require(next(remaining, None) is None and list(by_seed) == collection["accepted"], "Qualified cohort list changed")
    return collection, by_seed


@contextmanager
def replay_fork(native, bddl, seed, source, folder):
    folder.mkdir(parents=True)
    (folder / "frames").mkdir()
    scene = native.Simulator(bddl, folder, seed, False)
    try:
        inner = scene.env.env
        owners = {inner.sim.model.geom_name2id(geom): obj for obj in (*GOALS, "basket_1")
                  for geom in inner.objects_dict[obj].contact_geoms}
        mismatches = []
        initial = scene.capture()
        compare_native(initial, source["result"]["initial"], mismatches, False)
        with (folder / "replayed-actions.jsonl").open("x", encoding="utf-8") as output:
            for expected in source["actions"]:
                require(scene.frame == expected["start_step"], "Replay skipped an action boundary")
                captured = scene.advance(expected["actions"])
                actual = {**expected, "actions": expected["actions"][:captured["executed"]], **captured}
                compare_native(actual, expected, mismatches, False)
                output.write(json.dumps(actual, allow_nan=False) + "\n")
        require(read_lines(folder / "grasp-audit.jsonl") == source["grasp"], "Replay grasp prefix differs")
        require(fingerprint(captured) == source["result"]["terminal"], "Held fork RGB-D or physics differs")
        fork = record(scene, owners, "fork", (), False)
        require(fork["held"] == source["grasp"][-1]["held_by_target"] and fork["goals"] == source["grasp"][-1]["goals"],
                "Holding or occupancy differs at the shared fork")
        frame = archive_frame(scene, folder, fork, captured)
        yield scene, owners, fork, frame
    finally:
        scene.close()


def inspect_forks(root, bddl, sources):
    native = instrument(load_native())
    native.ensure_libero_config()
    index = {}
    for seed, source in sources.items():
        folder = root / "forks" / str(seed)
        with replay_fork(native, bddl, seed, source, folder) as (_, _, fork, frame):
            write(folder / FORK_FILE, {"sample": fork, "frame": frame})
        index[str(seed)] = sha(folder / FORK_FILE)
        print("FORK", seed, fork["step"], fork["goals"], flush=True)
    write(root / "inspect.json", index)


def read_index(root, filename, sources):
    index = read(root / filename)
    require(set(index) == {str(seed) for seed in sources}, f"Unexpected cases in {filename}")
    return index


def read_fork(root, seed, index):
    folder = root / "forks" / str(seed)
    result = read(pinned(folder / FORK_FILE, index[str(seed)]))
    frame, sample = result["frame"], result["sample"]
    require(sample["step"] == frame["step"] and sample["state_sha256"] == frame["state_sha256"],
            "Fork frame belongs to a different sample")
    observation(pinned(folder / frame["file"], frame["sha256"]), frame)
    return sample, frame


def screen_forks(root, bddl, sources):
    import near_table_placement_geometry as geometry

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("MUJOCO_GL") == "disable",
            "Disable CUDA and rendering for the collision screen")
    load_native().ensure_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv

    status = read_index(root, "inspect.json", sources)
    old = read(pinned(Path("stage4-near-table-fixed-refinement.json"), DEV_REFINEMENT_SHA))
    dev, dev_region, dev_target, _ = geometry.load_fork(Path("stage4-near-table-placement"))
    scene = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False,
                       has_offscreen_renderer=False, control_freq=20)

    def project(fork, region, target):
        rotation = fork["body_poses"][SOUP]["rotation"]
        meshes = geometry.restore_fork(scene.env, fork, region)
        coarse = geometry.sample_paths(scene.env, meshes, region, target, rotation, geometry.POSE_GRID_MM)
        cases, checked, strong = geometry.refine_fixed_pose(scene.env, fork, region, target, {"fixed_cases": coarse})
        return {"coarse": coarse, "cases": cases, "refined_candidates": checked, "three_mm_jitter_target": strong,
                "selected_target": select_refined_case(checked)}

    try:
        regression = project(dev, dev_region, dev_target)
        require(all(regression[key] == old[key] for key in ("cases", "refined_candidates"))
                and regression["three_mm_jitter_target"] == old["selected_target"]
                and regression["selected_target"]["offset_mm"] == [25, 35],
                "Frozen 931400 release selection or numerical screen changed")
        print("DEVELOPMENT_PARITY 931400 exact", flush=True)
        index = {}
        for seed in sources:
            fork, frame = read_fork(root, seed, status)
            region = frame["region"]
            target = [*region["position_m"][:2], region["objects"][SOUP]["upper"][2] + RELEASE_ABOVE_M]
            result = project(fork, region, target)
            filename = root / "forks" / str(seed) / "screen.json"
            write(filename, {"fork_sha256": status[str(seed)], **result})
            index[str(seed)] = sha(filename)
            choice = result["selected_target"]
            print("SCREEN", seed, None if choice is None else choice["offset_mm"], flush=True)
        write(root / "screen.json", index)
    finally:
        scene.close()


def read_screen(root, seed, index, inspect):
    path = root / "forks" / str(seed) / "screen.json"
    result = read(pinned(path, index[str(seed)]))
    require(result["fork_sha256"] == inspect[str(seed)]
            and result["selected_target"] == select_refined_case(result["refined_candidates"]),
            "Release screen and frozen selection disagree")
    return result["selected_target"]


def score(history):
    completed = tuple(obj for obj in COMPLETED if history[0]["goals"][obj])
    result = outcome(history, completed)
    stable = result.pop("stable_all_in_unheld")
    tail = history[-STABLE_STEPS:]
    return {**result, "stable_placement": stable, "stable_full_task": stable and all(all(r["goals"].values()) for r in tail),
            "first_soup_goal_step": next((r["step"] for r in history if r["goals"][SOUP]), None)}


def run_arm(native, bddl, seed, source, folder, expected, choice, *, rotation_feedback=None):
    with replay_fork(native, bddl, seed, source, folder) as (scene, owners, fork, shared):
        saved_fork, saved_frame = expected
        require(fork == saved_fork and fingerprint(shared) == fingerprint(saved_frame) and shared["region"] == saved_frame["region"],
                "Arms do not share the inspected fork")
        if choice is None:
            target, plan = lateral_target(fork, shared["region"])
        else:
            require(choice["soup_rotation"] == fork["body_poses"][SOUP]["rotation"], "Screened orientation differs")
            target = np.asarray(choice["release_target_m"])
            site = shared["region"]["objects"][SOUP]
            require(np.all(target[:2] > site["lower"][:2]) and np.all(target[:2] < site["upper"][:2])
                    and abs(target[2] - site["upper"][2] - RELEASE_ABOVE_M) < 1e-12, "Screened release bounds differ")
            plan = {"target_m": target.tolist(), "offset_mm": choice["offset_mm"],
                    "nominal_clearance_m": choice["min_clearance_m"], "jitter_clearance_m": choice["dense"]["min_clearance_m"]}
        guard = np.asarray(fork["body_poses"][SOUP]["rotation"])
        history, frames = [fork], {fork["step"]: shared}

        def snapshot(sample):
            step = sample["step"]
            if step not in frames:
                frames[step] = archive_frame(scene, folder, sample, scene.capture())
            return frames[step]

        with (folder / "physics.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
            trace.write(json.dumps(fork, allow_nan=False) + "\n")
            phases, stop, opening = run_placement(scene, owners, history, trace, snapshot, fork, target, guard,
                                                  rotation_feedback=rotation_feedback)
        preview = record_preview(snapshot(history[-1]), folder)
    write(folder / "result.json", {**plan, "outcome": score(history), "phases": phases, "opening": opening,
                                  "stop": stop, "frames": list(frames.values()), "preview": preview,
                                  "file_sha256": {name: sha(folder / name) for name in ARM_FILES}})
    return sha(folder / "result.json")


def run_pairs(root, bddl, sources):
    inspect = read_index(root, "inspect.json", sources)
    screen = read_index(root, "screen.json", sources)
    native = instrument(load_native())
    native.ensure_libero_config()
    index = {}
    for seed, source in sources.items():
        expected = read_fork(root, seed, inspect)
        choice = read_screen(root, seed, screen, inspect)
        folder = root / "pairs" / str(seed)
        arms = {"lateral": run_arm(native, bddl, seed, source, folder / "lateral", expected, None)}
        if choice is not None:
            arms["refined"] = run_arm(native, bddl, seed, source, folder / "refined", expected, choice)
        index[str(seed)] = arms
        print("PAIR", seed, {arm: read(folder / arm / "result.json")["outcome"]["stable_placement"] for arm in arms},
              flush=True)
    write(root / "pair.json", index)


def audit_arm(folder, digest, source, expected, target):
    result = read(pinned(folder / "result.json", digest))
    for name, checksum in result["file_sha256"].items():
        pinned(folder / name, checksum)
    require(result["target_m"] == target and read_lines(folder / "replayed-actions.jsonl") == source["actions"]
            and read_lines(folder / "grasp-audit.jsonl")[:len(source["grasp"])] == source["grasp"],
            "Selected target or replay prefix differs")
    history = read_lines(folder / "physics.jsonl")
    fork, frame = expected
    require(history[0] == fork and fingerprint(result["frames"][0]) == fingerprint(frame)
            and [r["step"] for r in history] == list(range(fork["step"], history[-1]["step"] + 1)),
            "Missing common fork or post-fork control steps")
    states = {row["step"]: row for row in history}
    for rgbd in result["frames"]:
        observation(pinned(folder / rgbd["file"], rgbd["sha256"]), rgbd)
        sample = states[rgbd["step"]]
        require(sample["state_sha256"] == rgbd["state_sha256"], "RGB-D joined to the wrong control step")
        region = rgbd["region"]
        for obj in GOALS:
            predicate = site_formula(region["position_m"], region["rotation"], region["size"],
                                     sample["body_poses"][obj]["position_m"])["strictly_inside"]
            require(predicate == sample["goals"][obj], "Archived region and LIBERO predicate differ")
    pinned(folder / result["preview"]["file"], result["preview"]["sha256"])
    opening = result["opening"]
    if opening is None:
        require(all(r["action"][-1] == 1 for r in history[1:]), "Unplanned release command")
    else:
        preceding = states[opening["state_step"]]
        guard = np.asarray(fork["body_poses"][SOUP]["rotation"])
        measured = placement_error(preceding, target, guard)
        require(preceding["held"][SOUP] and measured["within_tolerance"] and opening["within_tolerance"]
                and states[opening["command_step"]]["action"][-1] == -1
                and all(np.allclose(measured[key], opening[key], atol=1e-9, rtol=0)
                        for key in ("xy_error_m", "abs_z_error_m", "soup_rotation_error_deg")),
                "Opening command did not pass the frozen guard")
    require(score(history) == result["outcome"], "Placement summary disagrees with native trace")
    return {"opening_step": None if opening is None else opening["command_step"], "stop": result["stop"],
            "target_m": target, "action_intervals": len(source["actions"]), "frames": len(result["frames"]),
            **result["outcome"]}


def aggregate(rows):
    totals = Counter(held_states=len(rows))
    for row in rows:
        arms = row["arms"]
        first = arms["lateral"]["stable_placement"]
        totals["lateral_stable"] += first
        totals["lateral_full"] += arms["lateral"]["stable_full_task"]
        totals["lateral_regressions"] += bool(arms["lateral"]["reversals"] or arms["lateral"]["completed_regrasp_steps"])
        if "refined" in arms:
            second = arms["refined"]["stable_placement"]
            totals["targets_found"] += 1
            totals["matched_lateral_stable"] += first
            totals["refined_stable"] += second
            totals["refined_full"] += arms["refined"]["stable_full_task"]
            totals["refined_regressions"] += bool(arms["refined"]["reversals"] or arms["refined"]["completed_regrasp_steps"])
            totals["pair_wins"] += second and not first
            totals["pair_losses"] += first and not second
            totals["pair_ties"] += first == second
        else:
            totals["no_target"] += 1
    return dict(totals)


def audit_pairs(root, collection, sources):
    inspect = read_index(root, "inspect.json", sources)
    screen = read_index(root, "screen.json", sources)
    pairs = read_index(root, "pair.json", sources)
    rows = []
    for seed, source in sources.items():
        expected = read_fork(root, seed, inspect)
        fork, frame = expected
        require(fingerprint(frame) == source["result"]["terminal"], "Inspected fork changed from acquisition")
        choice = read_screen(root, seed, screen, inspect)
        digests = pairs[str(seed)]
        expected_arms = {"lateral"} if choice is None else {"lateral", "refined"}
        require(set(digests) == expected_arms, "Missing matching arm or unexplained abstention")
        lateral, _ = lateral_target(fork, frame["region"])
        targets = {"lateral": lateral.tolist()}
        if choice is not None:
            targets["refined"] = choice["release_target_m"]
        folder = root / "pairs" / str(seed)
        arms = {arm: audit_arm(folder / arm, digest, source, expected, targets[arm]) for arm, digest in digests.items()}
        rows.append({"seed": seed, "cohort": source["result"]["cohort"], "occupied": source["result"]["requested_basket"],
                     "step": fork["step"], "state_sha256": fork["state_sha256"], "refined_target": choice, "arms": arms})
    groups = {}
    for group in collection_plan()["cohorts"]:
        key = group["name"]
        attempts = [r for r in collection["attempts"] if r["cohort"] == key]
        groups[key] = {"attempts": len(attempts), "rejections": Counter(r["reason"] for r in attempts if not r["eligible"]),
                       **aggregate([row for row in rows if row["cohort"] == key])}
    summary = {"schema": "v2-placement-holdout-results/1", "total_attempts": len(collection["attempts"]), "groups": groups,
               "total": aggregate(rows), "rows": rows}
    write(root / "summary.json", summary)
    print("AUDIT_OK", json.dumps({"groups": groups, "total": summary["total"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "screen", "pair", "audit"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    collection, sources = load_sources(root)
    if args.stage == "audit":
        audit_pairs(root, collection, sources)
        return
    bddl, _ = validate_study(args.study.resolve())
    if args.stage != "screen":
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None
                and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"), "Pin the EGL device for replay")
    stages = {"inspect": inspect_forks, "screen": screen_forks, "pair": run_pairs}
    stages[args.stage](root, bddl, sources)


if __name__ == "__main__":
    main()
