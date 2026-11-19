"""Collect previously unused native soup grasps before choosing or executing a release."""

import argparse
import json
import os
from pathlib import Path
from shutil import copyfile

from stage4_fixture import require, validate_study
from common import BUDGET, CHEESE, CHUNK, GOALS, SAUCE, SOUP, read, sha, write
from actual_physics_worker import worker_class
from near_table_placement_pair import PHASES
from soup_placement_feasibility_v2 import fingerprint
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
COHORTS = (("empty", (), 935600, 6), ("cheese", (CHEESE,), 935610, 8),
           ("sauce", (SAUCE,), 935620, 8), ("both", (CHEESE, SAUCE), 935630, 12))
TARGET_COUNT = 2
POLICY_SEEDS = {CHEESE: 60, SAUCE: 61, SOUP: 48}
PLACE_CAP = 180
GRASP_LIFT_M = .03
FORK_LIMIT = BUDGET - sum(PHASES.values())
PROTOCOL = "placement_holdout_protocol.md"


def collection_plan():
    return {"schema": "v2-placement-holdout-collection/1",
            "development_commit": "da3963b3b206e7b33ab673d95a8f6896cf305f89",
            "protocol_sha256": sha(HERE / PROTOCOL), "collector_sha256": sha(Path(__file__)),
            "cohorts": [{"name": name, "goals": list(goals), "seeds": list(range(start, start + size))}
                        for name, goals, start, size in COHORTS],
            "quota": TARGET_COUNT, "policy_seeds": POLICY_SEEDS, "previous_object_cap": PLACE_CAP,
            "minimum_soup_lift_m": GRASP_LIFT_M, "latest_fork_step": FORK_LIMIT}


def grasp_row(stream, state):
    row = json.loads(stream.readline())
    require(row["step"] == state["step"] and row["state_sha256"] == state["state_sha256"]
            and row["goals"] == state["audit"]["goals"], "Native grasp sidecar differs from action boundary")
    return row


def collect_grasp(api, factory, policy, bddl, folder, seed, group, goals):
    phases = []
    with factory(bddl, folder, seed) as worker, (folder / "grasp-audit.jsonl").open(encoding="utf-8") as grasp:
        initial = fingerprint(worker.current)
        physical = grasp_row(grasp, worker.current)
        initial_z = worker.current["audit"]["positions"][SOUP][2]
        copyfile(worker.npz, folder / "initial.npz")
        require(not any(physical["goals"].values()) and not any(physical["held_by_target"].values()),
                "Expected an empty initial basket and gripper")
        completed = []
        failure = None
        for obj in (*goals, SOUP):
            prep = obj != SOUP
            allowed = {*completed, obj} if prep else set(completed)
            began = worker.current["step"]
            limit = min(FORK_LIMIT, began + PLACE_CAP) if prep else FORK_LIMIT
            text = api.instruction(obj)
            policy.reset(POLICY_SEEDS[obj])
            reached = False
            while worker.current["step"] < limit and not worker.terminal():
                actions = policy.act(worker, text, min(CHUNK, limit - worker.current["step"]))
                worker.step(actions, obj, text)
                physical = grasp_row(grasp, worker.current)
                observed = {key for key, value in physical["goals"].items() if value}
                reversals = worker.current["transitions"]
                if (not set(completed) <= observed
                        or any(row["object"] in completed and not row["in_basket"] for row in reversals)):
                    failure = "previously_placed_object_reversed"
                elif not observed <= allowed:
                    failure = "unexpected_basket_occupancy"
                elif any(physical["held_by_target"][key] for key in completed):
                    failure = "previously_placed_object_regrasped"
                if failure:
                    break
                if prep:
                    reached = obj in observed and not any(physical["held_by_target"].values())
                else:
                    height = worker.current["audit"]["positions"][SOUP][2] - initial_z
                    reached = physical["held_by_target"] == {key: key == SOUP for key in GOALS} and height >= GRASP_LIFT_M
                if reached:
                    break
            phases.append({"object": obj, "start_step": began, "end_step": worker.current["step"], "qualified": reached})
            if not reached:
                failure = failure or ("prefix_not_placed_unheld" if prep else "no_lifted_soup_grasp")
                break
            if prep:
                completed.append(obj)
        terminal = fingerprint(worker.current)
        require(not grasp.readline(), "Extra native grasp sidecar rows")
        if failure is None:
            require(0 < terminal["step"] <= FORK_LIMIT and physical["goals"] == {key: key in goals for key in GOALS},
                    "Qualified grasp has the wrong basket occupancy")
    record = {"layout_seed": seed, "cohort": group, "eligible": failure is None, "reason": failure, "phases": phases,
              "initial": initial, "terminal": terminal, "physical": physical, "requested_basket": list(goals),
              "soup_lift_m": worker.current["audit"]["positions"][SOUP][2] - initial_z,
              "file_sha256": {name: sha(folder / name) for name in
                              ("actions.jsonl", "grasp-audit.jsonl", "initial.npz", "observation.npz")}}
    observation(folder / "initial.npz", initial)
    observation(folder / "observation.npz", terminal)
    write(folder / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--card", required=True, type=int)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu and args.gpu.startswith("GPU-")
            and os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(args.card) and os.environ.get("MUJOCO_GL") == "egl",
            "Pin the selected CUDA and EGL device before collecting grasps")
    bddl, _ = validate_study(args.study.resolve())
    import run as api

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = collection_plan()
    plan_path = root / "collection-plan.json"
    if plan_path.exists():
        require(read(plan_path) == plan and not (root / "collection.json").exists(), "Collection plan changed or finished")
    else:
        write(plan_path, plan)
    factory = worker_class(api)
    policy = None
    attempts, accepted, initials = [], [], set()
    for group, goals, start, count in COHORTS:
        found = 0
        for seed in range(start, start + count):
            folder = root / "sources" / str(seed)
            result_path = folder / "result.json"
            if result_path.exists():
                record = read(result_path)
                require(record["layout_seed"] == seed and record["cohort"] == group,
                        "Existing fixture belongs to a different frozen case")
            else:
                if folder.exists():
                    folder.rename(folder.with_name(f"{seed}.incomplete-{os.getpid()}"))
                if policy is None:
                    policy = api.Pi05(args.gpu)
                record = collect_grasp(api, factory, policy, bddl, folder, seed, group, goals)
            initial_hash = record["initial"]["state_sha256"]
            require(initial_hash not in initials, "Distinct layout seeds share the same initial state")
            initials.add(initial_hash)
            attempts.append({"layout_seed": seed, "cohort": group, "eligible": record["eligible"],
                             "reason": record["reason"], "step": record["terminal"]["step"],
                             "result_sha256": sha(result_path)})
            if record["eligible"]:
                accepted.append(seed)
                found += 1
            print("GRASP", seed, group, record["terminal"]["step"], record["eligible"], record["reason"], flush=True)
            if found == TARGET_COUNT:
                break
    write(root / "collection.json", {"plan_sha256": sha(plan_path), "attempts": attempts, "accepted": accepted})


if __name__ == "__main__":
    main()
