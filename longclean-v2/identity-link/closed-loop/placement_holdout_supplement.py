"""Reach unfilled occupancy strata using the existing native reference pickup."""

import argparse
import os
from pathlib import Path
from shutil import copyfile

from stage4_fixture import require, validate_study
from common import CHEESE, GOALS, SAUCE, SOUP, V1, read, sha, write
from actual_physics_worker import worker_class
from near_table_placement_pair import compare_native
from placement_holdout import audit_pairs, inspect_forks, load_sources, run_pairs, screen_forks
from placement_holdout_collect import FORK_LIMIT, GRASP_LIFT_M, TARGET_COUNT, grasp_row
from soup_placement_feasibility_v2 import fingerprint, pinned, read_lines
from target_follow_audit import observation


REFERENCE_SHA = "6efa8f8fc6627d7173e4ef7e1dd54901df2b37cd5cd7c5aaea01b7689a723568"
PROTOCOL = Path(__file__).with_name("placement_holdout_supplement_protocol.md")


def make_plan(primary):
    collection, qualified = load_sources(primary)
    groups = []
    for name, goals in (("cheese", (CHEESE,)), ("both", (CHEESE, SAUCE))):
        wanted = TARGET_COUNT - sum(case["result"]["cohort"] == name for case in qualified.values())
        candidates = []
        for entry in collection["attempts"]:
            if entry["cohort"] != name or entry["eligible"]:
                continue
            seed = entry["layout_seed"]
            original = read(pinned(primary / "sources" / str(seed) / "result.json", entry["result_sha256"]))
            phases = original["phases"][:len(goals)]
            if [phase["object"] for phase in phases if phase["qualified"]] == list(goals):
                candidates.append({"seed": seed, "prefix_step": phases[-1]["end_step"],
                                   "source_result_sha256": entry["result_sha256"]})
        groups.append({"name": name, "quota": wanted, "goals": list(goals), "candidates": candidates})
    return {"schema": "v2-held-state-reference-supplement/1", "primary_collection_sha256": sha(primary / "collection.json"),
            "reference_sha256": REFERENCE_SHA, "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
            "groups": groups}


class PickupProbe:
    """Expose only the frozen reference pickup interface, with the preregistered fork stop."""

    def __init__(self, worker, grasp, physical, goals, initial_z):
        self.worker, self.grasp, self.physical = worker, grasp, physical
        self.expected = {obj: obj in goals for obj in GOALS}
        self.initial_z = initial_z
        self.reason = None

    @property
    def current(self):
        return self.worker.current

    def goal(self, obj):
        return self.worker.goal(obj)

    def sufficiently_lifted(self):
        return self.current["audit"]["positions"][SOUP][2] - self.initial_z >= GRASP_LIFT_M

    def qualified(self):
        return self.reason is None and self.sufficiently_lifted() and self.physical["held_by_target"] == {
            obj: obj == SOUP for obj in GOALS}

    def terminal(self):
        return bool(self.reason or self.qualified() or self.current["step"] >= FORK_LIMIT or self.worker.terminal())

    def step(self, actions, target, instruction):
        self.worker.step(actions, target, instruction)
        self.physical = grasp_row(self.grasp, self.current)
        if self.physical["goals"] != self.expected or any(self.physical["held_by_target"][obj]
                                                          for obj in GOALS if obj != SOUP):
            self.reason = "occupancy_or_companion_grasp_changed_during_pickup"


def collect_one(api, new_worker, primary, root, bddl, group, planned):
    seed = planned["seed"]
    original_dir = primary / "sources" / str(seed)
    original = read(pinned(original_dir / "result.json", planned["source_result_sha256"]))
    actions = read_lines(pinned(original_dir / "actions.jsonl", original["file_sha256"]["actions.jsonl"]))
    truths = read_lines(pinned(original_dir / "grasp-audit.jsonl", original["file_sha256"]["grasp-audit.jsonl"]))
    truth_by_step = {row["step"]: row for row in truths}
    folder = root / "sources" / str(seed)
    with new_worker(bddl, folder, seed) as worker, (folder / "grasp-audit.jsonl").open(encoding="utf-8") as grasp:
        compare_native(worker.current, original["initial"], [], False)
        require(grasp_row(grasp, worker.current) == truth_by_step[0], "Archived initial grasp differs")
        initial_z = worker.current["audit"]["positions"][SOUP][2]
        copyfile(worker.npz, folder / "initial.npz")
        physical = truth_by_step[0]
        for action in actions:
            if action["step"] > planned["prefix_step"]:
                break
            worker.step(action["actions"], action["target"], action["instruction"])
            compare_native(worker.current, worker_fields(action), [], False)
            physical = grasp_row(grasp, worker.current)
            require(physical == truth_by_step[action["step"]], "Archived occupancy prefix differs")
        require(worker.current["step"] == planned["prefix_step"]
                and physical["goals"] == {obj: obj in group["goals"] for obj in GOALS}
                and not any(physical["held_by_target"].values()), "Prefix does not have the planned occupancy")
        probe = PickupProbe(worker, grasp, physical, group["goals"], initial_z)
        api.scripted_soup(probe)
        require(not grasp.readline(), "Extra physical samples in reference pickup")
        final = fingerprint(worker.current)
        record = {"layout_seed": seed, "cohort": group["name"], "prefix": planned, "initial": original["initial"],
                  "terminal": final, "physical": probe.physical, "eligible": probe.qualified(),
                  "reason": None if probe.qualified() else probe.reason or "no_lifted_reference_grasp",
                  "requested_basket": group["goals"], "soup_lift_m": worker.current["audit"]["positions"][SOUP][2] - initial_z,
                  "file_sha256": {name: sha(folder / name) for name in
                                  ("actions.jsonl", "grasp-audit.jsonl", "initial.npz", "observation.npz")}}
    write(folder / "result.json", record)
    return record, sha(folder / "result.json")


def worker_fields(row):
    return {key: value for key, value in row.items() if key not in ("start_step", "target", "instruction", "actions")}


def collect(root, primary, bddl, plan):
    pinned(V1 / "grasp-reference.json", REFERENCE_SHA)
    import run as api

    root.mkdir(parents=True, exist_ok=False)
    write(root / "collection-plan.json", plan)
    new_worker = worker_class(api)
    attempts, eligible = [], []
    for group in plan["groups"]:
        accepted = 0
        for planned in group["candidates"]:
            if accepted == group["quota"]:
                break
            record, digest = collect_one(api, new_worker, primary, root, bddl, group, planned)
            seed = record["layout_seed"]
            attempts.append({"layout_seed": seed, "cohort": group["name"], "eligible": record["eligible"],
                             "reason": record["reason"], "result_sha256": digest})
            if record["eligible"]:
                eligible.append(seed)
                accepted += 1
            print("REFERENCE_GRASP", seed, group["name"], record["terminal"]["step"], record["eligible"], record["reason"],
                  flush=True)
    write(root / "collection.json", {"plan_sha256": sha(root / "collection-plan.json"), "attempts": attempts,
                                    "accepted": eligible})


def load_reference_sources(root, plan):
    plan_path = root / "collection-plan.json"
    collection = read(root / "collection.json")
    require(read(plan_path) == plan and collection["plan_sha256"] == sha(plan_path), "Supplement plan changed")
    remaining = iter(collection["attempts"])
    cases = {}
    for group in plan["groups"]:
        accepted = 0
        for planned in group["candidates"]:
            if accepted == group["quota"]:
                break
            seed = planned["seed"]
            row = next(remaining, None)
            require(row is not None and row["layout_seed"] == seed and row["cohort"] == group["name"],
                    "Supplement skipped or reordered a prefix")
            folder = root / "sources" / str(seed)
            result = read(pinned(folder / "result.json", row["result_sha256"]))
            require(result["prefix"] == planned and result["requested_basket"] == group["goals"]
                    and all(row[key] == result[key] for key in ("layout_seed", "cohort", "eligible", "reason")),
                    "Supplement source changed")
            for name, digest in result["file_sha256"].items():
                pinned(folder / name, digest)
            observation(folder / "initial.npz", result["initial"])
            observation(folder / "observation.npz", result["terminal"])
            actions, grasp = (read_lines(folder / name) for name in ("actions.jsonl", "grasp-audit.jsonl"))
            require([state["step"] for state in grasp] == [0, *[action["step"] for action in actions]]
                    and grasp[-1] == result["physical"] and grasp[-1]["state_sha256"] == result["terminal"]["state_sha256"],
                    "Supplement grasp trace differs")
            for action, state in zip(actions, grasp[1:], strict=True):
                require(action["state_sha256"] == state["state_sha256"] and action["audit"]["goals"] == state["goals"],
                        "Supplement action and grasp barriers differ")
            if result["eligible"]:
                require(result["terminal"]["step"] <= FORK_LIMIT and result["soup_lift_m"] >= GRASP_LIFT_M
                        and result["physical"]["goals"] == {obj: obj in group["goals"] for obj in GOALS}
                        and result["physical"]["held_by_target"] == {obj: obj == SOUP for obj in GOALS},
                        "Supplement did not have the verified lifted hold")
                cases[seed] = {"result": result, "actions": actions, "grasp": grasp}
                accepted += 1
    require(next(remaining, None) is None and list(cases) == collection["accepted"], "Supplement order changed")
    return collection, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "inspect", "screen", "pair", "audit"))
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = make_plan(args.primary)
    if args.stage not in ("screen", "audit"):
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin a free CUDA and EGL device")
    if args.stage == "collect":
        bddl, _ = validate_study(args.study)
        collect(args.output, args.primary, bddl, plan)
        return
    collection, sources = load_reference_sources(args.output, plan)
    if args.stage == "audit":
        audit_pairs(args.output, collection, sources)
        return
    bddl, _ = validate_study(args.study)
    stages = {"inspect": inspect_forks, "screen": screen_forks, "pair": run_pairs}
    stages[args.stage](args.output, bddl, sources)


if __name__ == "__main__":
    main()
