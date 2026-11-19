"""Independently verify native action traces and count paired wins by layout."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from common import ARMS, BUDGET, EXECUTE, file_sha, read_json, read_physics, write_json


CONTRASTS = (("corrected", "base"), ("corrected", "natural"), ("corrected", "rule"), ("natural", "base"))


def audit_episode(run, case, seed, arm, plan_sha, gpu, adapters):
    cohort, layout = case["cohort"], case["layout_seed"]
    folder = run / "eval" / cohort / str(layout) / str(seed) / arm
    result = read_json(folder / "result.json")
    fields = (result["cohort"], result["layout_seed"], result["policy_seed"], result["arm"],
              result["initial"], result["test_plan_sha256"], result["gpu"]["uuid"], result["adapter_sha256"])
    if fields != (cohort, layout, seed, arm, case["initial"], plan_sha, gpu, adapters.get(arm)):
        raise ValueError(f"Episode differs from its frozen assignment: {folder}")
    for name, key in (("actions.jsonl", "actions_sha256"), ("audit.jsonl", "audit_sha256")):
        if file_sha(folder / name) != result[key]:
            raise ValueError(f"Evidence checksum differs: {folder / name}")
    if "video_sha256" in result and file_sha(folder / "replay.mp4") != result["video_sha256"]:
        raise ValueError(f"Replay checksum differs: {folder}")
    trace = [json.loads(line) for line in (folder / "actions.jsonl").read_bytes().splitlines()]
    cursor = 0
    observed = []
    for row in trace:
        if (row["start_step"] != cursor
                or not 1 <= row["executed"] == len(row["commands"]) == len(row["proposals"]) <= EXECUTE):
            raise ValueError("Non-contiguous or oversized control chunk")
        for step, (before, after) in enumerate(zip(row["proposals"], row["commands"], strict=True), cursor + 1):
            if len(before) != 7 or len(after) != 7:
                raise ValueError("Invalid native action shape")
            if before != after:
                if before[:-1] != after[:-1] or before[-1] <= -0.5 or after[-1] != -1 or arm != "rule":
                    raise ValueError("Unexpected runtime intervention")
                observed.append({"step": step, "policy_gripper": before[-1], "gripper": after[-1]})
        cursor += row["executed"]
    if (cursor != result["steps"] or not 1 <= cursor <= BUDGET
            or observed != result["edits"] or trace[-1]["success"] != result["success"]
            or (not result["success"] and not result["done"] and cursor != BUDGET)):
        raise ValueError("Trace does not agree with official terminal or release edits")
    for name, value in read_physics(folder, cursor, result["success"]).items():
        if result[name] != value:
            raise ValueError(f"Physical result does not replay: {name}")
    return result


def per_arm(rows):
    return {
        "episodes": len(rows),
        "official_success": sum(row["success"] for row in rows),
        "held": sum(row["first_held"] is not None for row in rows),
        "release_through_terminal": sum(row["release"] for row in rows),
        "contact_episode_count": sum(row["non_goal_contact_steps"] != 0 for row in rows),
        "non_goal_contact_steps": sum(row["non_goal_contact_steps"] for row in rows),
        "median_first_hold_step": median([row["first_held"] for row in rows if row["first_held"] is not None])
        if any(row["first_held"] is not None for row in rows) else None,
        "median_first_release_step": median([row["first_release_in_basket"] for row in rows
                                             if row["first_release_in_basket"] is not None])
        if any(row["first_release_in_basket"] is not None for row in rows) else None,
        "median_steps": median([row["steps"] for row in rows]) if rows else None,
        "median_seconds": median([row["seconds"] for row in rows]) if rows else None,
        "peak_allocated_mib": max((row["peak_allocated_mib"] for row in rows), default=None),
    }


def paired(cases, candidate, control, metric):
    changes = defaultdict(int)
    wins, losses = 0, 0
    for trial in cases:
        a, b = trial[candidate], trial[control]
        delta = int(a[metric]) - int(b[metric])
        wins += delta > 0
        losses += delta < 0
        changes[a["layout_seed"]] += delta
    return {"wins": wins, "losses": losses, "ties": len(cases) - wins - losses, "net": wins - losses,
            "positive_layouts": sorted(layout for layout, count in changes.items() if count > 0),
            "negative_layouts": sorted(layout for layout, count in changes.items() if count < 0)}


def summarize(run):
    plan = run / "test-plan.json"
    plan_sha = file_sha(plan)
    frozen = read_json(plan)
    adapters = {arm: read_json(run / "train" / arm / "selected.json")["adapter_sha256"] for arm in ARMS[2:]}
    hashes = [case["initial"]["raw_sha256"] for case in frozen["cases"]]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Repeated independent test layout")
    native = []
    for case in frozen["cases"]:
        for seed in case["inference_seeds"]:
            arm_results = {arm: audit_episode(run, case, seed, arm, plan_sha, frozen["gpu_uuid"], adapters)
                           for arm in ARMS}
            if case["cohort"] == "challenge" and arm_results["rule"]["edits"][0]["step"] != case["first_v1_override"]:
                raise ValueError("Effect is from a different reference opportunity")
            native.append(arm_results)
    stats = {}
    for group in ("regular", "challenge", "overall"):
        subset = [trial for trial in native if group == "overall" or trial["base"]["cohort"] == group]
        stats[group] = {
            "arms": {arm: per_arm([trial[arm] for trial in subset]) for arm in ARMS},
            "contrasts": {
                f"{cand}_vs_{ref}": {metric: paired(subset, cand, ref, metric) for metric in ("success", "release")}
                for cand, ref in CONTRASTS
            },
        }
    overall = stats["overall"]["contrasts"]["corrected_vs_base"]["success"]
    regular = stats["regular"]["contrasts"]["corrected_vs_base"]["success"]
    criterion = len(overall["positive_layouts"]) >= 2 and overall["net"] > 0 and regular["losses"] == 0
    return {"test_plan_sha256": plan_sha, "independent_layouts": len(hashes),
            "episodes": len(native) * len(ARMS), "adapted_checkpoint_sha256": adapters,
            "meets_independent_success_criterion": criterion, "groups": stats, "pairs": native}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    write_json(args.run / "summary.json", result)
    print(json.dumps({name: value for name, value in result.items() if name != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
