"""Verify and score stage 4 closed-loop matched simulation on a CPU."""
import argparse
from collections import Counter
import json
from math import hypot, isfinite
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
sys.path.insert(0, str(V2))
from common import (BUDGET, CAMERAS, CHEESE, CHUNK, GOALS, LABELS, OFFSETS, PREFIX_TASK, SAUCE, SOUP,
                    array_hash, read, sha, write)


ARMS = ("baseline", "candidate")
FINGERPRINT = ("step", "raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256")
REQUIRED_OBSERVATIONS = {"eef_pos", "eef_quat", "gripper_qpos"} | {
    name + suffix for name in CAMERAS for suffix in ("", "_depth", "_intrinsic", "_extrinsic")
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def stamp(row):
    value = {key: row[key] for key in FINGERPRINT}
    require(type(value["step"]) is int and value["step"] >= 0 and
            all(hash_string(value[key]) for key in FINGERPRINT[1:4]) and
            set(value["observation_sha256"]) == REQUIRED_OBSERVATIONS and
            all(hash_string(h) for h in value["observation_sha256"].values()),
            f"Malformed step fingerprint: {value}")
    return value


def audit_snapshot(value):
    goals = value["goals"]
    require(set(goals) == set(GOALS) and all(type(b) is bool for b in goals.values())
            and type(value["held_soup"]) is bool, "Unexpected official goal or grasp audit schema")
    return goals


def load_frozen(root):
    path = root / "frozen.json"
    lock = read(path)
    plan_path = root / "plan.json"
    require(sha(plan_path) == lock["plan_sha256"], "Frozen plan checksum mismatch")
    for name, digest in lock["code"].items():
        source = (HERE / name).resolve()
        require(source.is_relative_to(V2) and sha(source) == digest, f"Frozen source differs: {source}")
    plan = read(plan_path)
    cases = plan["cases"]
    phase = plan["phase"]
    require(phase in ("smoke", "pilot"), f"Unexpected audit phase: {phase}")
    require(plan["budget"] == BUDGET and plan["policy_chunk"] == CHUNK
            and plan["observer_stride"] == 5 and tuple(plan["arms"]) == ARMS,
            "Frozen budget, cadence, or arm assignment differs from protocol")
    if "cases" in lock:
        require(lock["cases"] == cases, "Case list disagrees with freeze")
    expected = ((1, 1) if phase == "smoke" else (8, 4))
    require((sum(case["moved"] is True for case in cases), sum(case["moved"] is False for case in cases))
            == expected and len(cases) == sum(expected), f"Unexpected {phase} condition allocation")
    require(len({c["layout_seed"] for c in cases}) == len({c["tag"] for c in cases}) == len(cases),
            "Duplicate layout or case tag")
    for case in cases:
        require(type(case["layout_seed"]) is int and type(case["policy_seed"]) is int
                and case["card"] in (2, 3) and sorted(case["order"]) == sorted(ARMS)
                and Path(case["tag"]).name == case["tag"] and case["tag"] not in (".", ".."),
                f"Invalid frozen case: {case}")
    return lock, plan


def verify_execution_manifests(root, plan):
    freeze_sha = sha(root / "frozen.json")
    for card in (2, 3):
        manifest = read(root / f"run-{card}.json")
        cases = [case for case in plan["cases"] if case["card"] == card]
        expected = [(case["tag"], name) for case in cases for name in case["order"]]
        require(manifest["complete"] is True and manifest["card"] == card and manifest["phase"] == plan["phase"]
                and manifest["freeze_sha256"] == freeze_sha and manifest["planned"] == [c["tag"] for c in cases]
                and [(r["tag"], r["arm"]) for r in manifest["episodes"]] == expected,
                f"Incomplete or reordered execution manifest on card {card}")
        for r in manifest["episodes"]:
            path = root / "episodes" / r["tag"] / r["arm"] / "result.json"
            require(sha(path) == r["result_sha256"], f"Result changed after execution: {path}")


def load_prefix(root, case):
    fixture_folder = root / "fixtures" / str(case["layout_seed"])
    prefix_path = fixture_folder / "prefix.json"
    fixture_path = fixture_folder / "fixture.json"
    require(sha(prefix_path) == case["prefix_sha256"] and sha(fixture_path) == case["fixture_sha256"],
            f"Frozen fixture differs: {fixture_folder}")
    prefix, fixture = read(prefix_path), read(fixture_path)
    require(fixture["eligible"] is True and fixture["layout_seed"] == case["layout_seed"]
            and prefix["observed_cheese"] is True and "failure" not in prefix
            and stamp(prefix["end"]) == stamp(fixture["prefix"]) and stamp(prefix["initial"])["step"] == 0
            and prefix["initial"]["state_sha256"] == case["initial_state_sha256"]
            and prefix["initial"]["raw_sha256"] == case["raw_sha256"],
            f"Invalid common prefix: {fixture_folder}")
    if case["moved"]:
        offset = fixture["offset"]
        require(tuple(offset) in OFFSETS and .119 <= hypot(*offset) <= .121 and
                fixture["perturbed_checkpoint"]["step"] == prefix["end"]["step"] and
                fixture["perturbed_checkpoint"]["state_sha256"] != prefix["end"]["state_sha256"],
                f"Invalid frozen displaced soup state: {fixture_folder}")
    else:
        offset = None
    if "offset" in case:
        require(case["offset"] == offset, f"Case and fixture disagree on displacement: {fixture_folder}")
    return prefix, fixture, offset


def trace_rows(directory, arm, prefix):
    action_path = directory / "actions.jsonl"
    require(sha(action_path) == arm["actions_sha256"], f"Action file checksum mismatch: {directory}")
    rows = []
    initial = stamp(prefix["initial"])
    last = 0
    goals = dict.fromkeys(GOALS, False)
    with action_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step = stamp(row)
            n = len(row["actions"])
            require(row["start_step"] == last and step["step"] == last + n
                    and 0 < n <= CHUNK and row["executed"] == n and step["step"] <= BUDGET
                    and step["raw_sha256"] == initial["raw_sha256"]
                    and step["bddl_sha256"] == initial["bddl_sha256"], f"Invalid control interval: {directory}:{last}")
            # Frozen validate_actions only requires finite realized 7D values.
            require(all(isinstance(action, list) and len(action) == 7 and
                        all(type(a) in (float, int) and isfinite(a) for a in action)
                        for action in row["actions"]), f"Nonfinite or invalid OSC action: {directory}:{last}")
            require(type(row["completed_regrasp_steps"]) is int and 0 <= row["completed_regrasp_steps"] <= n and
                    type(row["done"]) is bool, f"Malformed per-step counters: {directory}:{last}")
            seen = set()
            before = last
            for event in row["transitions"]:
                when, obj, value = event["step"], event["object"], event["in_basket"]
                require(type(when) is int and last < when <= step["step"] and before <= when
                        and obj in GOALS and type(value) is bool and (when, obj) not in seen
                        and goals[obj] is not value, f"Invalid goal transition: {directory}:{event}")
                before = when
                seen.add((when, obj))
                goals[obj] = value
            require(goals == audit_snapshot(row["audit"]), f"Action transitions and final predicates disagree: {directory}:{last}")
            rows.append(row)
            last = step["step"]
    k = len(prefix["chunks"])
    require(len(rows) > k and k > 0, f"No post-fork controls: {directory}")
    for expected, actual in zip(prefix["chunks"], rows[:k], strict=True):
        require(actual["target"] == CHEESE and actual["instruction"] == PREFIX_TASK
                and actual["actions"] == expected["actions"] and stamp(actual) == stamp(expected["end"]),
                f"Executed prefix deviates from the frozen prefix: {directory}:{actual['step']}")
    require(audit_snapshot(rows[k - 1]["audit"]) == {o: o == CHEESE for o in GOALS}, f"Incorrect prefix goals: {directory}")
    suffix = rows[k:]
    for index, row in enumerate(suffix):
        target = row["target"]
        require(target in (SOUP, SAUCE) and row["instruction"] == f"put the {LABELS[target]} in the basket"
                and (len(row["actions"]) == 5 or index == len(suffix) - 1),
                f"Incorrect suffix target, prompt or cadence: {directory}:{row['step']}")
    return rows, suffix


def recalculate_events(entries, by_step, kind, directory):
    correct, wrong, commits = [], [], []
    for event in entries:
        if event["event"] != "in_basket":
            continue
        step, obj = event["step"], event["object"]
        require(step in by_step and obj in GOALS and (kind != "sidecar" or obj == SOUP),
                f"Observer confirmation has no independently audited control boundary: {directory}:{event}")
        audit = by_step[step]["audit"]
        held = audit["held_soup"] if obj == SOUP else None
        valid = audit["goals"][obj] and (held is not True)
        require(event["matches_audit"] is audit["goals"][obj] and event["confirmed_in_audit"] is valid
                and event["held_soup"] is audit["held_soup"] and type(event["applied"]) is bool,
                f"Placement logged truth disagrees with independent audit: {directory}:{step}")
        record = {"step": step, "object": obj, "held_soup": held, "valid": valid,
                  "source": event.get("source"), "writer": event.get("writer"), "applied": event.get("applied")}
        if kind == "sidecar":
            name = event["evidence_file"]
            require(name == f"soup-commit-{step}.npz", f"Unexpected proof path: {directory}:{step}")
            proof = directory / name
            hash_observation(proof, by_step[step]["observation_sha256"])
            record["evidence_sha256"] = sha(proof)
        (correct if valid else wrong).append(record)
        if event.get("applied"):
            commits.append(record)
    return {"correct": correct, "wrong": wrong, "commits": commits}


def checked_policy_chunks(chunks, suffix, before, directory):
    by_last = {row["step"]: i for i, row in enumerate(suffix)}
    offset = 0
    previous = before
    expected_reassign = []
    for chunk in chunks:
        start = chunk["start_step"]
        finish = chunk["end_step"]
        end = by_last.get(finish, -1)
        require(offset <= end and 1 <= end - offset + 1 <= 2 and previous["step"] == start
                and suffix[offset]["start_step"] == start and 1 <= finish - start <= CHUNK
                and (finish - start == CHUNK or end == len(suffix) - 1),
                f"Inference/action block disagreement: {directory}:{start}")
        acted = suffix[offset:end + 1]
        target = acted[0]["target"]
        require(all(row["target"] == target for row in acted)
                and chunk["target"] == target and chunk["requested"] == min(CHUNK, BUDGET - start)
                and chunk["executed"] == finish - start and chunk["sensor_steps"] == [r["step"] for r in acted],
                f"Target switched within an already-made inference: {directory}:{start}")
        if previous["audit"]["goals"][target]:
            expected_reassign.append({"step": start, "target": target})
        previous = acted[-1]
        offset = end + 1
    require(offset == len(suffix), f"Post-fork controls missing policy chunk correspondence: {directory}")
    return expected_reassign


def hash_observation(path, recorded):
    with np.load(path, allow_pickle=False) as npz:
        require(set(npz) == set(recorded) and all(array_hash(npz[k]) == h for k, h in recorded.items()),
                f"Observation arrays not reproduced: {path}")


def audit_arm(directory, name, case, prefix, fixture, offset):
    arm = read(directory / "result.json")
    require(arm["arm"] == name and all(arm[k] == case[k] for k in
                                      ("moved", "layout_seed", "policy_seed", "prefix_sha256", "fixture_sha256"))
            and arm["method"] == "replan", f"Unexpected assignment: {directory}")
    if "offset" in arm:
        require(arm["offset"] == offset, f"Wrong applied offset: {directory}")
    require(stamp(arm["initial"]) == stamp(prefix["initial"]) and
            stamp(arm["base_checkpoint"]) == stamp(prefix["end"]) and
            stamp(arm["trial_checkpoint"]) == stamp(fixture["perturbed_checkpoint"] if case["moved"] else prefix["end"]),
            f"Initial/prefix/post-perturbation mismatch: {directory}")
    fork = directory / "fork.npz"
    require(sha(fork) == arm["fork_sha256"], f"Fork file bytes differ: {directory}")
    hash_observation(fork, arm["trial_checkpoint"]["observation_sha256"])
    rows, suffix = trace_rows(directory, arm, prefix)
    goals = rows[-1]["audit"]["goals"]
    success = all(goals.values())
    terminal_reason = arm["terminal_reason"]
    require(terminal_reason in ("goal", "budget", "observed_complete"), f"Unknown terminal reason: {directory}")
    require(success == arm["success"] and goals == arm["terminal_goals"] and rows[-1]["step"] == arm["steps"]
            and arm["after_fork_steps"] == arm["steps"] - prefix["end"]["step"]
            and arm["terminal_audit"] == rows[-1]["audit"] and
            stamp(arm["terminal_checkpoint"]) == stamp(rows[-1]) and
            ((terminal_reason == "goal") == success), f"Summary mismatches official terminal predicate: {directory}")
    if terminal_reason == "budget":
        require(arm["steps"] == BUDGET or rows[-1]["done"], f"Premature stop reported as exhausted budget: {directory}")
    if terminal_reason == "observed_complete":
        require(all(v == "in_basket" for v in arm["observed_terminal"].values()),
                f"Bad observed-complete stop: {directory}")
    by_step = {row["step"]: row for row in suffix}
    wrong_progress = recalculate_events(arm["state_events"], by_step, "progress", directory)
    sidecar = recalculate_events(arm["soup_events"], by_step, "sidecar", directory)
    progress_soup = {(x["step"], x["source"]) for x in wrong_progress["correct"] + wrong_progress["wrong"] if x["object"] == SOUP}
    committed_soup = {(x["step"], x["source"]) for x in sidecar["commits"]}
    require((not committed_soup if name == "baseline" else progress_soup == committed_soup),
            f"Soup state writes escaped the assigned observer: {directory}")
    regrasp = sum(r["completed_regrasp_steps"] for r in suffix)
    require(regrasp == arm["completed_regrasp_steps"], f"Regrasp metric does not reproduce: {directory}")
    reassigned = checked_policy_chunks(arm["policy_chunks"], suffix, rows[len(prefix["chunks"]) - 1], directory)
    require(reassigned == arm["completed_reassigned_chunks"], f"Reassignment metric does not reproduce: {directory}")
    digest = arm["actions_sha256"]
    output = {"success": success, "steps": arm["steps"], "after_fork_steps": arm["after_fork_steps"],
              "terminal_reason": terminal_reason, "terminal_goals": goals,
              "false_observed_complete": terminal_reason == "observed_complete" and not success,
              "wrong_confirmations": wrong_progress["wrong"], "wrong_sidecar_confirmations": sidecar["wrong"],
              "raw_confirmation_evidence": sidecar["correct"] + sidecar["wrong"],
              "soup_writes": sorted(progress_soup), "candidate_raw_confirmations": len(sidecar["correct"] + sidecar["wrong"]),
              "completed_regrasp_steps": regrasp, "completed_reassigned_chunks": len(reassigned),
              "schedule_changes": arm["schedule_changes"], "target_runs": arm["target_runs"],
              "actions_sha256": digest, "fork_sha256": arm["fork_sha256"], "result_sha256": sha(directory / "result.json"),
              "policy_calls": len(arm["policy_chunks"])}
    return output, suffix


def divergence(left, right):
    def control(rows):
        return [(r["start_step"] + i + 1, r["target"], a)
                for r in rows for i, a in enumerate(r["actions"])]
    left_a, right_a = control(left), control(right)
    initial = min(left_a[0][0], right_a[0][0])
    delta = next((min(a[0], b[0]) for a, b in zip(left_a, right_a) if a != b), None)
    if delta is None and len(left_a) != len(right_a):
        delta = initial + min(len(left_a), len(right_a))
    if delta is None:
        prefix_len = len(left_a)
    else:
        prefix_len = delta - initial
    right_by_step = {r["step"]: r for r in right}
    for row in left:
        step = row["step"]
        if step >= initial + prefix_len:
            break
        if step in right_by_step:
            require(stamp(row) == stamp(right_by_step[step]),
                    f"Paired observation/physics diverged before controls at step {step}")
    return delta


def summarize(cases, phase):
    results = {}
    for group in ("moved", "sham", "all"):
        rows = [r for r in cases if group == "all" or r["condition"] == group]
        counter = Counter("win" if r["arms"]["candidate"]["success"] and not r["arms"]["baseline"]["success"] else
                          "loss" if r["arms"]["baseline"]["success"] and not r["arms"]["candidate"]["success"] else
                          "both_success" if r["arms"]["baseline"]["success"] else "both_failure" for r in rows)
        performance = {}
        for arm in ARMS:
            selected = [r["arms"][arm] for r in rows]
            delays = [r["after_fork_steps"] for r in selected if r["success"]]
            performance[arm] = {
                "successes": len(delays), "episodes": len(selected),
                "median_steps_after_fork_if_successful": median(delays) if delays else None,
                "wrong_confirmations": sum(len(r["wrong_confirmations"]) for r in selected),
                "wrong_candidate_observations": sum(len(r["wrong_sidecar_confirmations"]) for r in selected),
                "false_observed_complete": sum(r["false_observed_complete"] for r in selected),
                "completed_regrasp_steps": sum(r["completed_regrasp_steps"] for r in selected),
                "episodes_with_completed_regrasp": sum(r["completed_regrasp_steps"] > 0 for r in selected),
                "completed_reassigned_chunks": sum(r["completed_reassigned_chunks"] for r in selected),
                "terminal_reasons": dict(Counter(r["terminal_reason"] for r in selected)),
            }
        results[group] = {"arms": performance, "pairs": len(rows),
                          "candidate_wins": counter["win"], "candidate_losses": counter["loss"],
                          "both_success": counter["both_success"], "both_failure": counter["both_failure"],
                          "differing_controls": sum(r["first_differing_control_step"] is not None for r in rows)}
    totals = results["all"]["arms"]
    safety = (totals["candidate"]["wrong_confirmations"] == 0
              and totals["candidate"]["false_observed_complete"] == 0)
    shadow_safety = all(row["wrong_candidate_observations"] == 0 for row in totals.values())
    no_sham_backslide = results["sham"]["candidate_losses"] == 0
    moved_gain = results["moved"]["candidate_wins"] - results["moved"]["candidate_losses"]
    return {"conditions": results, "gate": {
        "zero_wrong_or_premature_candidate_confirmations": safety,
        "zero_wrong_candidate_observations_on_both_arms": shadow_safety,
        "no_unperturbed_success_backslide": no_sham_backslide, "net_moved_wins": moved_gain,
        "smoke_release_pilot": phase == "smoke" and safety and no_sham_backslide,
        "pilot_expand_formal": (phase == "pilot" and safety and no_sham_backslide
                                 and results["moved"]["candidate_wins"] >= 2 and moved_gain > 0),
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Sealed phase-specific output root")
    parser.add_argument("--out", type=Path, help="New result path, default <run>/audit4.json")
    args = parser.parse_args()
    root = args.run.resolve()
    out = args.out or root / "audit4.json"
    require(not out.exists(), f"Refusing to overwrite pre-existing audit: {out}")
    _, plan = load_frozen(root)
    verify_execution_manifests(root, plan)
    cases = []
    hashes = set()
    for case in plan["cases"]:
        prefix, fixture, offset = load_prefix(root, case)
        require(prefix["initial"]["bddl_sha256"] == plan["bddl_sha256"],
                f"Fixture does not match the excluded task definition: {case['tag']}")
        initial = (prefix["initial"]["state_sha256"], prefix["initial"]["raw_sha256"])
        require(not any(a == initial[0] or b == initial[1] for a, b in hashes), "Repeated real initial layout")
        hashes.add(initial)
        arms, traces = {}, {}
        for method in ARMS:
            directory = root / "episodes" / case["tag"] / method
            arms[method], traces[method] = audit_arm(directory, method, case, prefix, fixture, offset)
        first = divergence(traces["baseline"], traces["candidate"])
        if first is not None:
            # No difference is expected before a progress-writing difference
            # can reach the next policy decision in either arm.
            writes = {step for step, _ in arms["baseline"]["soup_writes"]} ^ {
                step for step, _ in arms["candidate"]["soup_writes"]}
            require(any(step < first for step in writes),
                    f"Control diverged before the observer treatment could change a policy decision: {case['tag']}")
        cases.append({"case": case, "condition": "moved" if case["moved"] else "sham",
                      "offset": offset, "first_differing_control_step": first,
                      "initial_state_sha256": initial[0], "arms": arms})
    result = {"schema": "tableclear-stage4-audit-v1", "phase": plan["phase"],
              "plan_sha256": sha(root / "plan.json"), "freeze_sha256": sha(root / "frozen.json"),
              "audit_sha256": sha(Path(__file__)), "episodes": 2 * len(cases), "distinct_layouts": len(cases),
              **summarize(cases, plan["phase"]), "cases": cases}
    write(out, result)
    print(json.dumps({k: value for k, value in result.items() if k != "cases"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
