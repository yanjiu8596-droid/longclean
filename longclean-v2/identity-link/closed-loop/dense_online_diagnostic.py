"""Read-only causal-chain diagnostics after all twelve online arms are audited."""
import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER_SHA = "6de3c9241ad9f592a5c04639e022bcb623f3ee98d5368d7670813b2cd7a2654f"
AUDIT_SHA = "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4"
LAYOUTS = tuple(range(931300, 931304))
ARMS = ("baseline", "actual10", "adaptive1")
CHEESE, SOUP, SAUCE = "cream_cheese_1", "alphabet_soup_1", "tomato_sauce_1"
GOALS = (CHEESE, SOUP, SAUCE)
PAIRINGS = (("actual10", "adaptive1"), ("baseline", "adaptive1"))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(root, name, digest):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()) and path.is_file()
            and sha(path) == digest, f"Changed or unsafe evidence: {name}")
    return path


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_arm(root, entry):
    path = checked(root, entry["result"], entry["result_sha256"])
    result = read(path)
    require(result["layout_seed"] == entry["layout_seed"] and result["arm"] == entry["arm"], "Audited arm identity changed")
    native = jsonl(checked(path.parent, "actions.jsonl", result["actions_sha256"]))
    samples = jsonl(checked(path.parent, "grasp-audit.jsonl", result["grasp_audit_sha256"]))
    controls, endpoint = [], {}
    for row in native:
        require(row["start_step"] == len(controls) and row["step"] == len(controls) + len(row["actions"]), "Noncontiguous controls")
        controls.extend(row["actions"])
        endpoint[row["step"]] = row
    require(len(endpoint) == len(native) and len(controls) == result["terminal"]["step"], "Duplicate or missing native row")
    physical = {r["step"]: r for r in samples}
    require(len(physical) == len(samples) and list(physical) == [0, *endpoint], "Physical sampling alignment differs")
    for step, row in endpoint.items():
        require(row["state_sha256"] == physical[step]["state_sha256"] and row["audit"]["goals"] == physical[step]["goals"],
                "Physical sample does not describe native state")
    commits = [e for e in result["prefix_events"] + result["state_events"] if e["applied"]]
    for event in commits:
        sample, obj = physical[event["step"]], event["object"]
        valid = bool(sample["goals"][obj] and not sample["held_by_target"][obj])
        require(event["physically_confirmed"] == valid, "Audited commit correctness changed")
    return {"result": result, "entry": entry, "native": native, "endpoint": endpoint, "controls": controls,
            "physical": physical, "commits": commits,
            "chunks": {c["start_step"]: c for c in result["policy_chunks"]},
            "decisions": {d["step"]: d for d in result["schedule_decisions"]}}


def first_control_difference(a, b):
    return next((i for i, (left, right) in enumerate(zip(a["controls"], b["controls"]), 1) if left != right), None)


def inputs_at(a, b, step):
    left, right = a["endpoint"][step], b["endpoint"][step]
    return {"step": step, "state_sha256_equal": left["state_sha256"] == right["state_sha256"],
            "all_observation_hashes_equal": left["observation_sha256"] == right["observation_sha256"],
            "rgb_hashes_equal": all(left["observation_sha256"][c] == right["observation_sha256"][c] for c in ("camera1", "camera2")),
            "left_state_sha256": left["state_sha256"], "right_state_sha256": right["state_sha256"],
            "left_observation_sha256": left["observation_sha256"], "right_observation_sha256": right["observation_sha256"]}


def chunk_difference(a, b, start=0):
    for step in sorted(set(a["chunks"]) & set(b["chunks"])):
        if step < start:
            continue
        left, right = a["chunks"][step], b["chunks"][step]
        if left["issued_actions"] == right["issued_actions"]:
            continue
        require(len(left["issued_actions"]) == len(right["issued_actions"]) == 10, "Unexpected issued chunk length")
        delta = [float(y) - float(x) for la, ra in zip(left["issued_actions"], right["issued_actions"], strict=True)
                 for x, y in zip(la, ra, strict=True)]
        observation = inputs_at(a, b, step)
        return {"start_step": step, "left_target": left["target"], "right_target": right["target"],
                "left_prompt": left["prompt"], "right_prompt": right["prompt"], "prompt_changed": left["prompt"] != right["prompt"],
                "issued_actions_l2": math.sqrt(sum(v * v for v in delta)), "issued_actions_max_abs": max(map(abs, delta)),
                "common_input": observation["state_sha256_equal"] and observation["all_observation_hashes_equal"],
                "input_comparison": observation}
    return None


def valid_commits_before(arm, step):
    return [{k: e[k] for k in ("step", "object", "assigned", "physically_confirmed", "held_by_target", "matches_audit")}
            for e in arm["commits"] if e["step"] <= step and e["step"] > arm["result"]["fork"]["step"]
            and e["physically_confirmed"]]


def outcomes_after(arm, step):
    """Native goal changes are per-control; grasp changes are sampled endpoints only."""
    result, physical = arm["result"], arm["physical"]
    require(step in physical, "Diagnostic reference is not an observed endpoint")
    previous_step, previous = step, physical[step]["held_by_target"]
    grasp_rises = []
    for when, row in physical.items():
        if when <= step:
            continue
        new = [obj for obj in GOALS if row["held_by_target"][obj] and not previous[obj]]
        if new:
            grasp_rises.append({"sample_step": when, "previous_sample_step": previous_step,
                                "newly_sampled_held_objects": new, "goals_at_sample": row["goals"]})
        previous_step, previous = when, row["held_by_target"]
    transitions = [t for r in arm["native"] for t in r["transitions"]]
    objects = {}
    for obj in (SOUP, SAUCE):
        entered = [t["step"] for t in transitions if t["object"] == obj and t["in_basket"]]
        lost = [t["step"] for t in transitions if t["object"] == obj and not t["in_basket"]]
        released = [s for s, r in physical.items() if r["goals"][obj] and not r["held_by_target"][obj]]
        objects[obj] = {"official_goal_true_at_reference": physical[step]["goals"][obj],
                        "official_entry_steps_entire_episode": entered, "official_entry_steps_after_reference": [s for s in entered if s > step],
                        "official_reversal_steps_entire_episode": lost, "official_reversal_steps_after_reference": [s for s in lost if s > step],
                        "first_sampled_goal_true_and_unheld_entire_episode": next(iter(released), None),
                        "sampled_goal_true_and_unheld_at_reference": physical[step]["goals"][obj] and not physical[step]["held_by_target"][obj],
                        "first_sampled_goal_true_and_unheld_after_reference": next((s for s in released if s > step), None)}
    reversals = []
    for event in arm["commits"]:
        lost = [t["step"] for t in transitions if t["object"] == event["object"] and not t["in_basket"] and t["step"] > event["step"]]
        reversals.append({"object": event["object"], "commit_step": event["step"], "physically_confirmed_at_commit": event["physically_confirmed"],
                          "later_official_false_steps": lost, "terminal_official_goal": result["terminal_native_audit"]["goals"][event["object"]]})
    return {"reference_step": step, "held_at_reference": physical[step]["held_by_target"],
            "first_sampled_new_grasp": next(iter(grasp_rises), None), "sampled_new_grasps": grasp_rises,
            "object_outcomes": objects, "committed_object_reversals": reversals,
            "terminal_step": result["terminal"]["step"], "terminal_reason": result["terminal_reason"],
            "terminal_goals": result["terminal_native_audit"]["goals"], "terminal_held": result["terminal_held_by_target"],
            "terminal_progress": result["terminal_progress"], "qualified_success": result["qualified_success"],
            "completed_reassigned_chunks": result["completed_reassigned_chunks"],
            "sampled_grasp_caveat": "False-to-true between stored samples, not the exact first grasp control step; capture cadence differs by arm."}


def compare(a, b):
    left, right = a["result"], b["result"]
    require(left["layout_seed"] == right["layout_seed"] and left["policy_seed"] == right["policy_seed"]
            and left["initial"] == right["initial"] and left["fork"] == right["fork"]
            and left["prefix_sha256"] == right["prefix_sha256"] and left["fork_npz_sha256"] == right["fork_npz_sha256"], "Unpaired input fork")
    fork = left["fork"]["step"]
    common = sorted(set(a["decisions"]) & set(b["decisions"]))
    require(all(s % 10 == 0 for s in common), "Non-original target decision boundary")
    different = [s for s in common if a["decisions"][s]["target"] != b["decisions"][s]["target"]]
    first = next(iter(different), None)
    changed = first_control_difference(a, b)
    proof = None
    if first is not None:
        ldec, rdec = a["decisions"][first], b["decisions"][first]
        inputs = inputs_at(a, b, first)
        prefix_equal = a["controls"][:first] == b["controls"][:first]
        commits = valid_commits_before(b, first)
        removed = [e for e in commits if e["object"] in ldec["order"] and e["object"] not in rdec["order"]]
        lchunk, rchunk = a["chunks"].get(first), b["chunks"].get(first)
        proof = {"step": first, "left_decision": ldec, "right_decision": rdec,
                 "left_valid_precommits": valid_commits_before(a, first), "right_valid_precommits": commits,
                 "right_correctly_removed_objects_still_in_left_order": removed, "exact_executed_controls_before_decision_equal": prefix_equal,
                 "inputs": inputs, "right_correct_progress_change_with_common_input": bool(removed and prefix_equal
                     and inputs["state_sha256_equal"] and inputs["all_observation_hashes_equal"]),
                 "issued_chunk_present_both": lchunk is not None and rchunk is not None,
                 "issued_chunk_identical_at_target_change": None if lchunk is None or rchunk is None else lchunk["issued_actions"] == rchunk["issued_actions"],
                 "left_prompt": None if lchunk is None else lchunk["prompt"], "right_prompt": None if rchunk is None else rchunk["prompt"]}
    reference = first if first is not None else fork
    return {"layout_seed": left["layout_seed"], "left": left["arm"], "right": right["arm"],
            "shared_fork": left["fork"], "fork_npz_sha256": left["fork_npz_sha256"], "shared_policy_seed": left["policy_seed"],
            "common_decision_steps_checked": common, "selected_target_difference_steps": different,
            "first_target_difference": proof, "first_differing_executed_control_step": changed,
            "common_executed_horizon": min(len(a["controls"]), len(b["controls"])),
            "first_different_issued_chunk": chunk_difference(a, b),
            "first_different_issued_chunk_at_or_after_target_change": None if first is None else chunk_difference(a, b, first),
            "reference_kind": "first_selected_target_difference" if first is not None else "shared_fork_no_target_difference",
            "left_outcomes": outcomes_after(a, reference), "right_outcomes": outcomes_after(b, reference),
            "net_qualified_success_right_minus_left": int(right["qualified_success"]) - int(left["qualified_success"]),
            "inputs": {arm["result"]["arm"]: {k: arm["entry"][k] for k in ("result", "result_sha256", "actions_sha256", "grasp_audit_sha256")}
                       for arm in (a, b)}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "audit", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--audit-sha", required=True)
    args = cli.parse_args()
    root, out = args.run.resolve(), args.out.resolve()
    require(not out.exists() and sha(args.audit) == args.audit_sha == AUDIT_SHA and sha(HERE / "dense_online.py") == RUNNER_SHA,
            "Existing output, changed final audit or changed frozen runner")
    audit = read(args.audit)
    require(audit["schema"] == "v2-dense-online-audit/1" and audit["runner_sha256"] == RUNNER_SHA
            and audit["audit_sha256"] == sha(HERE / "dense_online_audit.py")
            and audit["all_four_unique_initials_verified"] is True and all(audit["prefix_eligible"].values())
            and len(audit["cards"]) == 2 and all(r["complete"] for r in audit["cards"])
            and audit["adaptive_safety_stop"] is None and audit["interrupted_records"] == 0
            and len(audit["records"]) == 12
            and {(r["layout_seed"], r["arm"]) for r in audit["records"]} == {(s, a) for s in LAYOUTS for a in ARMS}
            and all(r.get("result") is not None for r in audit["records"]), "Diagnostic requires all twelve fully audited, uninterrupted rollouts")
    for card in audit["cards"]:
        checked(root, f"dense-online-{card['card']}.json", card["manifest_sha256"])
        checked(root, f"dense-online-plan-{card['card']}.json", card["plan_sha256"])
    arms = {(r["layout_seed"], r["arm"]): load_arm(root, r) for r in audit["records"]}
    comparisons = [compare(arms[layout, left], arms[layout, right]) for left, right in PAIRINGS for layout in LAYOUTS]
    summaries = []
    for left, right in PAIRINGS:
        pairs = [r for r in comparisons if r["left"] == left and r["right"] == right]
        summaries.append({"left": left, "right": right, "role": "primary" if left == "actual10" else "secondary",
            "layouts_with_selected_target_difference": [r["layout_seed"] for r in pairs if r["first_target_difference"] is not None],
            "layouts_with_verified_right_progress_target_change": [r["layout_seed"] for r in pairs
                if r["first_target_difference"] is not None and r["first_target_difference"]["right_correct_progress_change_with_common_input"]],
            "net_qualified_success_right_minus_left": sum(r["net_qualified_success_right_minus_left"] for r in pairs)})
    output = {"schema": "v2-dense-online-diagnostic/1", "source_sha256": sha(Path(__file__)), "audit_sha256": args.audit_sha,
              "auditor_source_sha256": audit["audit_sha256"], "runner_sha256": RUNNER_SHA,
              "scope": "Read-only diagnosis of four fully audited development layouts. No threshold changes, policy feedback, or new execution.",
              "comparisons": comparisons, "branch_evidence": summaries, "independent_pilot_release": False,
              "interpretation_limits": ["A native goal entry may precede physical release; report sampled goal-true-and-unheld separately.",
                  "Action-vector differences after state divergence do not isolate prompt effects.",
                  "Absence of sampled grasp is not proof that no brief grasp occurred between samples."]}
    with out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print("DENSE_ONLINE_DIAGNOSTIC", summaries, flush=True)


if __name__ == "__main__":
    main()
