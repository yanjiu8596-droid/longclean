"""Audit the fixed-action diagnostic for progress on an unrequested object."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from target_follow_audit import (BDDL, CHUNK, GOALS, PROMPTS, SAUCE, SOUP, STRIDE, WINDOW, audit_trace, checked_path,
                                expected_rows, matches, observation, read, require, sha, stamp, write)
from common import ATTEMPT, BUDGET, CHEESE


HERE = Path(__file__).resolve().parent
PLACEMENT = HERE.parents[1] / "placement-next"
SOURCE_AUDIT = "0d00abc23a0264bb783084332b41ea91e538d76e6e992dad8f836d08592c2be3"
OBSERVER_FREEZE = "b3e35eeb53b784132a42260c3b378a91be49b6539c6fabe4029e216d577e8994"
OBSERVER_SHA = "81d6b3b34f1e968093f3396a99d76d60eea3905e77b58f222dbc920134624dd2"
COMPARE_SHA = "7bc4215e57310facb19a2f3e2ac3a548ebeb904a1d2549f385f034d2171d499e"
PHYSICS_WORKER = "be5e073d13e30ad0fb0e0faaf9c4897404d4c80eddf8086ac65fd45a9db5cbff"
PHYSICS_SIMULATOR = "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8"
CASES = {2: (931000, 931200), 3: (931005, 931201)}
ARMS = ("baseline", "candidate")
OBJECTS = (SOUP, SAUCE)
STABILITY = 20


def check_prior(source):
    prior = read(checked_path(source, "audit.json", SOURCE_AUDIT))
    freeze = read(checked_path(PLACEMENT, "candidate-freeze.json", OBSERVER_FREEZE))
    require(prior["schema"] == "v2-stage4-target-follow-audit/1" and prior["scope"] == "development_only"
            and sha(HERE / "target_follow_audit.py") == prior["audit_sha256"]
            and sha(HERE / "target_follow.py") == prior["runner_sha256"]
            and all(sha(HERE / p) == h for p, h in prior["frozen_native_source_sha256"].items())
            and freeze["source"]["observer.py"] == sha(PLACEMENT / "observer.py") == OBSERVER_SHA
            and freeze["source"]["compare.py"] == sha(PLACEMENT / "compare.py") == COMPARE_SHA
            and sha(HERE / "actual_physics_worker.py") == PHYSICS_WORKER
            and sha(HERE / "actual_physics_simulator.py") == PHYSICS_SIMULATOR,
            "Frozen target-follow or placement-observer source changed")
    return prior


def source_inputs(source, prior, card):
    lock, = [m for m in prior["manifests"] if m["card"] == card]
    plan = read(checked_path(source, f"target-follow-plan-{card}.json", lock["plan_sha256"]))
    manifest = read(checked_path(source, f"target-follow-{card}.json", lock["manifest_sha256"]))
    require(matches(plan, {"card": card, "budget": WINDOW, "policy_chunk": CHUNK, "observer_stride": STRIDE,
                           "motion": False, "bddl_sha256": BDDL}) and plan["cases"] == expected_rows(card)
            and manifest["complete"] is True and manifest["plan_sha256"] == lock["plan_sha256"],
            f"Changed sealed target-follow plan: card {card}")
    return lock, manifest


def source_case(source, prior, manifest, card, layout):
    expected, = [r for r in expected_rows(card) if r["layout_seed"] == layout and r["prefix"]
                 and r["prompt_id"] == "standard"]
    row, = [r for r in prior["records"] if matches(r, expected)]
    item, = [r for r in manifest["records"] if matches(r, expected)]
    setup, = [s for s in manifest["prefixes"] if s["layout_seed"] == layout]
    name = f"{layout}/cheese-prefix/standard/result.json"
    require(row["status"] == item["status"] == "complete" and row["result"] == item["result"] == name
            and row["result_sha256"] == item["result_sha256"] and setup["available"] is True
            and row["category"] == "sauce_first", f"Unrecognized target-follow source result for {layout}")
    result_path = checked_path(source, name, row["result_sha256"])
    r = read(result_path)
    p = read(checked_path(source, f"{layout}/prefix-setup/prefix.json", setup["sha256"]))
    require(matches(r, expected) and r["status"] == "complete" and r["motion"] is False
            and r["budget"] == r["suffix_steps"] == WINDOW and r["terminal_reason"] == "budget"
            and r["prefix_record"] == setup["prefix"] == f"{layout}/prefix-setup/prefix.json"
            and r["prefix_sha256"] == setup["sha256"] and r["initial"] == setup["initial"] == p["initial"] == row["initial"]
            and r["fork"] == setup["end"] == p["end"] == row["fork"] and r["terminal_checkpoint"] == row["terminal"]
            and r["actions"] == "actions.jsonl" and r["actions_sha256"] == row["actions_sha256"]
            and r["fork_npz_sha256"] == row["fork_npz_sha256"] and "failure" not in p
            and setup["observed_cheese"] is p["observed_cheese"] is True,
            f"Sealed action record, prefix, or checkpoints differ: {layout}")
    action_path = checked_path(result_path.parent, "actions.jsonl", r["actions_sha256"])
    observation(checked_path(result_path.parent, "fork.npz", r["fork_npz_sha256"]), stamp(r["fork"]))
    first, tail, group, _, suffix = audit_trace(action_path, expected, stamp(r["initial"]), stamp(r["fork"]), p)
    require(first == row["first_entry_steps"] and first[SAUCE] < first.get(SOUP, BUDGET + 1)
            and not tail["done"] and not all(tail["audit"]["goals"].values())
            and tail["audit"]["goals"] == r["terminal_goals"] == row["terminal_goals"]
            and group == r["model_chunks"] and len(suffix) == WINDOW // STRIDE,
            f"Sealed source truth or full suffix length differs: {layout}")
    rows = read_rows(action_path)
    require(rows[-len(suffix):] == suffix, f"Raw suffix changed after parsing: {layout}")
    return {"audit_row": row, "expected": expected, "original": r, "prefix": p,
            "prefix_sha256": setup["sha256"], "rows": rows, "suffix": suffix, "first": first}


def replay_schedule(writes, rows, fork, end):
    """Recompute the normally called 10-step schedule, once the cheese prefix is complete."""
    order, current, until, completed, out = list(OBJECTS), None, 0, {CHEESE}, []
    for step in range(fork, end + 1, CHUNK):
        old = {"order_before": order.copy(), "prior_target": current, "until_before": until}
        for applied_step in range(max(fork + 1, step - CHUNK + 1), step + 1):
            completed.update(writes.get(applied_step, ()))
        order = [k for k in order if k not in completed]
        if order:
            if current not in order:
                current, until = order[0], step + ATTEMPT
            elif step >= until:
                order.append(order.pop(0))
                current, until = order[0], step + ATTEMPT
            target = current
        else:
            target = None
        out.append({"step": step, "target": target, "order": order.copy(), "until": until, **old,
                    "official_success": all(rows[step]["audit"]["goals"].values())})
    return out


def sauce_stability(rows, step, end):
    """Predicate persistence; explicitly does not prove the gripper released sauce."""
    future_exit = next((event["step"] for k, row in sorted(rows.items()) if k > step
                        for event in row["transitions"] if event["step"] > step
                        and event["object"] == SAUCE and not event["in_basket"]), None)
    if future_exit is not None and future_exit <= step + STABILITY:
        return False
    return True if end >= step + STABILITY else None


def boundary_view(row):
    return {k: row[k] for k in ("step", "target", "order", "official_success")}


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def audit_grasp(path, initial, rows):
    """Tie the additional read-only grasp query to every unchanged native capture."""
    original_frames = [{**initial, "audit": {"goals": dict.fromkeys(GOALS, False), "held_soup": False}}, *rows]
    captured = read_rows(path)
    require(len(captured) == len(original_frames), f"Grasp capture count differs from the native trace: {path}")
    held = {}
    for side, native in zip(captured, original_frames, strict=True):
        at, grasp = side["step"], side["held_by_target"]
        require(type(at) is int and at == native["step"] and at not in held
                and set(side) == {"step", "state_sha256", "goals", "held_by_target"}
                and side["state_sha256"] == native["state_sha256"] and side["goals"] == native["audit"]["goals"]
                and set(grasp) == set(GOALS) and all(type(v) is bool for v in grasp.values())
                and grasp[SOUP] is native["audit"]["held_soup"],
                f"Grasp capture fails native state/predicate/query parity: {path}:{at}")
        held[at] = grasp
    return held


def audit_events(events, arm, rows, held, folder, start, final, prefix=False):
    applied, writes, wrong, still_held = [], defaultdict(list), [], []
    seen, proven_frames = set(), set()
    last = 0 if prefix else start
    for event in events:
        step, obj, kind, wrote = (event[key] for key in ("step", "object", "event", "applied"))
        require(type(step) is int and last <= step and start < step <= final and step % CHUNK == 0 and step in rows,
                f"Event is outside the audited cadence/section: {folder}:{event}")
        last = step
        writer = "legacy" if arm == "baseline" or prefix else "placement_next"
        assigned = CHEESE if prefix else SOUP
        require(event["writer"] == writer and event["assigned"] == assigned and rows[step]["target"] == assigned
                and obj in ((assigned,) if writer == "legacy" else OBJECTS)
                and kind in (("in_basket",) if writer == "legacy" else
                             ("in_basket", "pending_release", "post_release_sample", "abstained"))
                and wrote is (kind == "in_basket"), f"Unexpected writer/object/applied event: {folder}:{event}")
        sample = rows[step]
        official = sample["audit"]["goals"][obj]
        physical = held[step][obj]
        prior_held = max((s for s in held if s < step and held[s][obj]), default=None)
        require(event["matches_audit"] is official and event["held_soup"] is sample["audit"]["held_soup"]
                and event["released_in_audit"] is (not physical) and event["held_by_target"] is physical
                and event["previous_sampled_held_step"] == prior_held,
                f"Runner's declared truth fields disagree: {folder}:{step}")
        if not wrote:
            continue
        require(obj not in seen, f"Repeated progress commit: {folder}:{step}:{obj}")
        seen.add(obj)
        require(event["evidence_file"] == f"commit-{step}.npz", f"Unexpected observation filename at {folder}:{step}")
        path = checked_path(folder, event["evidence_file"], event["evidence_sha256"])
        if path not in proven_frames:
            observation(path, stamp(sample))
            proven_frames.add(path)
        annotated = {**event, "official_predicate_correct": official,
                     "official_object_still_grasped": physical, "official_entry_and_unheld": official and not physical,
                     "noninstruction_target": obj != assigned}
        applied.append(annotated)
        writes[step].append(obj)
        if not official:
            wrong.append(annotated)
        if physical:
            still_held.append(annotated)
    return {"applied": applied, "wrong_predicate": wrong, "still_grasped": still_held, "writes": writes}


def compare_tracks(arms, rows, held, first, end):
    additions = []
    raw = {arm: arms[arm]["applied"] for arm in ARMS}
    first_removal = {arm: {obj: next((b["step"] for b in arms[arm]["boundaries"] if obj not in b["order"]), None)
                          for obj in OBJECTS} for arm in ARMS}
    for e in raw["candidate"]:
        at = e["step"]
        if e["object"] != SAUCE or not e["official_predicate_correct"] or not e["noninstruction_target"]:
            continue
        require(first_removal["candidate"][SAUCE] == at, "Candidate did not remove newly completed sauce at its next decision")
        old_hit = next((old["step"] for old in raw["baseline"] if old["object"] == SAUCE and old["official_predicate_correct"]), None)
        next_index = next(i for i, b in enumerate(arms["candidate"]["boundaries"]) if b["step"] == at)
        current_truth = rows[at]["audit"]["goals"]
        additions.append({"commit": e, "later_or_missing_baseline_hit": old_hit is None or at < old_hit,
                          "official_entry_step": first[SAUCE], "predicate_delay_steps": at - first[SAUCE],
                          "predicate_stays_true_for_20_steps": sauce_stability(rows, at, end),
                          "before_diagnostic_cutoff": at < end,
                          "nonterminal_soup_pending": (at < end and not rows[at]["done"] and not current_truth[SOUP]
                                                       and first.get(SOUP, BUDGET + 1) > at),
                          "commit_boundary": {arm: boundary_view(arms[arm]["boundaries"][next_index]) for arm in ARMS}})
    boundaries = []
    for before, after in zip(arms["baseline"]["boundaries"], arms["candidate"]["boundaries"], strict=True):
        step = before["step"]
        require(after["step"] == step, "Schedule grids differ")
        old, new = before["target"], after["target"]
        eligible = (SAUCE in before["order"]) != (SAUCE in after["order"])
        if old == new and not eligible:
            continue
        official = rows[step]["audit"]["goals"]
        until_next = any(a["commit"]["step"] <= step and a["commit"]["official_entry_and_unheld"]
                         and a["nonterminal_soup_pending"]
                         and a["later_or_missing_baseline_hit"] for a in additions)
        can_skip = (until_next and old == SAUCE and new is not None and new != old and not official[new]
                    and official[SAUCE] and not held[step][SAUCE] and not rows[step]["done"] and not all(official.values())
                    and SAUCE in before["order"] and SAUCE not in after["order"])
        boundaries.append({"step": step, "baseline": boundary_view(before), "candidate": boundary_view(after),
                           "sauce_eligibility_differs": eligible, "selected_target_differs": old != new,
                           "hypothetical_avoided_completed_sauce_request": can_skip,
                           "official": official, "official_sauce_still_grasped": held[step][SAUCE],
                           "within_source": step < end, "remaining_episode_budget": BUDGET - step})
    return {"first_removal": first_removal, "extra_predicate_correct_noninstruction_sauce_commits": [a for a in additions
                                                                       if a["later_or_missing_baseline_hit"]],
            "first_eligibility_difference": next((b for b in boundaries if b["sauce_eligibility_differs"]), None),
            "first_selected_target_difference": next((b for b in boundaries if b["selected_target_differs"]), None),
            "within_source_avoided_sauce_requests": [b for b in boundaries if b["hypothetical_avoided_completed_sauce_request"]
                                                       and b["within_source"]],
            "cutoff_avoided_sauce_requests": [b for b in boundaries if b["hypothetical_avoided_completed_sauce_request"]
                                                 and not b["within_source"]], "boundaries": boundaries}


def score_case(root, serial, report, expected, source):
    layout = source["expected"]["layout_seed"]
    r, p = source["original"], source["prefix"]
    folder = (root / str(layout)).resolve()
    require(folder.is_relative_to(root) and report["actions"] == "actions.jsonl"
            and report["grasp_audit"] == "grasp-audit.jsonl", f"Unexpected replay artifact paths: {layout}")
    fork, terminal = (stamp(report[key]) for key in ("fork", "terminal"))
    start, end = fork["step"], terminal["step"]
    require(report["fork"] == r["fork"] and report["terminal"] == r["terminal_checkpoint"] and end - start == WINDOW,
            f"Replay fork or terminal differs from sealed source: {layout}")
    observation(checked_path(folder, "fork.npz", report["fork_npz_sha256"]), fork)
    path = checked_path(folder, "actions.jsonl", report["actions_sha256"])
    rows = read_rows(path)
    require(rows == source["rows"] and len(rows) == report["replayed_samples"] == len(p["chunks"]) + WINDOW // STRIDE
            and report["prefix_action_steps"] == start and report["action_steps"] == WINDOW
            and report["total_action_steps"] == end and report["suffix_anchors"] == WINDOW // STRIDE,
            f"Realized controls, native state/predicate/observation or prefix parity failed: {path}")
    by_step = {row["step"]: row for row in rows}
    require(report["fork_audit"] == by_step[start]["audit"] and report["terminal_audit"] == by_step[end]["audit"]
            and report["source_done"] is by_step[end]["done"] is False
            and report["source_stop_reason"] == "diagnostic_cutoff" and report["original_episode_budget"] == BUDGET
            and report["unexecuted_episode_steps"] == BUDGET - end,
            f"Fork, terminal or remaining budget provenance differs: {layout}")
    grasp = checked_path(folder, report["grasp_audit"], report["grasp_audit_sha256"])
    held = audit_grasp(grasp, stamp(r["initial"]), rows)
    require(set(report["arms"]) == set(ARMS), f"Invalid assignment of paired observer arms: {layout}")
    prefix = audit_events(report["prefix_events"], "baseline", by_step, held, folder, 0, start, prefix=True)
    last_cheese = False
    for chunk in p["chunks"]:
        at = chunk["end"]["step"]
        last_cheese = last_cheese or CHEESE in prefix["writes"].get(at, [])
        require(last_cheese is chunk["cheese_observed"] and by_step[at]["audit"]["goals"][CHEESE] is chunk["cheese_truth"],
                f"Shared cheese-prefix observer ledger does not match its seal: {layout}:{at}")
    require(last_cheese, f"The original cheese prefix did not complete: {layout}")
    arms = {}
    for arm in ARMS:
        raw = report["arms"][arm]
        a = audit_events(raw["events"], arm, by_step, held, folder, start, end)
        completed = {CHEESE} | {e["object"] for e in a["applied"]}
        a["terminal_progress"] = {obj: "in_basket" if obj in completed else "pending" for obj in GOALS}
        a["boundaries"] = replay_schedule(a["writes"], by_step, start, end)
        require(a["terminal_progress"] == raw["terminal_progress"] and a["boundaries"] == raw["schedule_boundaries"],
                f"Schedule or completion status disagrees with independent ledger: {layout}:{arm}")
        require(report["wrong_writes"][arm] == len(a["wrong_predicate"])
                and report["held_writes"][arm] == len(a["still_grasped"])
                and report["unsafe_writes"][arm] == sum(not e["official_entry_and_unheld"] for e in a["applied"]),
                f"Runner and independent physical safety counts differ: {layout}:{arm}")
        arms[arm] = a
    comparisons = compare_tracks(arms, by_step, held, source["first"], end)
    for en, own in (("first_target_difference_within_source", next((b for b in comparisons["boundaries"]
                                                                     if b["selected_target_differs"] and b["within_source"]), None)),
                    ("checkpoint_target_difference", next((b for b in comparisons["boundaries"]
                                                            if b["selected_target_differs"] and not b["within_source"]), None))):
        saved = report[en]
        require((saved is None and own is None) or (saved and own and saved["step"] == own["step"]
                and all(boundary_view(saved[arm]) == own[arm] for arm in ARMS)), f"Misclassified target difference: {layout}:{en}")
    return {"layout_seed": layout, "policy_seed": source["expected"]["policy_seed"], "card": expected["card"],
            "fresh_at_source_diagnostic": source["expected"]["fresh"], "report_sha256": serial["report_sha256"],
            "source_result_sha256": source["audit_row"]["result_sha256"], "prefix_sha256": source["prefix_sha256"],
            "source_actions_sha256": r["actions_sha256"], "replay_actions_sha256": report["actions_sha256"],
            "grasp_audit_sha256": report["grasp_audit_sha256"], "checked_native_capture_rows": len(rows) + 1,
            "checked_replayed_action_rows": len(rows), "fork": fork, "source_terminal": terminal,
            "source_stop_reason": report["source_stop_reason"], "remaining_original_episode_steps": BUDGET - end,
            "native_first_entry_steps": source["first"], "prefix_commits": prefix["applied"],
            "arms": {arm: {"predicate_false": a["wrong_predicate"], "still_grasped": a["still_grasped"],
                            "applied_writes": a["applied"], "terminal_observed_progress": a["terminal_progress"]}
                     for arm, a in arms.items()}, **comparisons}


def score_card(root, source, prior, card, runner_hash):
    lock, original = source_inputs(source, prior, card)
    plan_path = root / f"actual-progress-plan-{card}.json"
    card_path = root / f"actual-progress-{card}.json"
    plan, manifest = read(plan_path), read(card_path)
    params = {"card": card, "scope": "development_only", "source_audit_sha256": SOURCE_AUDIT,
              "source_manifest": lock, "bddl_sha256": BDDL, "source_sha256": runner_hash,
              "observer_sha256": OBSERVER_SHA, "compare_sha256": COMPARE_SHA,
              "physics_worker_sha256": PHYSICS_WORKER, "physics_simulator_sha256": PHYSICS_SIMULATOR,
              "prefix": True, "motion": False, "policy_chunk": CHUNK, "source_stride": STRIDE,
              "observer_stride": CHUNK, "schedule_attempt": ATTEMPT, "steps": WINDOW, "source_target": SOUP,
              "source_prompt": PROMPTS["standard"], "hypothetical_schedules": True, "sauce_release_audit_available": True}
    require(matches(plan, {"schema": "v2-stage4-actual-progress-plan/1", **params})
            and matches(manifest, {"schema": "v2-stage4-actual-progress-manifest/1", "card": card, "complete": True,
                                   "plan_sha256": sha(plan_path), "source_audit_sha256": SOURCE_AUDIT})
            and len(plan["cases"]) == len(manifest["records"]) == len(CASES[card]), f"Incomplete or modified plan: {card}")
    rows = []
    for layout, declared, serial in zip(CASES[card], plan["cases"], manifest["records"], strict=True):
        orig = source_case(source, prior, original, card, layout)
        case = {"layout_seed": layout, "policy_seed": orig["expected"]["policy_seed"],
                "source_result_sha256": orig["audit_row"]["result_sha256"],
                "source_actions_sha256": orig["audit_row"]["actions_sha256"], "prefix_sha256": orig["prefix_sha256"]}
        require(matches(declared, case) and matches(serial, {"layout_seed": layout, "report": f"{layout}/report.json"}),
                f"Unexpected layout/results sequence: {layout}")
        report = read(checked_path(root, serial["report"], serial["report_sha256"]))
        require(matches(report, {**params, "schema": "v2-stage4-actual-progress/1", "layout_seed": layout,
                                 "policy_seed": case["policy_seed"], "record_dir": str(layout),
                                 "prefix_sha256": case["prefix_sha256"], "source_actions_sha256": case["source_actions_sha256"],
                                 "source": orig["audit_row"], "plan_sha256": manifest["plan_sha256"]}),
                f"Report metadata differs from the frozen plan and source: {layout}")
        rows.append(score_case(root, serial, report, params, orig))
    return rows, {"card": card, "plan_sha256": sha(plan_path), "manifest_sha256": sha(card_path)}


def score(root, source, out):
    require(root != source and not root.is_relative_to(source) and not source.is_relative_to(root)
            and out.is_relative_to(root) and not out.exists(), "Use a new audit destination inside the separate replay run")
    prior = check_prior(source)
    runner = sha(HERE / "actual_progress_replay.py")
    cases, manifests = [], []
    for card in CASES:
        rows, manifest = score_card(root, source, prior, card, runner)
        cases.extend(rows)
        manifests.append(manifest)
    counts = {arm: {"predicate_false": sum(len(c["arms"][arm]["predicate_false"]) for c in cases),
                    "still_grasped": sum(len(c["arms"][arm]["still_grasped"]) for c in cases),
                    "commits": sum(len(c["arms"][arm]["applied_writes"]) for c in cases)} for arm in ARMS}
    prefix_unsafe = sum(not e["official_entry_and_unheld"] for c in cases for e in c["prefix_commits"])
    safe = not (counts["candidate"]["predicate_false"] or counts["candidate"]["still_grasped"] or prefix_unsafe)
    useful_live = [c["layout_seed"] for c in cases if c["within_source_avoided_sauce_requests"]]
    useful_cutoff = [c["layout_seed"] for c in cases if c["cutoff_avoided_sauce_requests"]]
    useful = set(useful_live) | set(useful_cutoff)
    fresh = {c["layout_seed"] for c in cases if c["fresh_at_source_diagnostic"]}
    extra = [c["layout_seed"] for c in cases if c["extra_predicate_correct_noninstruction_sauce_commits"]]
    released = [c["layout_seed"] for c in cases if any(e["commit"]["official_entry_and_unheld"] and e["nonterminal_soup_pending"]
                for e in c["extra_predicate_correct_noninstruction_sauce_commits"])]
    gate = safe and len(useful) >= 2 and bool(useful & fresh)
    output = {"schema": "v2-stage4-actual-progress-audit/1", "scope": "seen_development_only_same_action",
              "audit_sha256": sha(Path(__file__)), "runner_sha256": runner, "source_audit_sha256": SOURCE_AUDIT,
              "observer_sha256": OBSERVER_SHA, "observer_freeze_sha256": OBSERVER_FREEZE,
              "observer_scoring_source_sha256": COMPARE_SHA, "physics_worker_sha256": PHYSICS_WORKER,
              "physics_simulator_sha256": PHYSICS_SIMULATOR, "manifests": manifests,
              "layouts": len(cases), "checked_replayed_action_rows": sum(c["checked_replayed_action_rows"] for c in cases),
              "checked_all_object_grasp_captures": sum(c["checked_native_capture_rows"] for c in cases),
              "wrong_or_held_shared_prefix_commits": prefix_unsafe, "suffix_counts": counts,
              "layouts_with_extra_predicate_correct_noninstruction_sauce": extra,
              "layouts_with_extra_true_nonterminal_unheld_sauce": released,
              "layouts_with_hypothetical_useful_selection_within_source": useful_live,
              "layouts_with_hypothetical_useful_selection_at_cutoff": useful_cutoff,
              "gate": {"candidate_full_replay_zero_false_and_still_held": safe,
                       "qualifying_distinct_layouts": sorted(useful), "qualifying_new_development_layouts": sorted(useful & fresh),
                       "prepare_tiny_online_development_pair": gate, "release_original_pilot": False},
              "interpretation": "Read-only grasp truth was joined to writes after replay. Selected targets are hypothetical; "
                                "the replay never uses them to choose actions. The gate does not establish task success or broad safety.",
              "cases": cases}
    write(out, output)
    print(f"ACTUAL_PROGRESS_AUDIT layouts={len(cases)} extra_predicate={len(extra)} extra_unheld={len(released)} "
          f"useful_within_source={useful_live} useful_cutoff={useful_cutoff} candidate_safe={safe} prepare_dev={gate}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    score(a.run.resolve(), a.source.resolve(), a.out.resolve())


if __name__ == "__main__":
    main()
