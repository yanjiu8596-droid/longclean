"""Independently validate the fixed-action target-progress development diagnostic."""

import argparse
from collections import defaultdict
from pathlib import Path

from target_follow_audit import (BDDL, CHUNK, GOALS, PROMPTS, SAUCE, SOUP, STRIDE, WINDOW, audit_trace, checked_path,
                                expected_rows, matches, observation, read, require, sha, stamp, write)
from common import ATTEMPT, BUDGET, CHEESE


HERE = Path(__file__).resolve().parent
SOURCE_AUDIT = "0d00abc23a0264bb783084332b41ea91e538d76e6e992dad8f836d08592c2be3"
TRACKER_SHA = "b61113ded60b3f258b3bcb697f292d0f92bc01e14441cdfb82ff234521886f34"
CASES = {2: (931200, 46), 3: (931201, 47)}
ARMS = ("baseline", "candidate")
WRITER = "stage3d_plane"


def sample_truth(row):
    return row["audit"]["goals"], row["audit"]["held_soup"], row["done"]


def physical_support(row, obj=SOUP):
    goals, held, _ = sample_truth(row)
    return goals[obj] and (obj != SOUP or not held)


def audit_events(events, arm, rows, folder):
    """Audit each observer claim independently of the runner's flags."""
    applied, wrong_writes, wrong_plane, by_step = [], [], [], defaultdict(list)
    shadow = []
    last = 0
    frame_digests = {}
    for e in events:
        step, writer, event, obj, wrote = (e[k] for k in ("step", "writer", "event", "object", "applied"))
        require(type(step) is int and last <= step and step in rows and obj == SOUP and type(wrote) is bool
                and writer in (WRITER, "legacy"), f"Malformed or unordered event at {folder}:{e}")
        current = rows[step]
        goals, held, _ = sample_truth(current)
        confirmed = physical_support(current, obj)
        require(e["matches_audit"] is goals[obj] and e["held_soup"] is held and e["confirmed_in_audit"] is confirmed,
                f"Event truth flags differ from native simulator trace: {folder}:{step}")
        if writer == "legacy":
            require(arm == "baseline" and event == "in_basket" and wrote,
                    f"Legacy soup writer must only affect baseline: {folder}:{step}")
        else:
            require(step % STRIDE == 0, f"Bad stage-3d observer cadence: {folder}:{step}")
            shadow.append({k: v for k, v in e.items() if k not in ("applied", "evidence_file", "evidence_sha256")})
            require(wrote is (arm == "candidate" and event == "in_basket" and not applied),
                    f"Stage-3d write or shadow ownership differs: {folder}:{step}")
        if event == "in_basket":
            require(e["evidence_file"] == f"soup-commit-{step}.npz", f"Unexpected evidence file: {folder}:{step}")
            path = checked_path(folder, e["evidence_file"], e["evidence_sha256"])
            if path not in frame_digests:
                observation(path, stamp(current))
                frame_digests[path] = e["evidence_sha256"]
            require(frame_digests[path] == e["evidence_sha256"], f"Conflicting evidence digest: {path}")
            if writer == WRITER and not confirmed:
                wrong_plane.append(e)
        if wrote:
            require(not applied and event == "in_basket", f"Duplicate or invalid progress mutation: {folder}:{step}")
            applied.append(e)
            by_step[step].append(obj)
            if not confirmed:
                wrong_writes.append(e)
        last = step
    return {"applied": applied, "by_step": by_step, "shadow": shadow,
            "wrong_writes": wrong_writes, "wrong_plane": wrong_plane}


def replay_schedule(writes, rows, end):
    """Independent state machine for frozen quiet Schedule.next, with no policy calls."""
    order, target, until, out, completed = [SOUP, SAUCE, CHEESE], None, 0, [], set()
    for step in range(0, end + 1, CHUNK):
        old = {"order_before": order.copy(), "prior_target": target, "until_before": until}
        for applied_step in range(max(1, step - CHUNK + 1), step + 1):
            completed.update(writes.get(applied_step, ()))
        order = [k for k in order if k not in completed]
        if order:
            if target not in order:
                target, until = order[0], step + ATTEMPT
            elif step >= until:
                order.append(order.pop(0))
                target, until = order[0], step + ATTEMPT
            chosen = target
        else:
            chosen = None
        out.append({"step": step, "target": chosen, "order": order.copy(), "until": until, **old,
                    "official_success": bool(step and all(rows[step]["audit"]["goals"].values()))})
    return out


def boundary_view(row):
    return {k: row[k] for k in ("step", "target", "order", "official_success")}


def compare_boundaries(checked, rows, first_steps, end):
    baseline, candidate = (checked[arm]["schedule_boundaries"] for arm in ARMS)
    valid_writes = {}
    for arm in ARMS:
        valid_writes[arm] = []
        for e in checked[arm]["applied"]:
            step = e["step"]
            goals, held, done = sample_truth(rows[step])
            if (physical_support(rows[step]) and not goals[SAUCE] and not all(goals.values())
                    and not done and first_steps.get(SAUCE, end + 1) > step):
                before = next((b for b in reversed(checked[arm]["schedule_boundaries"]) if b["step"] < step), None)
                after = next((b for b in checked[arm]["schedule_boundaries"] if b["step"] >= step), None)
                if before and after and SOUP in before["order"] and SOUP not in after["order"]:
                    valid_writes[arm].append({"event": e, "official": goals, "held_soup": held,
                                              "boundary_before": boundary_view(before),
                                              "next_boundary": boundary_view(after),
                                              "soup_still_in_at_next_boundary": physical_support(rows[after["step"]])})
    first_removal = {arm: next((boundary_view(b) for b in checked[arm]["schedule_boundaries"]
                               if SOUP not in b["order"]), None) for arm in ARMS}
    first_difference, eligibility_only, final_only, live = None, [], [], []
    differences = []
    for old, new in zip(baseline, candidate, strict=True):
        step = old["step"]
        require(new["step"] == step, "Arms are on different decision grids")
        elig_diff = (SOUP in old["order"]) != (SOUP in new["order"])
        target_diff = old["target"] != new["target"]
        if not elig_diff and not target_diff:
            continue
        goals, held, done = sample_truth(rows[step])
        row = {"step": step, "baseline": boundary_view(old), "candidate": boundary_view(new),
               "soup_eligibility_differs": elig_diff, "target_differs": target_diff,
               "official": goals, "held_soup": held, "environment_done": done,
               "within_source": step < end, "remaining_task_budget": BUDGET - step}
        if target_diff and first_difference is None:
            first_difference = row
        if elig_diff and not target_diff:
            eligibility_only.append(row)
        differences.append(row)
        advance = first_removal["candidate"]
        old_removal = first_removal["baseline"]
        causal = bool(advance and (old_removal is None or advance["step"] < old_removal["step"])
                      and advance["step"] <= step
                      and any(v["event"]["step"] < end and v["soup_still_in_at_next_boundary"]
                              for v in valid_writes["candidate"]))
        if target_diff and causal and physical_support(rows[step]) and not done and not all(goals.values()):
            if step < end:
                live.append(row)
            elif end == WINDOW and step == end and step < BUDGET:
                final_only.append(row)
    return {"first_soup_removal": first_removal, "valid_writes_before_sauce": valid_writes,
            "first_target_difference": first_difference, "eligibility_only_boundaries": eligibility_only,
            "schedule_differences": differences, "live_target_differences": live,
            "cutoff_target_differences": final_only}


def sealed_case(source, history, card):
    layout, seed = CASES[card]
    expected, = [r for r in expected_rows(card) if r["layout_seed"] == layout and not r["prefix"]
                 and r["prompt_id"] == "standard"]
    row, = [r for r in history["records"] if matches(r, expected)]
    recorded, = [r for r in history["manifests"] if r["card"] == card]
    plan = read(checked_path(source, f"target-follow-plan-{card}.json", recorded["plan_sha256"]))
    manifest = read(checked_path(source, f"target-follow-{card}.json", recorded["manifest_sha256"]))
    item, = [r for r in manifest["records"] if matches(r, expected)]
    require(matching_plan(plan, manifest, recorded, row, item, card), f"Changed source case: {layout}")
    path = checked_path(source, row["result"], row["result_sha256"])
    r = read(path)
    require(matches(r, expected) and r["initial"] == r["fork"] == row["initial"] == row["fork"]
            and r["terminal_checkpoint"] == row["terminal"] and row["status"] == r["status"] == "complete"
            and r["budget"] == r["suffix_steps"] == WINDOW and r["motion"] is False
            and r["actions"] == "actions.jsonl" and row["actions_sha256"] == r["actions_sha256"]
            and row["fork_npz_sha256"] == r["fork_npz_sha256"], f"Changed sealed result for layout {layout}")
    action_path = checked_path(path.parent, "actions.jsonl", row["actions_sha256"])
    observation(checked_path(path.parent, "fork.npz", row["fork_npz_sha256"]), stamp(row["fork"]))
    first, tail, _, _, rows = audit_trace(action_path, expected, stamp(row["initial"]), stamp(row["fork"]), None)
    require(first == row["first_entry_steps"] and first[SOUP] < first.get(SAUCE, WINDOW + 1)
            and stamp(tail) == row["terminal"] and tail["step"] == WINDOW and not tail["done"]
            and not all(tail["audit"]["goals"].values()) and tail["audit"]["goals"] == row["terminal_goals"]
            and len(rows) == WINDOW // STRIDE, f"Source replay truth differs from earlier audit: {layout}")
    return row, recorded, expected, rows, first


def matching_plan(plan, manifest, checksum, row, item, card):
    prefix_free = {"card": card, "budget": WINDOW, "policy_chunk": CHUNK, "observer_stride": STRIDE,
                   "motion": False, "bddl_sha256": BDDL}
    return (matches(plan, prefix_free) and plan["cases"] == expected_rows(card)
            and manifest["complete"] is True and manifest["plan_sha256"] == checksum["plan_sha256"]
            and row["status"] == item["status"] == "complete" and row["result"] == item["result"]
            and row["result_sha256"] == item["result_sha256"] and row["category"] == "soup_first"
            and row["soup_pending_sauce_boundary"] is not None)


def score_case(root, source, history, card, runner_sha):
    layout, seed = CASES[card]
    row, recorded, expected, source_rows, first = sealed_case(source, history, card)
    manifest_path = root / f"target-progress-{card}.json"
    plan_path = root / f"target-progress-plan-{card}.json"
    plan, report = read(plan_path), read(manifest_path)
    params = {"card": card, "layout_seed": layout, "policy_seed": seed, "prefix": False, "motion": False,
              "scope": "development_only", "source_audit_sha256": SOURCE_AUDIT, "source_result_sha256": row["result_sha256"],
              "source_actions_sha256": row["actions_sha256"], "source_manifest": recorded, "source_sha256": runner_sha,
              "bddl_sha256": BDDL, "steps": WINDOW, "observer_stride": STRIDE, "policy_chunk": CHUNK,
              "schedule_attempt": ATTEMPT, "source_target": SOUP, "source_prompt": PROMPTS["standard"],
              "hypothetical_schedules": True}
    require(matches(plan, {"schema": "v2-stage4-target-progress-plan/1", **params})
            and matches(report, {"schema": "v2-stage4-target-progress/1", "source": row, "record_dir": str(layout),
                                 "plan_sha256": sha(plan_path), **params}), f"Changed replay plan, source or report: {layout}")
    folder = (root / str(layout)).resolve()
    require(folder.is_relative_to(root) and report["actions"] == "actions.jsonl", f"Unexpected replay paths: {layout}")
    require(report["fork"] == row["fork"] == row["initial"] and report["terminal"] == row["terminal"],
            f"Initial or terminal replay fingerprint differs from sealed source: {layout}")
    action_path = checked_path(folder, "actions.jsonl", report["actions_sha256"])
    observation(checked_path(folder, "fork.npz", report["fork_npz_sha256"]), stamp(report["fork"]))
    _, tail, _, _, replay_rows = audit_trace(action_path, expected, stamp(report["fork"]), stamp(report["fork"]), None)
    require(replay_rows == source_rows and report["replayed_samples"] == len(replay_rows)
            and report["terminal_audit"] == tail["audit"] and report["action_steps"] == tail["step"]
            and report["source_done"] is False and report["source_stop_reason"] == "diagnostic_cutoff"
            and report["original_episode_budget"] == BUDGET
            and report["unexecuted_episode_steps"] == BUDGET - WINDOW,
            f"Replay actions/fingerprints/truth, or budget accounting, differs from sealed source: {layout}")
    require(report["fork_audit"]["goals"] == dict.fromkeys(GOALS, False)
            and report["fork_audit"]["held_soup"] is False, f"Unexpected fork progress: {layout}")
    require(set(report["arms"]) == set(ARMS), f"Unexpected progress writers: {layout}")
    by_step = {r["step"]: r for r in replay_rows}
    assessed = {}
    for arm in ARMS:
        raw = report["arms"][arm]
        checked = audit_events(raw["events"], arm, by_step, folder)
        boundaries = replay_schedule(checked["by_step"], by_step, tail["step"])
        completed = {e["object"] for e in checked["applied"]}
        statuses = {name: "in_basket" if name in completed else "pending" for name in GOALS}
        require(boundaries == raw["schedule_boundaries"] and statuses == raw["terminal_progress"],
                f"Schedule and terminal progress do not derive from the observed writes: {layout}:{arm}")
        assessed[arm] = {**checked, "schedule_boundaries": boundaries, "terminal_progress": statuses}
    old, new = (assessed[arm] for arm in ARMS)
    require(old["shadow"] == new["shadow"], f"Candidate observer differed in the two branches: {layout}")
    independent = compare_boundaries(assessed, by_step, first, tail["step"])
    wrong_writes = sum(len(a["wrong_writes"]) for a in assessed.values())
    wrong_plane = len(old["wrong_plane"])  # shared claims counted once per trace
    require(report["wrong_writes"] == wrong_writes and report["wrong_candidate_observations"] == wrong_plane,
            f"Runner and independent safety counts disagree: {layout}")
    for field, check in (("first_target_difference_within_source", next((d for d in independent["schedule_differences"]
                                                                         if d["target_differs"] and d["within_source"]), None)),
                         ("checkpoint_target_difference", next((d for d in independent["schedule_differences"]
                                                                if d["target_differs"] and not d["within_source"]), None))):
        actual = report[field]
        require((actual is None and check is None) or (actual is not None and check is not None
                and actual["step"] == check["step"]
                and all(boundary_view(actual[a]) == check[a] for a in ARMS)), f"Runner divergence report differs: {layout}:{field}")
    return {"card": card, "layout_seed": layout, "policy_seed": seed, "plan_sha256": sha(plan_path),
            "report_sha256": sha(manifest_path), "sealed_source_result_sha256": row["result_sha256"],
            "sealed_source_actions_sha256": row["actions_sha256"], "replay_actions_sha256": report["actions_sha256"],
            "fork": report["fork"], "source_end": report["terminal"], "source_first_entry_steps": first,
            "source_terminal_goals": tail["audit"]["goals"], "source_stop_reason": report["source_stop_reason"],
            "remaining_original_task_steps": BUDGET - WINDOW, "checked_samples": len(replay_rows),
            "wrong_writes": wrong_writes, "wrong_stage3d_claims": wrong_plane,
            "raw_applied_writes": {a: assessed[a]["applied"] for a in ARMS},
            "wrong_applied_writes": {a: assessed[a]["wrong_writes"] for a in ARMS},
            "wrong_stage3d_events": old["wrong_plane"], "tracker_status": {a: assessed[a]["terminal_progress"] for a in ARMS},
            **independent}


def score(root, source, out):
    require(root != source and not root.is_relative_to(source) and not source.is_relative_to(root)
            and out.is_relative_to(root) and not out.exists(), "Use a new audit destination inside the separate replay run")
    for path in (*[root / f"target-progress-{card}.json" for card in CASES],
                 *[root / f"target-progress-plan-{card}.json" for card in CASES]):
        require(out != path.resolve(), "Audit destination overlaps input evidence")
    history = read(checked_path(source, "audit.json", SOURCE_AUDIT))
    require(history["schema"] == "v2-stage4-target-follow-audit/1" and history["scope"] == "development_only"
            and history["pilot_release"] is False and history["recommendation"] == {"prefix": False, "prompt_id": "standard"}
            and sha(HERE / "target_follow_audit.py") == history["audit_sha256"]
            and sha(HERE / "target_follow.py") == history["runner_sha256"]
            and all(sha(HERE / p) == h for p, h in history["frozen_native_source_sha256"].items())
            and sha(HERE / "progress.py") == TRACKER_SHA, "Pinned source or progress observer changed")
    runner_sha = sha(HERE / "target_progress_replay.py")
    cases = [score_case(root, source, history, card, runner_sha) for card in CASES]
    wrong, shadow = (sum(c[key] for c in cases) for key in ("wrong_writes", "wrong_stage3d_claims"))
    safe = (wrong == shadow == 0)
    valid = sum(len(c["valid_writes_before_sauce"]["candidate"]) for c in cases)
    live = sum(bool(c["live_target_differences"]) for c in cases)
    checkpoint = sum(bool(c["cutoff_target_differences"]) for c in cases)
    output = {"schema": "v2-stage4-target-progress-audit/1", "scope": "development_only_fixed_actions",
              "source_audit_sha256": SOURCE_AUDIT, "source_runner_sha256": runner_sha,
              "audit_sha256": sha(Path(__file__)), "stage3d_progress_sha256": TRACKER_SHA,
              "replayed_layouts": len(cases), "checked_samples": sum(c["checked_samples"] for c in cases),
              "wrong_applied_writes": wrong, "wrong_stage3d_claims": shadow, "safe_on_this_replay": safe,
              "valid_nonterminal_candidate_writes": valid, "layouts_with_live_target_difference": live,
              "layouts_with_checkpoint_target_difference": checkpoint,
              "gate": {"valid_nonterminal_write": safe and bool(valid),
                       "within_source_online_eligible": safe and bool(live),
                       "checkpoint_continuation_eligible": safe and bool(checkpoint),
                       "release_original_pilot": False},
              "interpretation": "The same sealed soup actions were applied regardless of either schedule prediction. "
                                "A different selected target would only affect future online decisions; no action or task-success gain was tested.",
              "cases": cases}
    write(out, output)
    print(f"TARGET_PROGRESS_AUDIT layouts={len(cases)} valid={valid} wrong={wrong} shadow={shadow} "
          f"within_source_eligible={safe and bool(live)} checkpoint_eligible={safe and bool(checkpoint)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    score(a.run.resolve(), a.source.resolve(), a.out.resolve())


if __name__ == "__main__":
    main()
