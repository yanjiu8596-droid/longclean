"""Audit the twelve prespecified online development arms against their native evidence."""
import argparse
from collections import defaultdict
from itertools import combinations
from math import isfinite
from pathlib import Path

from stage4_fixture import BDDL_SHA, GPU, require
from common import ATTEMPT, BUDGET, CHEESE, CHUNK, GOALS, LABELS, PREFIX_TASK, SAUCE, SOUP, read, sha, write
from legacy_parity import trace
from release_dense_replay import rows as jsonl
from target_follow_audit import checked_path, observation, stamp
from dense_online_extra import ARMS, LAYOUTS, ORDINARY_SLICE, SAFETY_MARKER, SOURCE_SHA


HERE = Path(__file__).resolve().parent
RUNNER_SHA = "68de384ab9dceaaa4e9a8f2d0f17e60fedc64848954cd9b26b3bfee18bc2a9f2"
CONTROL_PLAN_SHA = "e8014407a15daa4c2a2cdccb3482bdfad7bc3a4f5ce42f10fbd05dc2f3559e17"
REGRESSION_SOURCE_SHA = "45fe8af63157ce4aa1db094891c905977314018fce319bfae980ae73248574aa"
CONTROLS_SOURCE_SHA = "57e21eed8b43b5b8b60bbd8b59bbeb7270ffb050e307e08b1cf7e875df8eb80a"
NORMAL = {"goal", "budget", "environment_done"}
STOP = {"adaptive_safety_stop", "adaptive_safety_stop_peer", "adaptive_safety_stop_skipped"}


def planned(card):
    return [{"layout_seed": seed, "policy_seed": 48 + seed % 2, "card": card,
             "order": [ARMS[(i + j) % len(ARMS)] for j in range(len(ARMS))]}
            for i, seed in enumerate(LAYOUTS) if seed % 2 == card - 2]


def evidence(regression, controls):
    r, c = read(regression), read(controls)
    adaptive, placement = SOURCE_SHA["dense_release_observer.py"], SOURCE_SHA["../../placement-next/observer.py"]
    hits = [{k: e[k] for k in ("step", "pixels", "predicate_true", "held_by_target")} for case in r["cases"]
            for e in case["candidate_events"] if e["event"] == "in_basket"]
    require(r["schema"] == "v2-dense-release-regression/1"
            and r["wrapper_sha256"] == adaptive and r["observer_sha256"] == placement
            and r["source_sha256"] == REGRESSION_SOURCE_SHA == sha(HERE / "dense_release_regression.py")
            and r["expected_archived_signatures_match"] is True and r["incorrect_or_held_writes_in_archived_prefixes"] == 0
            and sorted(case["layout_seed"] for case in r["cases"]) == [931000, 931005, 931200, 931201]
            and sorted((e["step"], e["pixels"]) for e in hits) == [(224, [31, 123]), (236, [0, 86]), (244, [0, 48])]
            and all(e["predicate_true"] is True and e["held_by_target"] is False for e in hits),
            "Required four-layout release regression or artifact code differs")
    require(c["schema"] == "v2-sauce-controls-audit/1"
            and c["candidate_sha256"] == adaptive and c["observer_sha256"] == placement
            and c["audit_sha256"] == CONTROLS_SOURCE_SHA == sha(HERE / "sauce_controls_audit.py")
            and c["plan_sha256"] == CONTROL_PLAN_SHA == sha(HERE / "stage4-sauce-controls-plan.json")
            and sorted(c["layouts_checked"]) == list(range(929336, 929352))
            and c["wrong_or_held_candidate_commits"] == 0 and c["development_online_ready"] is True,
            "Required complete sauce-controls audit or artifact code differs")
    return {"regression_sha256": sha(regression), "regression_source_sha256": r["source_sha256"],
            "controls_audit_sha256": sha(controls), "controls_plan_sha256": CONTROL_PLAN_SHA}


def actions_finite(controls):
    return all(isinstance(a, list) and len(a) == 7
               and all(type(v) in (int, float) and isfinite(v) for v in a) for a in controls)


def native_parity(actions, initial):
    goals = dict.fromkeys(GOALS, False)
    terminal = False
    for r in actions:
        step, s = r["step"], r["start_step"]
        require(not terminal and s < step <= BUDGET and 0 < len(r["actions"]) <= CHUNK
                and stamp(r)["bddl_sha256"] == initial["bddl_sha256"] == BDDL_SHA
                and r["raw_sha256"] == initial["raw_sha256"]
                and actions_finite(r["actions"]) and r["executed"] == len(r["actions"])
                and type(r["done"]) is bool and type(r["audit"]["held_soup"]) is bool
                and type(r["completed_regrasp_steps"]) is int and 0 <= r["completed_regrasp_steps"] <= step - s,
                f"Invalid or post-terminal native actions at {step}")
        last = s
        for t in r["transitions"]:
            when, obj, placed = t["step"], t["object"], t["in_basket"]
            require(s < when <= step and last <= when and obj in GOALS and type(placed) is bool
                    and goals[obj] is not placed, "Invalid official native transition")
            goals[obj] = placed
            last = when
        require(r["audit"]["goals"] == goals and all(type(v) is bool for v in r["audit"]["goals"].values()),
                "Native official endpoint does not match predicate transitions")
        terminal = r["done"] or all(goals.values()) or step >= BUDGET


def prefix_fixture(root, saved, case):
    layout = case["layout_seed"]
    name = f"fixtures/{layout}/fixture.json"
    require(saved["fixture"] == name, "Unexpected fixture path")
    path = checked_path(root, name, saved["fixture_sha256"])
    p = path.parent
    fixture = read(path)
    require(fixture == {k: v for k, v in saved.items() if k not in {"fixture", "fixture_sha256"}}
            and fixture["layout_seed"] == layout and fixture["prefix_seed"] == 60, "Unpaired declared fixture")
    prefix = read(checked_path(p, "prefix.json", fixture["prefix_sha256"]))
    raw = list(trace(checked_path(p, "prefix/actions.jsonl", fixture["prefix_actions_sha256"])))
    init, final = stamp(fixture["initial"]), stamp(fixture["fork"])
    require(init == stamp(prefix["initial"]) and init["step"] == 0 and final == stamp(prefix["end"])
            and len(raw) == len(prefix["chunks"]), "Mismatched frozen prefix endpoints")
    observation(checked_path(p, "checkpoint.npz", fixture["prefix_checkpoint_npz_sha256"]), final)
    native_parity(raw, init)
    for chunk, r in zip(prefix["chunks"], raw, strict=True):
        require(r["target"] == CHEESE and r["instruction"] == PREFIX_TASK and r["actions"] == chunk["actions"]
                and stamp(r) == stamp(chunk["end"]) and r["audit"]["goals"][CHEESE] is chunk["cheese_truth"]
                and type(chunk["cheese_observed"]) is bool, "Frozen cheese prefix actions or truth differ")
    tail = raw[-1] if raw else None
    goal = tail["audit"]["goals"] if tail else None
    require(final == (stamp(tail) if tail else init) and goal == fixture["terminal_official_goals"],
            "Prefix terminal mismatch")
    qualified = bool("failure" not in prefix and prefix["observed_cheese"] is True and tail
                     and final["step"] % CHUNK == 0 and final["step"] < BUDGET
                     and goal == {k: k == CHEESE for k in GOALS} and not tail["done"]
                     and prefix["chunks"][-1]["cheese_truth"] is True)
    require(fixture["qualified"] is qualified
            and fixture["reason"] == (None if qualified else prefix.get("failure", "common_cheese_prefix_unavailable")),
            "Incorrect prefix eligibility")
    return fixture, prefix, raw


def physical_trace(folder, result, native, initial):
    captures = jsonl(checked_path(folder, "grasp-audit.jsonl", result["grasp_audit_sha256"]))
    anchor = {**initial, "audit": {"goals": dict.fromkeys(GOALS, False)}}
    at = {}
    for row, s in zip([anchor, *native], captures, strict=True):
        step, hold = s["step"], s["held_by_target"]
        require(step == row["step"] and step not in at and s["state_sha256"] == row["state_sha256"]
                and s["goals"] == row["audit"]["goals"] and set(hold) == set(GOALS)
                and all(type(v) is bool for v in hold.values())
                and (step == 0 or hold[SOUP] is row["audit"]["held_soup"]),
                "Evaluator grasp sidecar disagrees with native endpoint")
        at[step] = s
    return at


def events(folder, result, native, physical, fork):
    arm = result["arm"]
    at = {r["step"]: r for r in native}
    observed, writes = defaultdict(list), {}
    prior, verified_files, applied = {}, set(), {}
    ordered = result["prefix_events"] + result["state_events"]
    for is_prefix, stream in ((True, result["prefix_events"]), (False, result["state_events"])):
        require([e["step"] for e in stream] == sorted(e["step"] for e in stream), "Unordered events")
        for e in stream:
            require(e["step"] in at and ((e["step"] <= fork) == is_prefix), "Event outside its evidence section")
            observed[e["step"]].append(e)
    for step, s in physical.items():
        for e in observed.get(step, []):
            obj, kind = e["object"], e["event"]
            assigned = at[step]["target"]
            is_prefix = step <= fork
            actual = not is_prefix and arm != "baseline"
            writer = "prefix_legacy" if is_prefix else arm
            commit = kind == "in_basket"
            goal, held = s["goals"][obj], s["held_by_target"][obj]
            require(e["writer"] == writer and e["assigned"] == assigned and e["applied"] is commit
                    and (obj in {SOUP, SAUCE} if actual else obj == assigned)
                    and (kind in {"in_basket", "pending_release", "post_release_sample", "abstained"} if actual else commit)
                    and (step % CHUNK == 0 or (arm == "adaptive1" and not is_prefix and commit and obj == SAUCE))
                    and e["matches_audit"] is goal and e["held_by_target"] is held
                    and e["physically_confirmed"] is (goal and not held)
                    and e["noninstruction_target"] is (obj != assigned)
                    and e["previous_sampled_held_step"] == prior.get(obj), "Writer, event cadence or posthoc truth differs")
            if commit:
                require(obj not in applied and e["evidence_file"] == f"commit-{step}.npz", "Invalid repeated commit/evidence")
                ev = checked_path(folder, e["evidence_file"], e["evidence_sha256"])
                if ev not in verified_files:
                    observation(ev, stamp(at[step]))
                    verified_files.add(ev)
                applied[obj] = step
                writes[step] = obj
        prior.update({obj: step for obj, val in s["held_by_target"].items() if val})
    require(result["terminal_progress"] == {obj: "in_basket" if obj in applied else "pending" for obj in GOALS},
            "Terminal progress does not follow the audited writers")
    suffix = result["state_events"]
    wrong = [e for e in suffix if e["applied"] and not e["matches_audit"]]
    held = [e for e in suffix if e["applied"] and e["held_by_target"]]
    require(wrong == result["incorrect_predicate_commits"] and held == result["still_held_commits"],
            "Incorrect runner error totals")
    return observed, writes, [e for e in ordered if e["applied"] and not e["physically_confirmed"]]


def decisions_and_controls(result, suffix, writes, observed, physical, fork, end):
    arm, reason = result["arm"], result["terminal_reason"]
    policy = result["policy_chunks"]
    observed_targets = {CHEESE}
    order, prior, until = [SOUP, SAUCE], None, 0
    repeats, checked_calls, single = [], 0, 0
    native = iter(suffix)
    cursor = fork
    pending, pending_at = None, None
    writes_left = iter(sorted((s, o) for s, o in writes.items() if s > fork))
    next_write = next(writes_left, None)
    decisions = result["schedule_decisions"]
    stopped_on_empty = False
    for i, d in enumerate(decisions):
        s = d["step"]
        require(s == cursor and s % CHUNK == 0 and s in physical and s < BUDGET, "Illegal scheduler call cadence")
        old = {"order_before": order.copy(), "prior_target": prior, "until_before": until}
        while next_write is not None and next_write[0] <= s:
            observed_targets.add(next_write[1])
            next_write = next(writes_left, None)
        order = [obj for obj in order if obj not in observed_targets]
        target = None
        if order:
            if prior not in order:
                prior, until = order[0], s + ATTEMPT
            elif s >= until:
                order.append(order.pop(0))
                prior, until = order[0], s + ATTEMPT
            target = prior
        truth = physical[s]
        require(d == {"step": s, "target": target, "order": order.copy(), **old, "until": until,
                      "official_success": all(truth["goals"].values()), "goals_before": truth["goals"],
                      "held_before": truth["held_by_target"]} and not d["official_success"],
                "Schedule changed relative to the committed observations")
        if target is None:
            require(i == len(decisions) - 1 and reason == "observed_complete_incomplete", "Undeclared empty schedule")
            stopped_on_empty = True
            break
        if truth["goals"][target]:
            repeats.append({"step": s, "target": target, "held_at_selection": truth["held_by_target"][target]})
        chunk = policy[checked_calls]
        checked_calls += 1
        issued, executed = chunk["issued_actions"], []
        instruction = f"put the {LABELS[target]} in the basket"
        require(chunk["start_step"] == s and chunk["target"] == target and chunk["prompt"] == instruction
                and len(issued) == CHUNK and actions_finite(issued), "Wrong target, prompt or policy action block")
        for sample in chunk["samples"]:
            row = next(native)
            dense = arm == "adaptive1" and pending == SAUCE
            width = min(1 if dense else ORDINARY_SLICE, CHUNK - len(executed))
            n = len(row["actions"])
            require(row["start_step"] == cursor and row["step"] == cursor + n and row["target"] == target
                    and row["instruction"] == instruction and 0 < n <= width
                    and row["actions"] == issued[len(executed):len(executed) + n]
                    and (n == width or (row["step"] == end and (row["done"] or all(row["audit"]["goals"].values()))))
                    and sample == {"step": row["step"], "steps_executed": n, "dense_requested": dense},
                    "Realized native action differs from issued block or adaptive call condition")
            for e in observed.get(row["step"], []):
                if arm == "baseline":
                    continue
                k, obj = e["event"], e["object"]
                if k == "pending_release":
                    require(pending is None, "Multiple live pending releases")
                    pending, pending_at = obj, row["step"]
                else:
                    require(pending == obj, "Release event without corresponding pending transport")
                    if k == "in_basket":
                        require(e["release_step"] == pending_at and 0 < row["step"] - pending_at <= 30,
                                "Placement did not follow a valid release age")
                    if k in {"in_basket", "abstained"}:
                        pending, pending_at = None, None
            cursor = row["step"]
            executed.extend(row["actions"])
            single += dense
        require(chunk["end_step"] == cursor and chunk["executed_actions"] == executed
                and (len(executed) == CHUNK or (cursor == end and reason in (NORMAL | STOP))),
                "Partial action block lacks a terminal or safety stop")
    require(next(native, None) is None and cursor == end and checked_calls == len(policy) == result["policy_calls"]
            and stopped_on_empty == (reason == "observed_complete_incomplete")
            and repeats == result["completed_reassigned_chunks"] and single == result["single_control_observations_requested"]
            and sum(r["completed_regrasp_steps"] for r in suffix) == result["completed_regrasp_steps"],
            "Incomplete decisions, repeated targets, native regrasp or sample counts")
    return repeats


def arm_report(root, item, case, fixture_and_trace):
    fixture, prefix, initial_rows = fixture_and_trace
    layout, arm = case["layout_seed"], item["arm"]
    name = f"episodes/{layout}/{arm}/result.json"
    require(item["result"] == name and fixture["qualified"], "Unexpected online result location or prefix status")
    path = checked_path(root, name, item["result_sha256"])
    r = read(path)
    require(r["schema"] == "v2-dense-online-arm/1" and r["arm"] == arm
            and all(r[k] == case[k] for k in ("card", "policy_seed", "layout_seed"))
            and r["motion"] is False and r["budget"] == BUDGET and r["prefix_sha256"] == fixture["prefix_sha256"]
            and r["initial"] == fixture["initial"] and r["fork"] == fixture["fork"], "Unpaired online arm")
    raw = list(trace(checked_path(path.parent, "actions.jsonl", r["actions_sha256"])))
    require(raw[:len(initial_rows)] == initial_rows, "Online arm did not match every shared prefix action and endpoint")
    native_parity(raw, stamp(r["initial"]))
    fork = r["fork"]["step"]
    observation(checked_path(path.parent, "fork.npz", r["fork_npz_sha256"]), stamp(r["fork"]))
    side = physical_trace(path.parent, r, raw, stamp(r["initial"]))
    observed, writes, unsafe = events(path.parent, r, raw, side, fork)
    previous = False
    for chunk in prefix["chunks"]:
        at = chunk["end"]["step"]
        present = CHEESE == writes.get(at)
        require(present == (chunk["cheese_observed"] and not previous), "Prefix commitment disagrees with source")
        previous = chunk["cheese_observed"]
    tail, reason = raw[-1], r["terminal_reason"]
    end = tail["step"]
    official = all(tail["audit"]["goals"].values())
    on_terminal = official or tail["done"] or end >= BUDGET
    normal_reason = "goal" if official else "environment_done" if tail["done"] else "budget"
    unsafe_suffix = [e for e in unsafe if e["step"] > fork]
    qualified = official and reason in NORMAL
    require(r["terminal"] == stamp(tail) and r["terminal_native_audit"] == tail["audit"]
            and r["terminal_held_by_target"] == side[end]["held_by_target"] and r["after_fork_steps"] == end - fork
            and r["official_success"] is item["official_success"] is official
            and r["qualified_success"] is item["qualified_success"] is qualified and reason == item["status"]
            and (reason == normal_reason if reason in NORMAL else reason in STOP | {"observed_complete_incomplete"})
            and (on_terminal or reason not in NORMAL)
            and (reason == "adaptive_safety_stop") == (arm == "adaptive1" and bool(unsafe_suffix))
            and (arm == "adaptive1" or reason not in STOP)
            and (not unsafe_suffix or arm != "adaptive1" or all(e["step"] == end for e in unsafe_suffix)),
            "Incorrect official/qualified terminal or adaptive safety stop")
    repeats = decisions_and_controls(r, raw[len(initial_rows):], writes, observed, side, fork, end)
    gpu = r["gpu_placement"]
    require((gpu is None and r["policy_calls"] == 0)
            or (gpu["uuid"] == GPU[case["card"]] and gpu["egl_device_id"] == str(case["card"])),
            "Incorrect GPU/EGL placement evidence")
    return {"layout_seed": layout, "arm": arm, "status": reason, "valid_native_terminal": reason in NORMAL,
            "interrupted": reason in STOP, "official_success": official, "qualified_success": qualified,
            "steps": end, "repeats": len(repeats), "repeat_records": repeats, "regrasp_steps": r["completed_regrasp_steps"],
            "used_single_control_observations": r["single_control_observations_requested"], "commit_count": len(writes) - 1,
            "incorrect_predicate_commits": len(r["incorrect_predicate_commits"]),
            "still_held_commits": len(r["still_held_commits"]), "unsafe_prefix": [e for e in unsafe if e["step"] <= fork],
            "unsafe_suffix": unsafe_suffix, "evidence_events": [e for e in r["state_events"] if e["applied"]],
            "prefix_sha256": r["prefix_sha256"], "fork_npz_sha256": r["fork_npz_sha256"],
            "actions_sha256": r["actions_sha256"], "grasp_audit_sha256": r["grasp_audit_sha256"],
            "result": name, "result_sha256": item["result_sha256"]}


def comparisons(records):
    by = {(r["layout_seed"], r["arm"]): r for r in records}
    out = []
    for left, right in combinations(ARMS, 2):
        cases = []
        for seed in LAYOUTS:
            a, b = by[seed, left], by[seed, right]
            scorable = all(r["status"] in NORMAL | {"observed_complete_incomplete"} for r in (a, b))
            outcome = None if not scorable else ("tie" if a["qualified_success"] == b["qualified_success"]
                                                 else left if a["qualified_success"] else right)
            cases.append({"layout_seed": seed, "scorable": scorable, "success_comparison": outcome,
                          "both_valid_native_terminal": bool(a.get("valid_native_terminal") and b.get("valid_native_terminal")),
                          "completed_reassignment_delta_right_minus_left": b["repeats"] - a["repeats"] if scorable else None})
        out.append({"left": left, "right": right, "cases": cases, "n_paired": sum(c["scorable"] for c in cases),
                    "wins_left": sum(c["success_comparison"] == left for c in cases),
                    "wins_right": sum(c["success_comparison"] == right for c in cases),
                    "ties": sum(c["success_comparison"] == "tie" for c in cases)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    for key in ("regression", "controls", "study"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    root, out = args.run.resolve(), args.out.resolve()
    require(not out.exists() and sha(HERE / "dense_online_extra.py") == RUNNER_SHA, "Changed runner or existing output")
    gates = evidence(args.regression or root.parent / "stage4-release-dense/regression.json",
                     args.controls or root.parent / "stage4-sauce-controls/audit.json")
    study = args.study or root.parent.parents[1] / "study4"
    frozen = read(study / "frozen.json")
    require(sha(study / "plan.json") == frozen["plan_sha256"]
            and sha(study / read(study / "plan.json")["bddl"]) == BDDL_SHA, "Changed study plan/scene")
    lock = {**SOURCE_SHA, "dense_online_extra.py": RUNNER_SHA, "dense_release_regression.py": REGRESSION_SOURCE_SHA,
            **{f"../../{name}": digest for name, digest in frozen["code"].items()}}
    require(all(sha(HERE / name) == digest for name, digest in lock.items()), "Changed controller or observer code")
    marker = root / SAFETY_MARKER
    global_stop = read(marker) if marker.exists() else None
    records, fixtures, cards = [], {}, []
    for card in GPU:
        index_path = root / f"dense-online-{card}.json"
        index = read(index_path)
        plan_path = checked_path(root, f"dense-online-plan-{card}.json", index["plan_sha256"])
        p = read(plan_path)
        cases = planned(card)
        require(p == {"schema": "v2-dense-online-plan/1", "scope": "new_development_layouts", "card": card,
                      "cases": cases, "motion": False, "budget": BUDGET, "policy_chunk": CHUNK, "normal_slice": ORDINARY_SLICE,
                      "adaptive_pending_slice": 1, "bddl_sha256": BDDL_SHA, "study4_sha256": sha(study / "frozen.json"),
                      "gate_evidence": gates, "code": lock, "release_original_pilot": False},
                f"Changed preassigned card-{card} plan")
        require(index["schema"] == "v2-dense-online-manifest/1" and index["card"] == card
                and type(index["complete"]) is bool and (index["complete"] or global_stop is not None)
                and index["release_original_pilot"] is False
                and (not index["complete"] or (type(index["adaptive_safety_stop_present"]) is bool
                                               and (not index["adaptive_safety_stop_present"] or global_stop is not None))),
                "Missing completed or explicit safety-stopped card manifest")
        expected = [(c, arm) for c in cases for arm in c["order"]]
        completed = index["episodes"]
        prefix_records = index["prefixes"]
        require(len(completed) <= len(expected) and len(prefix_records) <= len(cases)
                and (not index["complete"] or (len(completed) == len(expected) and len(prefix_records) == len(cases))),
                "Missing or excessive online records")
        for c, saved in zip(cases[:len(prefix_records)], prefix_records, strict=True):
            fixtures[c["layout_seed"]] = prefix_fixture(root, saved, c)
        seen = []
        for case_idx, (c, arm) in enumerate(expected):
            seed = c["layout_seed"]
            fixture = fixtures.get(seed)
            if case_idx >= len(completed):
                seen.append({"layout_seed": seed, "arm": arm, "status": "unrecorded_after_explicit_stop", "valid_native_terminal": False,
                             "interrupted": True, "official_success": None, "qualified_success": None, "repeats": 0})
                continue
            item = completed[case_idx]
            require(item["layout_seed"] == seed and item["policy_seed"] == c["policy_seed"] and item["arm"] == arm
                    and fixture is not None, "Reordered or unsealed result assignment")
            status = item["status"]
            if item["result"] is None:
                require(item["result_sha256"] is None and (status == "prefix_unavailable" and not fixture[0]["qualified"]
                        or status == "adaptive_safety_stop_skipped" and fixture[0]["qualified"]
                        and arm == "adaptive1" and global_stop is not None), "Unjustified skipped episode")
                seen.append({"layout_seed": seed, "arm": arm, "status": status, "valid_native_terminal": False,
                             "interrupted": status in STOP, "official_success": None, "qualified_success": None, "repeats": 0})
            else:
                seen.append(arm_report(root, item, c, fixture))
        require(index["adaptive_wrong_commit"] is any(r["status"] == "adaptive_safety_stop" for r in seen),
                "Cached card safety flag differs from audited evidence")
        records.extend(seen)
        cards.append({"card": card, "complete": index["complete"], "plan_sha256": sha(plan_path), "manifest_sha256": sha(index_path)})
    have_four = len(fixtures) == len(LAYOUTS)
    require((have_four or global_stop is not None)
            and all(len({f[0]["initial"][k] for f in fixtures.values()}) == len(fixtures)
                    for k in ("raw_sha256", "state_sha256")), "Missing or duplicate independent initial states")
    for seed in LAYOUTS:
        forks = {r["fork_npz_sha256"] for r in records if r["layout_seed"] == seed and "fork_npz_sha256" in r}
        require(len(forks) <= 1, "Paired rollout fork snapshot bytes differ")
    own = [r for r in records if r["status"] == "adaptive_safety_stop"]
    require(bool(own) == (global_stop is not None) and (global_stop is not None or not any(r["interrupted"] for r in records)),
            "Global adaptive safety marker does not reflect the audited episodes")
    if global_stop is not None:
        require(any(global_stop == {"layout_seed": r["layout_seed"], "card": 2 + r["layout_seed"] % 2,
                                    "events": r["unsafe_suffix"]} for r in own),
                "Global adaptive safety marker is not backed by an audited unsafe write")
    write(out, {"schema": "v2-dense-online-audit/1", "scope": "new development layouts only",
                "audit_sha256": sha(Path(__file__)), "runner_sha256": RUNNER_SHA, "gate_evidence": gates,
                "cards": cards, "all_four_unique_initials_verified": have_four, "prefix_eligible": {
                    seed: fixture[0]["qualified"] for seed, fixture in sorted(fixtures.items())}, "records": records,
                "paired_success": comparisons(records), "adaptive_safety_stop": global_stop,
                "adaptive_safety_stop_sha256": sha(marker) if global_stop is not None else None,
                "interrupted_records": sum(r["interrupted"] for r in records),
                "valid_native_terminal_records": sum(r["valid_native_terminal"] for r in records),
                "independent_pilot_release": False})


if __name__ == "__main__":
    main()
