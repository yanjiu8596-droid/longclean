"""Independently audit frozen full-trajectory sauce controls on CPU."""
import argparse
import math
from pathlib import Path

from actual_progress_replay import HERE, OBSERVER_SHA, PHYSICS_SIMULATOR_SHA, PHYSICS_WORKER_SHA, PLACEMENT, load_exact
from common import CHUNK, GOALS, SAUCE, SOUP, read, sha, write
from legacy_parity import trace
from release_dense_replay import rows
from stage4_fixture import GPU, require


PLAN_SHA = "e8014407a15daa4c2a2cdccb3482bdfad7bc3a4f5ce42f10fbd05dc2f3559e17"
RUNNER_SHA = "6b982f7f31f5c2e120a21024034e23dabf28cad96c014eb85666ad22c1e6ecf5"
CANDIDATE_SHA = "839db7d67af2d67eb9fe2c2dd6f86cc2a3723f034fe3b5a0e97b58152d4a1c27"
LAYOUTS = list(range(929336, 929352))
ARMS = ("baseline", "candidate")
FP = ("step", "raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256")
SOURCE_MEASUREMENTS = {"gap", "eef_xyz"}


def select(record, keys):
    return {k: record[k] for k in keys}


def anchored_audit(saved):
    return {k: v for k, v in saved.items() if k not in SOURCE_MEASUREMENTS}


def adjudicate(events, physical, frames, window, observer):
    results = []
    for e in events:
        step, obj = e["step"], e["object"]
        true, held = physical[step]["goals"][obj], physical[step]["held_by_target"][obj]
        row = {**e, "posthoc_goal_true": true, "posthoc_held": held, "physically_correct": true and not held}
        if row["event"] == "in_basket":
            row.update(evidence_file=frames[step]["file"], evidence_sha256=frames[step]["sha256"])
        results.append(row)
    commits = [e for e in results if e["event"] == "in_basket"]
    correct = [e for e in commits if e["physically_correct"]]
    sauce = [e["step"] for e in correct if e["object"] == SAUCE]
    def first(items):
        return {obj: next((e["step"] for e in items if obj == e["object"]), None) for obj in GOALS}
    return {"events": results, "commits": commits, "all_commits_physically_correct": len(correct) == len(commits),
            "false_commits": [e for e in commits if not e["physically_correct"]],
            "first_commit_by_object": first(commits), "first_correct_commit_by_object": first(correct),
            "sauce_true_commit_anytime": bool(sauce),
            "sauce_true_commit_in_scored_window": any(window[0] <= step <= window[1] for step in sauce),
            "complete": sorted(observer.complete)}


def audit_case(directory, match, item, report, checked, load_frame, coarse_type, dense_type):
    def path(key, digest=None):
        return checked(directory, report[key], digest or report[key + "_sha256"])

    src_actions = list(trace(path("source_actions", item["actions_sha256"])))
    fixture = read(path("source_fixture", item["fixture_sha256"]))
    src_truth = read(path("source_physics", item["physics_audit_sha256"]))
    require(all(report[key] == digest for key, digest in (
        ("source_actions_sha256", item["actions_sha256"]), ("source_fixture_sha256", item["fixture_sha256"]),
        ("source_physics_sha256", item["physics_audit_sha256"]))), "Replay source hashes disagree with frozen attempt")
    actual = list(trace(path("actions")))
    audited = rows(path("grasp_audit"))
    end = fixture["end"]["step"]
    all_steps = src_truth["steps"]
    origin = report["initial"]
    require(src_truth["controlled_object"] == item["controlled_object"] == SAUCE
            and fixture["layout"] == item["layout_seed"] and fixture["card"] == item["card"]
            and fixture["initial"] == origin and origin["state_sha256"] == item["initial_state_sha256"]
            and fixture["frame_steps"] == item["original_frame_steps"] == [0, *[r["step"] for r in src_actions]]
            and [r["step"] for r in all_steps] == list(range(end + 1))
            and end == src_actions[-1]["step"] == actual[-1]["step"],
            "Archive does not cover the complete prescribed control trajectory")
    command, opening = (item[k] for k in ("controlled_release_command_step", "measured_open_step"))
    quality = src_truth["release_quality"]
    window = item["scored_native_steps"]
    require(fixture["release_quality"] == quality
            and quality["eligible"] == item["legacy_qualification"]
            and item["failed_checks"] == [k for k, v in quality["checks"].items() if not v]
            and quality["open_command_after_step"] == command and quality["measured_open_step"] == opening
            and window == [opening, opening + 30] and 0 <= command < opening <= end - 30
            and all_steps[command]["held"][SAUCE] and all_steps[opening]["gap"] >= .067
            and item["first_unheld_since_open_command"] == next(
                s for s in range(command, opening + 31) if not all_steps[s]["held"][SAUCE]),
            "Prespecified release-window bookkeeping disagrees with source physics")
    for record in all_steps:
        for key in ("held", "goals"):
            require(set(record[key]) == set(GOALS) and all(type(x) is bool for x in record[key].values()),
                    f"Invalid full-step source physics: {record['step']}")
        require(math.isfinite(record["gap"]), "Nonfinite source gap")
    state_steps = [0, *[r["step"] for r in actual]]
    require([r["step"] for r in audited] == [r["step"] for r in report["frames"]] == state_steps,
            "Capture, action and physical-audit endpoints differ")
    require(report["source_physics_steps"] == len(all_steps)
            and report["verified_replay_physics_captures"] == len(audited), "Incorrect archive coverage counts")
    frame_records = {r["step"]: r for r in report["frames"]}
    physical = {r["step"]: r for r in audited}
    sensors = {}
    initial = {**origin, "audit": anchored_audit(fixture["initial_audit"])}
    for native, truth in zip([initial, *actual], audited, strict=True):
        step = native["step"]
        saved, cooked = all_steps[step], frame_records[step]
        frame = load_frame(directory, cooked)
        require(select(cooked, FP) == select(native, FP)
                and cooked["file"] == f"snapshots/{step}.npz" and cooked["grasp_audit"] == truth
                and truth["state_sha256"] == native["state_sha256"] and truth["goals"] == native["audit"]["goals"] == saved["goals"]
                and truth["held_by_target"] == saved["held"] and truth["held_by_target"][SOUP] == native["audit"]["held_soup"]
                and native["audit"]["positions"] == saved["positions"]
                and cooked["grip_gap"] == frame.gap == saved["gap"]
                and cooked["eef_xyz"] == frame.geometry["eef"].tolist() == saved["eef_xyz"],
                f"Frame, native anchor or independent all-object physics differs at {step}")
        sensors[step] = frame
    coarse, dense = coarse_type(), dense_type()
    coarse.clear_view = sensors[0]
    dense.start(sensors[0], 0)
    events, calls, checks, acc = {arm: [] for arm in ARMS}, [], [], []
    controls = {}
    stream = iter(actual)
    for original in src_actions:
        s, stop = original["start_step"], original["step"]
        original_goals = [dict(step=t, object=obj, in_basket=all_steps[t]["goals"][obj])
                          for t in range(s + 1, stop + 1) for obj in GOALS
                          if all_steps[t]["goals"][obj] != all_steps[t - 1]["goals"][obj]]
        # The stored object order may be the BDDL goal order; preserve it separately below.
        require(sorted(original["transitions"], key=lambda r: (r["step"], r["object"]))
                == sorted(original_goals, key=lambda r: (r["step"], r["object"])),
                f"Original native transitions disagree with all-step physical predicates at {stop}")
        require(original["target"] is None and original["instruction"] == "physical release challenge"
                and 1 <= len(original["actions"]) <= 5 and original["completed_regrasp_steps"] == 0,
                "Source command identity, cadence or documented collector placeholder differs")
        pieces = []
        cursor = s
        while cursor < stop:
            tight = bool(dense.needs_dense)  # Recompute from sensors; never read the stored call annotations.
            width = min(1 if tight else 5, stop - cursor, CHUNK - cursor % CHUNK)
            batch = original["actions"][cursor - s:cursor - s + width]
            row = next(stream)
            step = cursor + width
            require(row["start_step"] == cursor and row["step"] == step
                    and row["actions"] == batch and row["executed"] == width
                    and row["done"] is (original["done"] if step == stop else False),
                    f"Independently requested adaptive controls differ at {step}")
            for j, action in enumerate(batch, cursor + 1):
                require(len(action) == 7 and all(math.isfinite(v) for v in action), "Invalid native control")
                controls[j] = action
            frame = sensors[step]
            events["candidate"].extend({"step": step, **e} for e in dense.observe(frame, batch, step))
            acc.extend(batch)
            if step % CHUNK == 0:
                require(len(acc) == CHUNK, "Incorrect coarse reconstruction")
                events["baseline"].extend({"step": step, **e} for e in coarse.observe(frame, acc, step))
                acc = []
            calls.append({"start_step": cursor, "step": step, "needs_dense_before": tight,
                          "executed": width, "source_anchor_step": stop, "original_10_step_sample": step % CHUNK == 0})
            cursor = step
            pieces.append(row)
        endpoint = pieces[-1]
        count = sum(r["completed_regrasp_steps"] for r in pieces)
        require(0 <= count <= stop - s, "Impossible native regrasp count")
        combined = {**endpoint, "start_step": s, "executed": sum(r["executed"] for r in pieces),
                    "actions": [a for r in pieces for a in r["actions"]],
                    "transitions": [e for r in pieces for e in r["transitions"]]}
        saved = {**original, "audit": anchored_audit(original["audit"])}
        require({k: v for k, v in combined.items() if k != "completed_regrasp_steps"}
                == {k: v for k, v in saved.items() if k != "completed_regrasp_steps"}
                and sensors[stop].gap == original["audit"]["gap"]
                and sensors[stop].geometry["eef"].tolist() == original["audit"]["eef_xyz"],
                f"Only the known literal regrasp count may differ at source anchor {stop}")
        checks.append({"step": stop, "start_step": s, "replay_calls": len(pieces),
                       "source_completed_regrasp_steps_literal": 0, "replay_completed_regrasp_steps": count,
                       "state_sha256": saved["state_sha256"], "observation_sha256": saved["observation_sha256"]})
    require(next(stream, None) is None and calls == report["calls"] and checks == report["source_anchor_checks"]
            and select(actual[-1], FP) == report["terminal"] == fixture["end"] and len(acc) == end % CHUNK
            and len(acc) == report["unobserved_trailing_baseline_actions"]
            and anchored_audit(fixture["end_audit"]) == actual[-1]["audit"], "Incomplete terminal or anchor audit")
    outcomes = {arm: adjudicate(events[arm], physical, frame_records, window, obs)
                for arm, obs in zip(ARMS, (coarse, dense), strict=True)}
    require(outcomes == report["arms"], "Saved observer events or scoring differ from independent CPU replay")
    held_open = [s for s in range(command + 1, window[1] + 1)
                 if all_steps[s]["held"][SAUCE] and controls[s][-1] < -.5]
    held_gap = [s for s in held_open if all_steps[s]["gap"] >= .067]
    dense_endpoints = {r["step"] for r in calls if r["needs_dense_before"]}
    return {"layout_seed": item["layout_seed"], "report_sha256": match["report_sha256"], "treatment": item["treatment"],
            "qualified": item["legacy_qualification"], "checked_original_anchors": len(src_actions) + 1,
            "source_physical_steps": len(all_steps), "checked_replay_frames_and_physical_captures": len(audited),
            "native_regrasp_total_uncompared_to_literal_zero": sum(c["replay_completed_regrasp_steps"] for c in checks),
            "held_opening_source_steps": held_open, "held_opening_sampled_steps": [s for s in held_open if s in sensors],
            "held_opening_gap_at_least_067_source_steps": held_gap,
            "held_opening_gap_at_least_067_sampled_steps": [s for s in held_gap if s in sensors],
            "held_opening_gap_at_least_067_adaptive_steps": [s for s in held_gap if s in dense_endpoints],
            "adaptive_single_step_calls": len(dense_endpoints),
            "arms": {arm: {k: v for k, v in outcome.items() if k != "events"} for arm, outcome in outcomes.items()}}


def summarize(cases, sources):
    positive = [r for r in cases if r["treatment"] == "sauce_inside" and r["qualified"]]
    outside = [r for r in cases if r["treatment"] == "sauce_outside" and r["qualified"]]
    require(len(positive) == 8 and len(outside) == 7 and [r["layout_seed"] for r in cases if not r["qualified"]] == [929345],
            "Changed archived positive/negative inventory")
    errors = {arm: sum(len(r["arms"][arm]["false_commits"]) for r in cases) for arm in ARMS}
    coverage = {field: {"layouts": [r["layout_seed"] for r in cases if r[field]],
                        "steps": sum(len(r[field]) for r in cases)}
                for field in cases[0] if field.startswith("held_opening_")}
    return {"schema": "v2-sauce-controls-audit/1", "audit_sha256": sha(Path(__file__)),
            "scope": "16 previously seen, full controlled trajectories; no deployed policy or independent success rate",
            "plan_sha256": PLAN_SHA, "candidate_sha256": CANDIDATE_SHA, "observer_sha256": OBSERVER_SHA,
            "runner_sha256": RUNNER_SHA, "sources": sources, "layouts_checked": [r["layout_seed"] for r in cases],
            "wrong_or_held_candidate_commits": errors["candidate"], "wrong_or_held_baseline_commits": errors["baseline"],
            "coverage": {"qualified_sauce_inside": len(positive), "qualified_sauce_outside": len(outside),
                         "unqualified_included_for_safety": [929345],
                         "sauce_hit_layouts_within_scored_window": {
                             arm: [r["layout_seed"] for r in positive if r["arms"][arm]["sauce_true_commit_in_scored_window"]]
                             for arm in ARMS},
                         "sauce_hit_layouts_anytime": {
                             arm: [r["layout_seed"] for r in positive if r["arms"][arm]["sauce_true_commit_anytime"]]
                             for arm in ARMS},
                         "held_opening_negatives": coverage},
            "development_online_ready": len(cases) == 16 and errors["candidate"] == 0,
            "independent_pilot_release": False, "cases": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=HERE / "stage4-sauce-controls-plan.json")
    a = p.parse_args()
    root, dest = a.run.resolve(), a.out.resolve()
    require(not dest.exists() and sha(a.manifest) == PLAN_SHA and sha(HERE / "sauce_controls_replay.py") == RUNNER_SHA
            and sha(HERE / "actual_physics_worker.py") == PHYSICS_WORKER_SHA
            and sha(HERE / "actual_physics_simulator.py") == PHYSICS_SIMULATOR_SHA,
            "Existing output or changed replay producer/source")
    lock = read(a.manifest)
    require([r["layout_seed"] for r in lock["attempts"]] == LAYOUTS, "Changed source assignment")
    from release_filter_analysis import checked, load_frame

    base = load_exact("sauce_controls_independent_baseline", PLACEMENT / "observer.py", OBSERVER_SHA)
    candidate = load_exact("sauce_controls_independent_adaptive", HERE / "dense_release_observer.py", CANDIDATE_SHA)
    expected = {"source_manifest_sha256": PLAN_SHA, "candidate_sha256": CANDIDATE_SHA, "observer_sha256": OBSERVER_SHA,
                "source_sha256": RUNNER_SHA, "bddl_sha256": lock["bddl_sha256"], "physics_worker_sha256": PHYSICS_WORKER_SHA,
                "physics_simulator_sha256": PHYSICS_SIMULATOR_SHA}
    results, sources = [], []
    for card in GPU:
        index_path = root / f"sauce-controls-{card}.json"
        index = read(index_path)
        plan_path = checked(root, f"sauce-controls-plan-{card}.json", index["plan_sha256"])
        per_card = read(plan_path)
        assigned = [r for r in lock["attempts"] if r["card"] == card]
        require(index["schema"] == "v2-sauce-controls-replay-manifest/1" and index["complete"] is True
                and index["card"] == per_card["card"] == card and index["source_manifest_sha256"] == PLAN_SHA
                and per_card["schema"] == "v2-sauce-controls-replay-plan/1"
                and all(per_card[k] == v for k, v in expected.items())
                and per_card["attempts"] == assigned
                and [r["layout_seed"] for r in index["records"]] == [r["layout_seed"] for r in assigned],
                f"Incomplete or changed card {card} provenance")
        sources.append({"card": card, "manifest_sha256": sha(index_path), "plan_sha256": sha(plan_path)})
        for entry, attempt in zip(index["records"], assigned, strict=True):
            seed = entry["layout_seed"]
            require(entry["report"] == f"{seed}/report.json", "Unexpected report path")
            path = checked(root, entry["report"], entry["report_sha256"])
            report = read(path)
            require(report["schema"] == "v2-sauce-controls-replay/1" and report["layout_seed"] == seed
                    and report["card"] == card and report["plan_sha256"] == sha(plan_path)
                    and report["attempt"] == attempt and all(report[k] == v for k, v in expected.items()),
                    f"Report provenance changed: {seed}")
            results.append(audit_case(path.parent, entry, attempt, report, checked, load_frame,
                                      base.PlacementObserver, candidate.DenseReleaseObserver))
            print("AUDIT_SAUCE_CONTROL", seed, flush=True)
    results.sort(key=lambda r: r["layout_seed"])
    require([r["layout_seed"] for r in results] == LAYOUTS, "A complete full-trajectory audit requires all 16 cases")
    write(dest, summarize(results, sources))


if __name__ == "__main__":
    main()
