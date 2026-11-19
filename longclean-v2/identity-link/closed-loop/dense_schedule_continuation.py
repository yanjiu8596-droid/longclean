"""Continue two sealed sensor replays to the archived next target decision, on CPU."""
import argparse
from pathlib import Path

import dense_release_regression as regression
from actual_progress_replay import ActualProgress
from common import CHUNK, SAUCE, SOUP, read, sha, write
from dense_release_observer import DenseReleaseObserver, frozen_module
from release_filter_analysis import checked, load_frame, project
from release_dense_replay import rows
from stage4_fixture import require
from vision import Schedule


HERE = Path(__file__).resolve().parent
REGRESSION_SHA = "024410dda19a86fd30788cc1ab1040cb7e7a94f875eee534aa3360b2fa62ac6a"
REGRESSION_CODE_SHA = "45fe8af63157ce4aa1db094891c905977314018fce319bfae980ae73248574aa"
WINDOW_ANALYSIS_SHA = "916e6202854d4221ac69143a44680bcacf64a0a56eae41a94fd9f0026cd825b8"
CANDIDATE_SHA = "839db7d67af2d67eb9fe2c2dd6f86cc2a3723f034fe3b5a0e97b58152d4a1c27"
LAYOUTS = (931000, 931005)


def table(directory, report, field):
    return rows(checked(directory, report[field], report[field + "_sha256"]))


def continue_case(dense_dir, dense, window_dir, window, previous, sealed):
    # First independently reproduce the exact already-approved dense clip.
    original = regression.replay_case(dense_dir, dense, load_frame, project, checked, Schedule)
    require(all(original[k] == sealed[k] for k in original), "Sealed dense regression no longer reproduces")
    old_actions = table(window_dir, window, "actions")
    old_physics = table(window_dir, window, "grasp_audit")
    dense_anchors = table(dense_dir, dense, "source_anchors")
    require(old_actions[:len(dense_anchors)] == dense_anchors and dense["source"] == window["source"]
            and dense["fork"] == window["fork"] == previous["fork"], "Continuation is not the same source prefix")
    fork, cut, end = dense["fork"]["step"], dense["window"]["end_step"], window["window"]["end_step"]
    require(end == dense["window"]["pending_step"] + 40 and end == cut + 10 and end % CHUNK == 0,
            "Expected only the next archived ten-step block")
    physical = {r["step"]: r for r in table(dense_dir, dense, "grasp_audit")}
    for row in old_physics:
        step = row["step"]
        if step in physical:
            require(row == physical[step], "Original and dense physical anchor differ")
        physical[step] = row
    source_at = {r["step"]: r for r in old_actions}
    archive = {r["step"]: r for r in dense["frames"]}
    frames = {s: load_frame(dense_dir, row) for s, row in archive.items()}
    additions = []
    for row in window["frames"]:
        step = row["step"]
        if step in archive:
            require(all(row[k] == archive[step][k] for k in ("state_sha256", "observation_sha256", "grasp_audit", "grip_gap")),
                    "Original five-step and dense observation anchor differ")
        if step <= cut:
            continue
        require(step in (cut + 5, end) and row["grasp_audit"] == physical[step]
                and row["state_sha256"] == source_at[step]["state_sha256"] == physical[step]["state_sha256"]
                and row["observation_sha256"] == source_at[step]["observation_sha256"]
                and source_at[step]["audit"]["goals"] == physical[step]["goals"], "Unverified continuation frame")
        frames[step] = load_frame(window_dir, row)
        require(frames[step].gap == row["grip_gap"], "Continuation sensor gap changed")
        additions.append({"step": step, "file": row["file"], "sha256": row["sha256"],
                          "state_sha256": row["state_sha256"], "observation_sha256": row["observation_sha256"]})
    require([r["step"] for r in additions] == [cut + 5, end], "Missing continuation frame")
    factual = [a for r in old_actions if r["start_step"] >= fork for a in r["actions"]]
    require(len(factual) == end - fork, "Missing exact source actions")
    baseline, candidate = frozen_module().PlacementObserver(), DenseReleaseObserver()
    baseline.clear_view = frames[fork]
    candidate.start(frames[fork], fork)
    planners = {"baseline": Schedule(ActualProgress(regression.CheesePrefix(), baseline), True, None),
                "candidate": Schedule(ActualProgress(regression.CheesePrefix(), candidate), True, None)}
    decisions = {arm: [regression.decision_at(planner, fork, physical)] for arm, planner in planners.items()}
    events, calls = {"baseline": [], "candidate": []}, []
    for step in range(fork + CHUNK, end + 1, CHUNK):
        block = factual[step - fork - CHUNK:step - fork]
        events["baseline"].extend(regression.audited(baseline.observe(frames[step], block, step), step, physical))
        decisions["baseline"].append(regression.decision_at(planners["baseline"], step, physical))
    sensor_events = [{k: v for k, v in e.items() if k != "predicate_true"} for e in events["baseline"]]
    require(project(sensor_events) == project(window["observer_events"]), "Extended original observer history differs")
    step = fork
    while step < end:
        requested = candidate.needs_dense
        stop = step + (1 if requested else CHUNK - (step - fork) % CHUNK)
        require(stop <= end and stop in frames, "Candidate requests an uncaptured observation; cannot extrapolate")
        result = candidate.observe(frames[stop], factual[step - fork:stop - fork], stop)
        events["candidate"].extend(regression.audited(result, stop, physical))
        calls.append({"from_step": step, "step": stop, "single_step_requested": requested})
        step = stop
        if (step - fork) % CHUNK == 0:
            decisions["candidate"].append(regression.decision_at(planners["candidate"], step, physical))
    for arm in ("baseline", "candidate"):
        require([e for e in events[arm] if e["step"] <= cut] == sealed[arm + "_events"], "Earlier event history changed")
    pairs = [{"step": b["step"], "baseline": b, "candidate": c}
             for b, c in zip(decisions["baseline"], decisions["candidate"], strict=True)]
    require([r for r in pairs if r["step"] <= cut] == sealed["schedule_boundaries"]
            and [r for r in calls if r["step"] <= cut] == sealed["candidate_calls"], "Earlier scheduling/cadence changed")
    source_coarse = [r for r in previous["arms"]["candidate"]["schedule_boundaries"] if r["step"] <= end]
    require(decisions["baseline"] == source_coarse, "Frozen placement schedule differs from original full source report")
    source_legacy, = [r for r in previous["arms"]["baseline"]["schedule_boundaries"] if r["step"] == 270]
    at270, = [r for r in pairs if r["step"] == 270]
    require(at270["baseline"]["target"] == source_legacy["target"] == SAUCE
            and at270["candidate"]["target"] == SOUP and SAUCE not in at270["candidate"]["order"]
            and not physical[270]["goals"][SOUP], "No verified sauce-to-soup target difference at step 270")
    commits = [e for e in events["candidate"] if e["event"] == "in_basket"]
    require(all(e["predicate_true"] and not e["held_by_target"] for e in commits), "Incorrect or held candidate assertion")
    differences = [r for r in pairs if r["baseline"]["target"] != r["candidate"]["target"]]
    return {"layout_seed": dense["layout_seed"], "fork_step": fork, "previous_cutoff": cut, "continued_to_step": end,
            "added_frames": additions, "baseline_events": events["baseline"], "candidate_events": events["candidate"],
            "candidate_calls": calls, "schedule_boundaries": pairs, "first_selected_target_difference": differences[0],
            "verified_step_270": at270, "original_assigned_rgbd_step_270": source_legacy,
            "hypothetical_only": True, "new_policy_actions_executed": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "window", "prior", "out"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    run, window, prior, out = (getattr(args, name).resolve() for name in ("run", "window", "prior", "out"))
    require(not out.exists() and sha(HERE / "dense_release_regression.py") == REGRESSION_CODE_SHA
            and sha(HERE / "dense_release_observer.py") == CANDIDATE_SHA, "Changed frozen code or existing output")
    regression_file, analysis_file = run / "regression.json", window / "analysis.json"
    require(sha(regression_file) == REGRESSION_SHA and sha(analysis_file) == WINDOW_ANALYSIS_SHA, "Changed sealed input reports")
    regression_data, analyzed = read(regression_file), read(analysis_file)
    results = []
    for layout in LAYOUTS:
        sealed, = [r for r in regression_data["cases"] if r["layout_seed"] == layout]
        previous_window, = [r for r in analyzed["cases"] if r["layout_seed"] == layout]
        dense_path = checked(run, sealed["report"], sealed["report_sha256"])
        window_path = checked(window, previous_window["report"], previous_window["report_sha256"])
        dense, old_window = read(dense_path), read(window_path)
        require(dense["prior_report_sha256"] == old_window["prior_report_sha256"] == previous_window["prior_report_sha256"],
                "Prior full replay link differs")
        prior_path = checked(prior, old_window["prior_report"], old_window["prior_report_sha256"])
        result = continue_case(dense_path.parent, dense, window_path.parent, old_window, read(prior_path), sealed)
        result["inputs"] = {"dense_report_sha256": sha(dense_path), "window_report_sha256": sha(window_path),
                            "prior_report_sha256": sha(prior_path), "source_actions_sha256": old_window["actions_sha256"],
                            "source_grasp_audit_sha256": old_window["grasp_audit_sha256"]}
        results.append(result)
        print("DENSE_SCHEDULE_CONTINUATION", layout, result["first_selected_target_difference"], flush=True)
    write(out, {"schema": "v2-dense-schedule-continuation/1", "source_sha256": sha(Path(__file__)),
                "regression_sha256": REGRESSION_SHA, "window_analysis_sha256": WINDOW_ANALYSIS_SHA,
                "candidate_sha256": CANDIDATE_SHA, "cases": results, "different_layouts_with_selected_target_proof": len(results),
                "scope": "Two seen trajectories, continued using existing archived sensors and identical controls. Hypothetical schedules only.",
                "online_release": False, "independent_pilot_release": False})


if __name__ == "__main__":
    main()
