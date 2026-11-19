"""CPU regression for conditional sauce sampling on four archived seen windows."""
import argparse
from pathlib import Path

from dense_release_observer import DenseReleaseObserver, frozen_module
from actual_progress_replay import ActualProgress, OBSERVER_SHA, boundary
from common import CHEESE, CHUNK, SAUCE, SOUP
from release_dense_replay import PRIOR_AUDIT_SHA, aggregate, rows
from stage4_fixture import read, require, sha, write


HERE = Path(__file__).resolve().parent
DENSE_RUNNER_SHA = "b9dfebf929e6591c0466ba7843241de2c664a73ca961787dccdf7eb501a534f4"
CASES = {2: (931000, 931200), 3: (931005, 931201)}
EXPECTED = {931000: (244, [0, 48]), 931005: (236, [0, 86]), 931200: (224, [31, 123]), 931201: None}
EXPECT_COARSE = {931000: None, 931005: None, 931200: (230, [4, 143]), 931201: None}
SCOPE = ("Only four already-seen clips through their archived release-window endpoint. "
         "Observer events and hypothetical schedules were replayed; no new policy action or whole episode was evaluated.")


class CheesePrefix:
    def complete(self, obj):
        return obj == CHEESE


def audited(events, step, physical):
    record = physical[step]
    return [{"step": step, **e, "predicate_true": record["goals"][e["object"]],
             "held_by_target": record["held_by_target"][e["object"]]} for e in events]


def decision_at(schedule, step, physical):
    return boundary(schedule, {"step": step, "audit": {"goals": physical[step]["goals"]}})


def replay_case(directory, report, loader, project, checked, schedule_type):
    def jsonl(key):
        return rows(checked(directory, report[key], report[key + "_sha256"]))

    fork = report["fork"]["step"]
    p, first, end = (report["window"][k] for k in ("pending_step", "start_step", "end_step"))
    require(report["window"]["stride"] == 1 and (first, end) == (p - 10, p + 30)
            and (end - fork) % CHUNK == 0 and report["observer_stride"] == CHUNK,
            "Changed release-window schedule")
    archive = report["frames"]
    require([s["step"] for s in archive] == [*range(fork, first, 5), *range(first, end + 1)],
            "Missing original or dense evidence")
    native, source, physics = (jsonl(key) for key in ("actions", "source_anchors", "grasp_audit"))
    frame = {s["step"]: loader(directory, s) for s in archive}
    # Reassemble source intervals independently of the new observer replay.
    fresh, old = iter(native), []
    for original in source:
        sub = [next(fresh)]
        while sub[-1]["step"] < original["step"]:
            sub.append(next(fresh))
        old.append(aggregate(original, sub))
    require(next(fresh, None) is None and old == report["source_anchor_checks"]
            and source[-1]["step"] == end and native[-1]["step"] == end,
            "Dense controls differ from original archived source")
    physical = {r["step"]: r for r in physics}
    native_at = {r["step"]: r for r in native}
    require([0, *native_at] == list(physical), "Physical audit is not aligned to dense history")
    for s in archive:
        step = s["step"]
        check = physical[step]
        n = native_at[step]
        require(s["grasp_audit"] == check and n["audit"]["goals"] == check["goals"]
                and s["state_sha256"] == check["state_sha256"] == n["state_sha256"]
                and s["observation_sha256"] == n["observation_sha256"]
                and frame[step].gap == s["grip_gap"], f"Observation provenance changed at {step}")
    factual_actions = [a for r in source if r["start_step"] >= fork for a in r["actions"]]
    require(len(factual_actions) == end - fork, "Missing original suffix controls")

    coarse, revised = frozen_module().PlacementObserver(), DenseReleaseObserver()
    coarse.clear_view = frame[fork]
    revised.start(frame[fork], fork)
    coarse_plan = schedule_type(ActualProgress(CheesePrefix(), coarse), True, None)
    revised_plan = schedule_type(ActualProgress(CheesePrefix(), revised), True, None)
    old_boundaries = [decision_at(coarse_plan, fork, physical)]
    new_boundaries = [decision_at(revised_plan, fork, physical)]
    old_events, new_events, calls = [], [], []
    for step in range(fork + CHUNK, end + 1, CHUNK):
        chunk = factual_actions[step - fork - CHUNK:step - fork]
        old_events.extend(audited(coarse.observe(frame[step], chunk, step), step, physical))
        old_boundaries.append(decision_at(coarse_plan, step, physical))
    stripped = [{k: v for k, v in e.items() if k not in {"predicate_true", "held_by_target"}} for e in old_events]
    require(stripped == project(report["observer_events"]), "Frozen original event stream changed")
    step = fork
    while step < end:
        requested = revised.needs_dense
        width = 1 if requested else CHUNK - (step - fork) % CHUNK
        stop = step + width
        require(stop <= end and stop in frame, f"No archived frame for adaptive request {step}..{stop}")
        actions = factual_actions[step - fork:stop - fork]
        events = revised.observe(frame[stop], actions, stop)
        new_events.extend(audited(events, stop, physical))
        calls.append({"from_step": step, "step": stop, "single_step_requested": requested})
        step = stop
        if (step - fork) % CHUNK == 0:
            new_boundaries.append(decision_at(revised_plan, step, physical))
    pairs = [{"step": b["step"], "baseline": b, "candidate": c}
             for b, c in zip(old_boundaries, new_boundaries, strict=True)]
    old_commits = [e for e in old_events if e["event"] == "in_basket"]
    new_commits = [e for e in new_events if e["event"] == "in_basket"]
    require(all(e["object"] == SAUCE and e["predicate_true"] and not e["held_by_target"]
                for e in old_commits + new_commits), "Unexpected false, held or other-object write in archived clip")
    require(next(e["step"] for e in new_events if e["event"] == "pending_release" and e["object"] == SAUCE) == p,
            "Revised first transport/release estimate changed")
    layout = report["layout_seed"]
    for commits, expect in ((old_commits, EXPECT_COARSE[layout]), (new_commits, EXPECTED[layout])):
        require([(e["step"], e["pixels"]) for e in commits] == ([] if expect is None else [expect]),
                f"Unexpected frozen commit signature for layout {layout}")
    result = None
    if new_commits:
        hit = new_commits[0]
        next_boundary = next(p for p in pairs if p["step"] >= hit["step"])
        before = [p for p in pairs if p["step"] < hit["step"]][-1]
        require(SAUCE in before["candidate"]["order"] and SAUCE not in next_boundary["candidate"]["order"],
                "Revised placement did not update next original decision")
        result = {"event": hit, "boundary_before": before["candidate"], "boundary_at_or_after": next_boundary,
                  "changed_sauce_eligibility": SAUCE in next_boundary["baseline"]["order"],
                  "changed_selected_target": next_boundary["baseline"]["target"] != next_boundary["candidate"]["target"],
                  "evaluation_soup_still_pending": not physical[next_boundary["step"]]["goals"][SOUP]}
    return {"layout_seed": layout, "fork": fork, "window": report["window"], "archive_frames": len(archive),
            "baseline_events": old_events, "candidate_events": new_events, "candidate_calls": calls,
            "candidate_single_step_calls": sum(r["single_step_requested"] for r in calls),
            "commit": result, "schedule_boundaries": pairs,
            "target_differences": [x for x in pairs if x["baseline"]["target"] != x["candidate"]["target"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    directory, out = args.run.resolve(), args.out.resolve()
    require(not out.exists() and sha(HERE / "release_dense_replay.py") == DENSE_RUNNER_SHA,
            "Existing regression output or changed archive producer")
    from release_filter_analysis import checked, load_frame, project
    from vision import Schedule

    records = []
    for card, layouts in CASES.items():
        manifest_path = directory / f"release-dense-{card}.json"
        manifest = read(manifest_path)
        plan_path = checked(directory, f"release-dense-plan-{card}.json", manifest["plan_sha256"])
        plan = read(plan_path)
        require(manifest["schema"] == "v2-stage4-release-dense-manifest/1" and manifest["complete"] is True
                and manifest["card"] == plan["card"] == card
                and [r["layout_seed"] for r in manifest["records"]] == list(layouts)
                and plan["source_sha256"] == DENSE_RUNNER_SHA and plan["observer_sha256"] == OBSERVER_SHA
                and plan["prior_audit_sha256"] == PRIOR_AUDIT_SHA,
                f"Archive manifest changed for card {card}")
        for item in manifest["records"]:
            report_path = checked(directory, item["report"], item["report_sha256"])
            report = read(report_path)
            require(report["schema"] == "v2-stage4-release-dense/1" and report["plan_sha256"] == sha(plan_path)
                    and report["source_sha256"] == DENSE_RUNNER_SHA
                    and report["layout_seed"] == item["layout_seed"] and report["prior_audit_sha256"] == PRIOR_AUDIT_SHA,
                    "Wrong archived input layout")
            checked_run = replay_case(report_path.parent, report, load_frame, project, checked, Schedule)
            records.append({**item, "manifest_sha256": sha(manifest_path), **checked_run})
            print("DENSE_RELEASE_REGRESSION", report["layout_seed"], checked_run["commit"], flush=True)
    require(sum(r["window"]["end_step"] - r["window"]["start_step"] + 1 for r in records) == 164,
            "Did not evaluate the four full dense windows")
    write(out, {"schema": "v2-dense-release-regression/1", "scope": SCOPE,
                "source_sha256": sha(Path(__file__)), "wrapper_sha256": sha(HERE / "dense_release_observer.py"),
                "observer_sha256": OBSERVER_SHA, "dense_runner_sha256": DENSE_RUNNER_SHA,
                "expected_archived_signatures_match": True, "incorrect_or_held_writes_in_archived_prefixes": 0,
                "cases": records, "full_trajectory_safety": False, "online_release": False,
                "independent_pilot_release": False})


if __name__ == "__main__":
    main()
