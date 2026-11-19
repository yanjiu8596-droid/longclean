"""Replay all 16 archived sauce controls; score two sensor-only observers."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from shutil import copyfile

from actual_progress_replay import (HERE, OBSERVER_SHA, PHYSICS_SIMULATOR_SHA, PHYSICS_WORKER_SHA,
                                    PLACEMENT, join_grasp_audit, load_exact)
from common import GOALS, SAUCE, read, sha, write
from legacy_parity import trace
from stage4_fixture import GPU, require, validate_gpu, validate_study
from target_follow_audit import observation


MANIFEST_SHA = "e8014407a15daa4c2a2cdccb3482bdfad7bc3a4f5ce42f10fbd05dc2f3559e17"
CANDIDATE_SHA = "839db7d67af2d67eb9fe2c2dd6f86cc2a3723f034fe3b5a0e97b58152d4a1c27"
METRIC_SCOPE = ("Exact actions, fingerprints, native audit, transitions and termination at original anchors; "
                "collector completed_regrasp_steps is a literal zero, so native replay regrasp counts are retained separately, not equated.")


def checked(root, name, digest):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve())
            and path.is_file() and sha(path) == digest, f"Changed or unsafe source: {name}")
    return path


def archive(api, worker, frame, dest):
    step = worker.current["step"]
    name = f"snapshots/{step}.npz"
    snapshot = dest / name
    copyfile(worker.npz, snapshot)
    fingerprint = api.fingerprint(worker.current)
    observation(snapshot, fingerprint)
    return {**fingerprint, "file": name, "sha256": sha(snapshot), "grip_gap": frame.gap,
            "eef_xyz": frame.geometry["eef"].tolist()}


def compare_audit(native, saved, frame):
    require(native["audit"] == {k: v for k, v in saved.items() if k not in {"gap", "eef_xyz"}}
            and frame.gap == saved["gap"] and frame.geometry["eef"].tolist() == saved["eef_xyz"],
            f"Source native audit or measured sensors differ at {native['step']}")


def anchor(original, pieces, frame):
    endpoint = pieces[-1]
    compare_audit(endpoint, original["audit"], frame)
    combined = {**endpoint, "start_step": original["start_step"],
                "actions": [a for r in pieces for a in r["actions"]],
                "executed": sum(r["executed"] for r in pieces),
                "transitions": [e for r in pieces for e in r["transitions"]]}
    source = {**original, "audit": {k: v for k, v in original["audit"].items() if k not in {"gap", "eef_xyz"}}}
    require(original["completed_regrasp_steps"] == 0, "Source collector regrasp metric was not the expected literal")
    require({k: v for k, v in combined.items() if k != "completed_regrasp_steps"}
            == {k: v for k, v in source.items() if k != "completed_regrasp_steps"},
            f"Reassembled source anchor differs at {original['step']}")
    return {"step": original["step"], "start_step": original["start_step"], "replay_calls": len(pieces),
            "source_completed_regrasp_steps_literal": 0,
            "replay_completed_regrasp_steps": sum(r["completed_regrasp_steps"] for r in pieces),
            "state_sha256": endpoint["state_sha256"], "observation_sha256": endpoint["observation_sha256"]}


def score(events, physical, frame_records, window):
    frames = {r["step"]: r for r in frame_records}
    for event in events:
        row, obj = physical[event["step"]], event["object"]
        event.update(posthoc_goal_true=row["goals"][obj], posthoc_held=row["held_by_target"][obj],
                     physically_correct=bool(row["goals"][obj] and not row["held_by_target"][obj]))
        if event["event"] == "in_basket":
            event.update(evidence_file=frames[event["step"]]["file"], evidence_sha256=frames[event["step"]]["sha256"])
    commits = [e for e in events if e["event"] == "in_basket"]
    sauce = [e for e in commits if e["object"] == SAUCE]
    return {"events": events, "commits": commits, "all_commits_physically_correct": all(e["physically_correct"] for e in commits),
            "false_commits": [e for e in commits if not e["physically_correct"]],
            "first_commit_by_object": {obj: next((e["step"] for e in commits if e["object"] == obj), None) for obj in GOALS},
            "first_correct_commit_by_object": {obj: next((e["step"] for e in commits if e["object"] == obj and e["physically_correct"]), None) for obj in GOALS},
            "sauce_true_commit_anytime": any(e["physically_correct"] for e in sauce),
            "sauce_true_commit_in_scored_window": any(e["physically_correct"] and window[0] <= e["step"] <= window[1] for e in sauce)}


def replay(api, worker_type, base_type, candidate_type, bddl, source_root, attempt, dest):
    fixture_path = checked(source_root, attempt["fixture"], attempt["fixture_sha256"])
    physics_path = checked(source_root, attempt["physics_audit"], attempt["physics_audit_sha256"])
    actions_path = checked(source_root, attempt["actions"], attempt["actions_sha256"])
    fixture, source_physics = read(fixture_path), read(physics_path)
    originals = list(trace(actions_path))
    require(fixture["layout"] == attempt["layout_seed"] and fixture["initial"]["step"] == 0
            and fixture["frame_steps"] == attempt["original_frame_steps"] == [0, *[r["step"] for r in originals]]
            and source_physics["controlled_object"] == SAUCE, "Changed control fixture")
    baseline, candidate = base_type(), candidate_type()
    events, native, frames, calls, checks, buffer = {"baseline": [], "candidate": []}, [], [], [], [], []
    with worker_type(bddl, dest, attempt["layout_seed"]) as worker:
        (dest / "snapshots").mkdir()
        initial = worker.current
        require(api.fingerprint(initial) == fixture["initial"], "Initial physical control state differs")
        frame = replace(api.Frame.read(worker.npz), source=None)
        compare_audit(initial, fixture["initial_audit"], frame)
        baseline.clear_view = frame
        candidate.start(frame, 0)
        frames.append(archive(api, worker, frame, dest))
        for original in originals:
            require(original["start_step"] == worker.current["step"] and 1 <= len(original["actions"]) <= 5
                    and original["target"] is None and original["instruction"] == "physical release challenge",
                    "Unexpected source action sequence")
            offset, pieces = 0, []
            while offset < len(original["actions"]):
                before = worker.current["step"]
                dense = bool(candidate.needs_dense)
                count = min(1 if dense else 5, len(original["actions"]) - offset, 10 - before % 10)
                request = original["actions"][offset:offset + count]
                require(not worker.terminal(), "Native simulator stopped before archived controls ended")
                realized = worker.step(request, original["target"], original["instruction"])
                require(realized == request, "Native simulator truncated archived controls")
                current = worker.current
                row = {"start_step": before, "target": original["target"], "instruction": original["instruction"],
                       "actions": realized, **current}
                native.append(row)
                pieces.append(row)
                offset += count
                frame = replace(api.Frame.read(worker.npz), source=None)
                frames.append(archive(api, worker, frame, dest))
                buffer.extend(realized)
                step = current["step"]
                events["candidate"].extend({"step": step, **e} for e in candidate.observe(frame, realized, step))
                if step % 10 == 0:
                    require(len(buffer) == 10, "Baseline original action cadence differs")
                    events["baseline"].extend({"step": step, **e} for e in baseline.observe(frame, buffer, step))
                    buffer = []
                calls.append({"start_step": before, "step": step, "needs_dense_before": dense,
                              "executed": count, "source_anchor_step": original["step"], "original_10_step_sample": step % 10 == 0})
            checks.append(anchor(original, pieces, frame))
        terminal = api.fingerprint(worker.current)
        require(terminal == fixture["end"] and len(buffer) == terminal["step"] % 10, "Final source endpoint or trailing block differs")
        compare_audit(worker.current, fixture["end_audit"], frame)
        complete = {"baseline": sorted(baseline.complete), "candidate": sorted(candidate.complete)}
    action_path, grasp_path = dest / "actions.jsonl", dest / "grasp-audit.jsonl"
    require(list(trace(action_path)) == native, "Saved adaptive replay actions differ")
    join_grasp_audit(grasp_path, native, initial, [])
    grasp = list(map(json.loads, grasp_path.read_text(encoding="utf-8").splitlines()))
    physics = {r["step"]: r for r in grasp}
    old = {r["step"]: r for r in source_physics["steps"]}
    require(list(old) == list(range(terminal["step"] + 1)), "Source per-step physics missing")
    for current, frame in zip([initial, *native], frames, strict=True):
        step = current["step"]
        saved, actual = old[step], physics[step]
        require(actual["held_by_target"] == saved["held"] and current["audit"]["goals"] == saved["goals"]
                and current["audit"]["positions"] == saved["positions"] and frame["grip_gap"] == saved["gap"]
                and frame["eef_xyz"] == saved["eef_xyz"], f"Source per-step physics differs at {step}")
        frame["grasp_audit"] = actual
    for name, source in (("source-actions.jsonl", actions_path), ("source-physics-audit.json", physics_path), ("source-fixture.json", fixture_path)):
        copyfile(source, dest / name)
    arms = {name: {**score(events[name], physics, frames, attempt["scored_native_steps"]), "complete": complete[name]}
            for name in ("baseline", "candidate")}
    return {"attempt": attempt, "initial": api.fingerprint(initial), "terminal": terminal, "frames": frames,
            "actions": action_path.name, "actions_sha256": sha(action_path), "grasp_audit": grasp_path.name,
            "grasp_audit_sha256": sha(grasp_path), "source_actions": "source-actions.jsonl",
            "source_actions_sha256": sha(dest / "source-actions.jsonl"), "source_physics": "source-physics-audit.json",
            "source_physics_sha256": sha(dest / "source-physics-audit.json"), "source_fixture": "source-fixture.json",
            "source_fixture_sha256": sha(dest / "source-fixture.json"), "source_anchor_checks": checks, "calls": calls,
            "arms": arms, "unobserved_trailing_baseline_actions": len(buffer), "source_metric_scope": METRIC_SCOPE,
            "source_physics_steps": len(old), "verified_replay_physics_captures": len(grasp)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--card", required=True, type=int, choices=GPU)
    cli.add_argument("--study", required=True, type=Path)
    cli.add_argument("--manifest", type=Path, default=HERE / "stage4-sauce-controls-plan.json")
    cli.add_argument("--source-root", type=Path, default=HERE.parent)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    source, out = args.source_root.resolve(), args.output.resolve()
    require(args.manifest.is_file() and sha(args.manifest) == MANIFEST_SHA, "Changed frozen 16-attempt manifest")
    manifest = read(args.manifest)
    require([r["layout_seed"] for r in manifest["attempts"]] == list(range(929336, 929352)), "Prespecified attempt missing")
    assigned = [r for r in manifest["attempts"] if r["card"] == args.card]
    require(len(assigned) == 8 and out != source and out.name != "stage4-pilot", "Unexpected assignment or output")
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    require(sha(bddl) == manifest["bddl_sha256"], "Control BDDL differs")
    base = load_exact("sauce_controls_frozen_placement", PLACEMENT / "observer.py", OBSERVER_SHA)
    candidate = load_exact("sauce_controls_dense_candidate", HERE / "dense_release_observer.py", CANDIDATE_SHA)
    physics = load_exact("sauce_controls_physics", HERE / "actual_physics_worker.py", PHYSICS_WORKER_SHA)
    require(sha(HERE / "actual_physics_simulator.py") == PHYSICS_SIMULATOR_SHA, "Changed physical audit worker")
    out.mkdir(parents=True, exist_ok=True)
    plan_path, marker = (out / f"sauce-controls{suffix}-{args.card}.json" for suffix in ("-plan", ""))
    require(not plan_path.exists() and not marker.exists() and all(not (out / str(r["layout_seed"])).exists() for r in assigned),
            "Existing sauce controls output")
    plan = {"schema": "v2-sauce-controls-replay-plan/1", "card": args.card, "scope": "previously_seen_development_controls",
            "source_manifest_sha256": MANIFEST_SHA, "source_sha256": sha(Path(__file__)), "bddl_sha256": sha(bddl),
            "observer_sha256": OBSERVER_SHA, "candidate_sha256": CANDIDATE_SHA, "physics_worker_sha256": PHYSICS_WORKER_SHA,
            "physics_simulator_sha256": PHYSICS_SIMULATOR_SHA, "attempts": assigned, "source_metric_scope": METRIC_SCOPE}
    write(plan_path, plan)
    progress = {"schema": "v2-sauce-controls-replay-manifest/1", "card": args.card, "plan_sha256": sha(plan_path),
                "source_manifest_sha256": MANIFEST_SHA, "complete": False, "records": []}
    write(marker, progress)
    import run as api
    worker = physics.worker_class(api)
    for attempt in assigned:
        layout = attempt["layout_seed"]
        dest = out / str(layout)
        report = {**{k: v for k, v in plan.items() if k != "attempts"}, "schema": "v2-sauce-controls-replay/1",
                  "layout_seed": layout, "plan_sha256": sha(plan_path),
                  **replay(api, worker, base.PlacementObserver, candidate.DenseReleaseObserver, bddl, source, attempt, dest)}
        path = dest / "report.json"
        write(path, report)
        progress["records"].append({"layout_seed": layout, "report": path.relative_to(out).as_posix(), "report_sha256": sha(path)})
        write(marker, progress)
        print("SAUCE_CONTROL", layout, {arm: report["arms"][arm]["first_commit_by_object"] for arm in report["arms"]}, flush=True)
    progress["complete"] = True
    write(marker, progress)


if __name__ == "__main__":
    main()
