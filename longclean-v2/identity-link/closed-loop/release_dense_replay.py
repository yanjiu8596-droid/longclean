"""Replay four seen releases and capture the frozen window at every control step."""
import argparse
import json
from pathlib import Path

from actual_progress_replay import (
    AUDIT_SHA as SOURCE_AUDIT_SHA, HERE, OBSERVER_SHA, PHYSICS_SIMULATOR_SHA, PHYSICS_WORKER_SHA,
    PLACEMENT, PROMPT, SUBSTRIDE, checked, event_row, join_grasp_audit, load_exact, source_card,
)
from common import CHEESE, CHUNK, GOALS, PREFIX_TASK, SOUP
from legacy_parity import trace
from release_window_replay import ACTUAL_RUNNER_SHA, PRIOR_AUDIT_SHA, archive, prior_card, projection, rows
from stage4_fixture import GPU, require, sha, validate_gpu, validate_study, write


RELEASE_WINDOW_RUNNER_SHA = "0f67a3b5e972bb94d08e769b1113a93d92e996e23e11b6f904a69abffb35b18d"
BEFORE, AFTER = 10, 30


def dense_cases(cases):
    output = []
    for meta, *source in cases:
        row = source[0]
        p = meta["window"]["pending_step"]
        start, end = p - BEFORE, p + AFTER
        require(row["fork"]["step"] <= start and end < row["terminal"]["step"]
                and start % CHUNK == end % CHUNK == 0,
                "Dense window must lie on original pre-terminal observer boundaries")
        output.append(({**meta, "window": {"pending_step": p, "start_step": start, "end_step": end, "stride": 1}},
                       *source))
    return output


def aggregate(original, pieces):
    """Only call-level metrics vary; reconstruct a complete original native row."""
    endpoint = pieces[-1]
    collapsed = {**endpoint, "start_step": original["start_step"],
                 "actions": [a for p in pieces for a in p["actions"]],
                 "executed": sum(p["executed"] for p in pieces),
                 "transitions": [t for p in pieces for t in p["transitions"]],
                 "completed_regrasp_steps": sum(p["completed_regrasp_steps"] for p in pieces)}
    differences = sorted(k for k in set(collapsed) | set(original)
                         if k not in collapsed or k not in original or collapsed[k] != original[k])
    require(not differences, f"Reassembled original anchor differs at {original['step']}: {differences}")
    return {"start_step": original["start_step"], "step": original["step"],
            "source_calls": 1, "replay_calls": len(pieces),
            "source_executed": original["executed"], "replay_executed_per_call": [p["executed"] for p in pieces],
            "done": endpoint["done"], "state_sha256": endpoint["state_sha256"],
            "observation_sha256": endpoint["observation_sha256"]}


def sealed_event_history(observed, old_events, anchor_grasp, end):
    """Compare against the old sampled history, retaining dense truth separately."""
    at = {r["step"]: r for r in anchor_grasp}
    result = []
    for event in observed:
        step, obj = event["step"], event["object"]
        held = at[step]["held_by_target"][obj]
        result.append(projection({**event, "held_by_target": held, "released_in_audit": not held,
                                  "previous_sampled_held_step": max((i for i, v in at.items()
                                                                    if i < step and v["held_by_target"][obj]),
                                                                   default=None)}))
    require(result == [projection(e) for e in old_events if e["step"] <= end],
            "Frozen 10-control observer event stream changed")
    return result


def replay(api, worker_type, observer_type, bddl, meta, row, prefix, source_file, prior_grasp, prior_events, dest):
    observer = observer_type()
    start = row["fork"]["step"]
    first, end = meta["window"]["start_step"], meta["window"]["end_step"]
    frames, dense_rows, anchors, checks = [], [], [], []
    observed, combined = [], []
    prefix_i = 0
    originals = list(trace(source_file))
    with worker_type(bddl, dest, row["layout_seed"]) as worker:
        require(api.fingerprint(worker.current) == row["initial"] and not worker.terminal(), "Initial state mismatch")
        initial = worker.current
        (dest / "snapshots").mkdir()
        for old in originals:
            s, stop = old["start_step"], old["step"]
            if stop > end:
                break
            in_prefix = s < start
            target, text = (CHEESE, PREFIX_TASK) if in_prefix else (SOUP, PROMPT)
            require(worker.current["step"] == s and old["target"] == target and old["instruction"] == text
                    and bool(old["actions"]) and (in_prefix or len(old["actions"]) == SUBSTRIDE),
                    f"Unexpected recorded control block at {s}")
            # A source block ending exactly at first remains intact; its endpoint is still saved.
            split = s >= first
            require((s < first and stop <= first) or (s >= first and not in_prefix),
                    "Boundary between sparse and dense may not cut a source interval")
            pieces = []
            for command in ([[a] for a in old["actions"]] if split else [old["actions"]]):
                previous = worker.current["step"]
                require(not worker.terminal(), f"Environment stopped before source control {previous + 1}")
                actual = worker.step(command, target, text)
                state = worker.current
                step = state["step"]
                require(actual == command and step == previous + len(command) and state["executed"] == len(command),
                        f"Realized controls differ at {step}")
                native = {"start_step": previous, "target": target, "instruction": text, "actions": actual, **state}
                pieces.append(native)
                dense_rows.append(native)
                if not in_prefix:
                    require(bool(frames), "Missing frozen fork before suffix")
                    combined.extend(actual)
                    frame = api.Frame.read(worker.npz)
                    frames.append(archive(worker, api, frame, f"snapshots/{step}.npz", dest))
                    if (step - start) % CHUNK == 0:
                        require(len(combined) == CHUNK, f"Incorrect grouping for the original observer at {step}")
                        observed.extend(event_row(e, state, target, "placement_next", worker)
                                        for e in observer.observe(frame, combined, step))
                        combined.clear()
            checks.append(aggregate(old, pieces))
            anchors.append(old)
            if in_prefix:
                chunk = prefix["chunks"][prefix_i]
                prefix_i += 1
                require(api.fingerprint(worker.current) == chunk["end"] and old["actions"] == chunk["actions"],
                        f"Common prefix diverged at {stop}")
                if stop == start:
                    require(prefix_i == len(prefix["chunks"]) and api.fingerprint(worker.current) == row["fork"]
                            and worker.current["audit"]["goals"] == {k: k == CHEESE for k in GOALS},
                            "No exact cheese-only fork")
                    frame = api.Frame.read(worker.npz)
                    observer.clear_view = frame
                    frames.append(archive(worker, api, frame, "fork.npz", dest))
        terminal = api.fingerprint(worker.current)
        expected_frames = list(range(start, first, SUBSTRIDE)) + list(range(first, end + 1))
        require(terminal["step"] == end and not worker.terminal() and not combined
                and [f["step"] for f in frames] == expected_frames,
                "Replay missed the expected pre-window or individual-control observations")
    dense_path = dest / "actions.jsonl"
    require(list(trace(dense_path)) == dense_rows, "Saved dense controls differ from realized controls")
    anchors_path = dest / "source-anchors.jsonl"
    with anchors_path.open("x", encoding="utf-8") as out:
        for anchor in anchors:
            out.write(json.dumps(anchor, allow_nan=False) + "\n")
    require(list(trace(anchors_path)) == [r for r in originals if r["step"] <= end],
            "Source anchors are not the exact original slice")
    grasp_path = dest / "grasp-audit.jsonl"
    # Validate all captures against exact worker records; extra truth does not enter the observer.
    join_grasp_audit(grasp_path, dense_rows, initial, [])
    grasp = rows(grasp_path)
    at = {r["step"]: r for r in grasp}
    old_grasp = [r for r in rows(prior_grasp) if r["step"] <= end]
    old_steps = [0, *[r["step"] for r in anchors]]
    require([r["step"] for r in old_grasp] == old_steps and [at[s] for s in old_steps] == old_grasp,
            "Original all-object physical anchors changed after splitting")
    for frame in frames:
        frame["grasp_audit"] = at[frame["step"]]
    return {"source": row, "fork": row["fork"], "terminal": terminal, "fork_npz_sha256": frames[0]["sha256"],
            "frames": frames, "actions": dense_path.name, "actions_sha256": sha(dense_path),
            "source_anchors": anchors_path.name, "source_anchors_sha256": sha(anchors_path),
            "source_anchor_steps": old_steps, "source_anchor_checks": checks,
            "grasp_audit": grasp_path.name, "grasp_audit_sha256": sha(grasp_path),
            "observer_events": sealed_event_history(observed, prior_events, old_grasp, end),
            "observer_stride": CHUNK, "pre_window_stride": SUBSTRIDE,
            "replayed_action_rows": len(dense_rows), "replayed_action_steps": end,
            "source_action_rows": len(anchors), "physically_verified_capture_rows": len(grasp),
            "dense_window_frames": end - first + 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", required=True, choices=GPU, type=int)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--prior", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    source, prior, out = a.source.resolve(), a.prior.resolve(), a.output.resolve()
    require(out.name != "stage4-pilot" and all(out != origin and not out.is_relative_to(origin)
                                              and not origin.is_relative_to(out) for origin in (source, prior)),
            "Archive into an independent output directory")
    validate_gpu(a.card)
    bddl, old = validate_study(a.study.resolve())
    checked(HERE / "release_window_replay.py", RELEASE_WINDOW_RUNNER_SHA)
    source_manifest, selected = source_card(source, a.card, bddl, old)
    audit_name, prior_manifest, prior_cases = prior_card(prior, a.card, selected)
    fixed = dense_cases(prior_cases)
    observer = load_exact("frozen_release_dense_observer", PLACEMENT / "observer.py", OBSERVER_SHA)
    physics = load_exact("release_dense_physics_worker", HERE / "actual_physics_worker.py", PHYSICS_WORKER_SHA)
    checked(HERE / "actual_physics_simulator.py", PHYSICS_SIMULATOR_SHA)
    out.mkdir(parents=True, exist_ok=True)
    plan_file, marker = (out / f"release-dense{suffix}-{a.card}.json" for suffix in ("-plan", ""))
    require(not plan_file.exists() and not marker.exists()
            and all(not (out / str(meta["layout_seed"])).exists() for meta, *_ in fixed),
            "A sealed single-control replay already exists")
    plan = {"schema": "v2-stage4-release-dense-plan/1", "card": a.card, "scope": "seen_development_only",
            "source_audit_sha256": SOURCE_AUDIT_SHA, "prior_audit": audit_name,
            "prior_audit_sha256": PRIOR_AUDIT_SHA, "bddl_sha256": sha(bddl), "source_sha256": sha(Path(__file__)),
            "actual_runner_sha256": ACTUAL_RUNNER_SHA, "release_window_runner_sha256": RELEASE_WINDOW_RUNNER_SHA,
            "observer_sha256": OBSERVER_SHA, "physics_worker_sha256": PHYSICS_WORKER_SHA,
            "physics_simulator_sha256": PHYSICS_SIMULATOR_SHA, "source_manifest": source_manifest,
            "prior_manifest": prior_manifest, "cases": [m for m, *_ in fixed]}
    write(plan_file, plan)
    manifest = {"schema": "v2-stage4-release-dense-manifest/1", "card": a.card, "complete": False,
                "plan_sha256": sha(plan_file), "prior_audit": audit_name, "prior_audit_sha256": PRIOR_AUDIT_SHA,
                "source_audit_sha256": SOURCE_AUDIT_SHA, "records": []}
    write(marker, manifest)
    import run as api

    worker = physics.worker_class(api)
    for meta, row, prefix, source_file, prior_grasp, events in fixed:
        layout = meta["layout_seed"]
        dest = out / str(layout)
        report = {**{k: v for k, v in plan.items() if k != "cases"}, **meta,
                  "schema": "v2-stage4-release-dense/1", "record_dir": dest.name, "plan_sha256": manifest["plan_sha256"],
                  **replay(api, worker, observer.PlacementObserver, bddl, meta, row, prefix, source_file,
                           prior_grasp, events, dest)}
        target = dest / "report.json"
        write(target, report)
        manifest["records"].append({"layout_seed": layout, "report": target.relative_to(out).as_posix(),
                                    "report_sha256": sha(target)})
        write(marker, manifest)
        print("RELEASE_DENSE", layout, report["window"], report["dense_window_frames"], "single-step frames", flush=True)
    manifest["complete"] = True
    write(marker, manifest)


if __name__ == "__main__":
    main()
