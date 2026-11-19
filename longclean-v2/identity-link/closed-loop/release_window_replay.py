"""Archive five-step RGB-D observations around four sealed release windows."""
import argparse
import json
from pathlib import Path
from shutil import copyfile

from actual_progress_replay import (
    AUDIT_SHA as SOURCE_AUDIT_SHA, HERE, OBSERVER_SHA, PHYSICS_SIMULATOR_SHA, PHYSICS_WORKER_SHA,
    PLACEMENT, PROMPT, SUBSTRIDE, checked, event_row, join_grasp_audit, load_exact, source_card,
)
from common import CHEESE, CHUNK, GOALS, PREFIX_TASK, SAUCE, SOUP
from legacy_parity import trace
from stage4_fixture import GPU, read, require, sha, validate_gpu, validate_study, write
from target_follow_audit import observation


PRIOR_AUDIT_SHA = "34970f0017ca2909a2dcbd3e3138b83f0b579f250c8072b7076ecf72e7b2bb7a"
ACTUAL_RUNNER_SHA = "f0c1000074559a2eb48859cd3ff44c22d123b0650d7dd962a9fe44ba05909bb2"
PENDING = {931000: 240, 931005: 230, 931200: 220, 931201: 240}
BEFORE, AFTER = 20, 40


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(map(json.loads, stream))


def prior_card(prior, card, selected):
    allowed = (prior / "audit.json", prior / "actual-progress-audit.json")
    pins = [p for p in allowed if p.is_file() and sha(p) == PRIOR_AUDIT_SHA]
    require(bool(pins), "Could not find the exact sealed actual-place audit in --prior")
    prior_audit_file = pins[0]
    audit = read(prior_audit_file)
    require(audit["schema"] == "v2-stage4-actual-progress-audit/1"
            and audit["gate"]["release_original_pilot"] is False
            and audit["source_audit_sha256"] == SOURCE_AUDIT_SHA
            and audit["runner_sha256"] == ACTUAL_RUNNER_SHA
            and audit["observer_sha256"] == OBSERVER_SHA
            and audit["physics_worker_sha256"] == PHYSICS_WORKER_SHA
            and audit["physics_simulator_sha256"] == PHYSICS_SIMULATOR_SHA,
            "Unexpected prior release-chain sources")
    checked(HERE / "actual_progress_replay.py", ACTUAL_RUNNER_SHA)
    stamp, = [r for r in audit["manifests"] if r["card"] == card]
    manifest = read(checked(prior / f"actual-progress-{card}.json", stamp["manifest_sha256"]))
    checked(prior / f"actual-progress-plan-{card}.json", stamp["plan_sha256"])
    require(manifest["complete"] and manifest["plan_sha256"] == stamp["plan_sha256"]
            and manifest["source_audit_sha256"] == SOURCE_AUDIT_SHA,
            "Prior per-card manifest does not match sealed audit")
    input_cases = []
    for row, prefix, actions, prefix_sha in selected:
        layout = row["layout_seed"]
        r, = [r for r in manifest["records"] if r["layout_seed"] == layout]
        audited, = [r for r in audit["cases"] if r["layout_seed"] == layout and r["card"] == card]
        require(r["report"] == f"{layout}/report.json" and r["report_sha256"] == audited["report_sha256"]
                and audited["source_actions_sha256"] == row["actions_sha256"]
                and audited["source_result_sha256"] == row["result_sha256"]
                and audited["prefix_sha256"] == prefix_sha, f"Prior sealed layout differs: {layout}")
        report_file = checked(prior / r["report"], r["report_sha256"])
        report = read(report_file)
        require(report["source"] == row and report["prefix_sha256"] == prefix_sha
                and report["actions_sha256"] == audited["replay_actions_sha256"]
                and report["grasp_audit_sha256"] == audited["grasp_audit_sha256"]
                and report["actions"] == "actions.jsonl" and report["grasp_audit"] == "grasp-audit.jsonl",
                f"Prior report does not describe the same actions: {layout}")
        checked(report_file.parent / report["actions"], report["actions_sha256"])
        prior_grasp = checked(report_file.parent / report["grasp_audit"], report["grasp_audit_sha256"])
        events = report["arms"]["candidate"]["events"]
        pending = next(e for e in events if e["event"] == "pending_release")
        p = PENDING[layout]
        require(pending["object"] == SAUCE and pending["step"] == p and p % CHUNK == 0
                and row["fork"]["step"] <= p - BEFORE and p + AFTER < row["terminal"]["step"],
                f"First prior release changed or does not leave the frozen window: {layout}")
        frozen = {"layout_seed": layout, "policy_seed": row["policy_seed"], "prior_report": r["report"],
                  "prior_report_sha256": r["report_sha256"], "prior_grasp_audit_sha256": report["grasp_audit_sha256"],
                  "source_result_sha256": row["result_sha256"], "source_actions_sha256": row["actions_sha256"],
                  "prefix_sha256": prefix_sha,
                  "window": {"pending_step": p, "start_step": p - BEFORE, "end_step": p + AFTER,
                             "stride": SUBSTRIDE}}
        input_cases.append((frozen, row, prefix, actions, prior_grasp, events))
    return prior_audit_file.name, stamp, input_cases


def archive(worker, api, frame, filename, dest):
    snapshot = dest / filename
    copyfile(worker.npz, snapshot)
    finger = api.fingerprint(worker.current)
    observation(snapshot, finger)
    return {"step": finger["step"], "file": filename, "sha256": sha(snapshot),
            "state_sha256": finger["state_sha256"], "observation_sha256": finger["observation_sha256"],
            "grip_gap": frame.gap}


def projection(event):
    return {k: v for k, v in event.items() if k not in {"evidence_file", "evidence_sha256"}}


def replay(api, worker_type, observer_type, bddl, meta, row, prefix, actions, prior_grasp, prior_events, dest):
    observer = observer_type()
    frames, trial, combined, observed = [], [], [], []
    source_rows = list(trace(actions))
    start, cut = row["fork"]["step"], meta["window"]["end_step"]
    prefix_i = 0
    with worker_type(bddl, dest, row["layout_seed"]) as worker:
        require(api.fingerprint(worker.current) == row["initial"] and not worker.terminal(),
                "Native initial state diverged")
        initial = worker.current
        (dest / "snapshots").mkdir()
        for source_row in source_rows:
            s, step = source_row["start_step"], source_row["step"]
            if step > cut:
                break
            in_prefix = s < start
            assigned, prompt = (CHEESE, PREFIX_TASK) if in_prefix else (SOUP, PROMPT)
            require(not worker.terminal() and source_row["target"] == assigned and source_row["instruction"] == prompt,
                    f"Unexpected source label or native terminal at {s}")
            actual = worker.step(source_row["actions"], source_row["target"], source_row["instruction"])
            current = worker.current
            require({"start_step": s, "target": assigned, "instruction": prompt, "actions": actual, **current}
                    == source_row, f"Exact native source replay diverged at step {step}")
            trial.append(source_row)
            if in_prefix:
                chunk = prefix["chunks"][prefix_i]
                prefix_i += 1
                require(api.fingerprint(current) == chunk["end"] and actual == chunk["actions"],
                        f"Frozen cheese actions differ at {step}")
                if step == start:
                    require(prefix_i == len(prefix["chunks"]) and api.fingerprint(current) == row["fork"]
                            and current["audit"]["goals"] == {o: o == CHEESE for o in GOALS},
                            "Fork was not the fixed cheese prefix")
                    frame = api.Frame.read(worker.npz)
                    observer.clear_view = frame
                    frames.append(archive(worker, api, frame, "fork.npz", dest))
                continue
            require(bool(frames) and len(actual) == SUBSTRIDE, f"Missing five-step source anchor: {step}")
            frame = api.Frame.read(worker.npz)
            frames.append(archive(worker, api, frame, f"snapshots/{step}.npz", dest))
            combined.extend(actual)
            if (step - start) % CHUNK == 0:
                require(len(combined) == CHUNK, f"Frozen ten-step observer not on grid: {step}")
                observed.extend(event_row(e, current, assigned, "placement_next", worker)
                                for e in observer.observe(frame, combined, step))
                combined.clear()
        terminal = api.fingerprint(worker.current)
        require(terminal["step"] == cut and not worker.terminal() and not combined
                and [f["step"] for f in frames] == list(range(start, cut + 1, SUBSTRIDE)),
                "Missing required release-window frames or unintended terminal")
    action_path = dest / "actions.jsonl"
    require(list(trace(action_path)) == trial, "Saved replay actions differ from sealed source slice")
    grasp = dest / "grasp-audit.jsonl"
    join_grasp_audit(grasp, trial, initial, observed)
    physical = rows(grasp)
    require(physical == [r for r in rows(prior_grasp) if r["step"] <= cut],
            "New physical sidecar differs from the sealed all-object prefix and suffix captures")
    at_step = {r["step"]: r for r in physical}
    for frame in frames:
        frame["grasp_audit"] = at_step[frame["step"]]
    events = [projection(e) for e in observed]
    require(events == [projection(e) for e in prior_events if e["step"] <= cut],
            "Frozen ten-step placement event stream changed in replay")
    return {"source": row, "fork": row["fork"], "terminal": terminal,
            "fork_npz_sha256": frames[0]["sha256"], "frames": frames,
            "actions": action_path.name, "actions_sha256": sha(action_path),
            "grasp_audit": grasp.name, "grasp_audit_sha256": sha(grasp),
            "observer_events": events, "observer_stride": CHUNK, "replayed_action_rows": len(trial),
            "replayed_action_steps": cut, "physically_verified_capture_rows": len(physical)}


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
    source_manifest, selected = source_card(source, a.card, bddl, old)
    prior_audit_name, prior_manifest, fixed = prior_card(prior, a.card, selected)
    observer = load_exact("frozen_release_window_observer", PLACEMENT / "observer.py", OBSERVER_SHA)
    physics = load_exact("release_window_physics_worker", HERE / "actual_physics_worker.py", PHYSICS_WORKER_SHA)
    checked(HERE / "actual_physics_simulator.py", PHYSICS_SIMULATOR_SHA)
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"release-window-plan-{a.card}.json"
    marker = out / f"release-window-{a.card}.json"
    require(not plan_file.exists() and not marker.exists()
            and all(not (out / str(meta["layout_seed"])).exists() for meta, *_ in fixed),
            "A sealed release-window capture already exists")
    plan = {"schema": "v2-stage4-release-window-plan/1", "card": a.card, "scope": "seen_development_only",
            "source_audit_sha256": SOURCE_AUDIT_SHA, "prior_audit": prior_audit_name,
            "prior_audit_sha256": PRIOR_AUDIT_SHA, "bddl_sha256": sha(bddl), "source_sha256": sha(Path(__file__)),
            "actual_runner_sha256": ACTUAL_RUNNER_SHA, "observer_sha256": OBSERVER_SHA,
            "physics_worker_sha256": PHYSICS_WORKER_SHA, "physics_simulator_sha256": PHYSICS_SIMULATOR_SHA,
            "source_manifest": source_manifest, "prior_manifest": prior_manifest, "cases": [m for m, *_ in fixed]}
    write(plan_file, plan)
    manifest = {"schema": "v2-stage4-release-window-manifest/1", "card": a.card, "complete": False,
                "plan_sha256": sha(plan_file), "prior_audit": prior_audit_name,
                "prior_audit_sha256": PRIOR_AUDIT_SHA, "source_audit_sha256": SOURCE_AUDIT_SHA, "records": []}
    write(marker, manifest)
    import run as api

    worker = physics.worker_class(api)
    for meta, row, prefix, actions, prior_grasp, events in fixed:
        layout = meta["layout_seed"]
        dest = out / str(layout)
        report = {**{k: v for k, v in plan.items() if k != "cases"}, **meta,
                  "schema": "v2-stage4-release-window/1", "record_dir": dest.name,
                  "plan_sha256": manifest["plan_sha256"],
                  **replay(api, worker, observer.PlacementObserver, bddl, meta, row, prefix, actions,
                           prior_grasp, events, dest)}
        report_file = dest / "report.json"
        write(report_file, report)
        manifest["records"].append({"layout_seed": layout, "report": report_file.relative_to(out).as_posix(),
                                    "report_sha256": sha(report_file)})
        write(marker, manifest)
        print("RELEASE_WINDOW", layout, report["window"], len(report["frames"]), "frames", flush=True)
    manifest["complete"] = True
    write(marker, manifest)


if __name__ == "__main__":
    main()
