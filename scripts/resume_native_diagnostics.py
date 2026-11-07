"""Resume the direct horizon sweep with disjoint concurrent episodes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_overnight_native import EPISODE_SCRIPT, HORIZONS, SCENES


PHASE = "direct_horizon_diagnostics"
METHOD = "direct_vla"


def planned_blocks():
    for seed in range(3):
        for init in range(15, 50):
            shift = (init + seed) % len(HORIZONS)
            yield [
                (f"{PHASE}-{scene:02d}-seed{seed}-init{init:02d}-h{horizon}-{METHOD}", scene, seed, init, horizon)
                for horizon in (*HORIZONS[shift:], *HORIZONS[:shift]) for scene in SCENES
            ]


def result_for(job, artifact, code, elapsed):
    tag, scene, seed, init, horizon = job
    path = artifact / "episode.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        record = {}
    completed = (
        code == 0 and isinstance(record.get("success"), bool)
        and isinstance(record.get("finished_at"), str)
        and record.get("terminal_reason") in {"simulator_success", "simulator_done", "step_budget"}
        and record.get("task_id") == f"libero/libero-10-{scene:02d}"
        and record.get("seed") == seed and record.get("init_state_index") == init
        and record.get("chunk_horizon") == horizon and record.get("method") == METHOD
    )
    return {
        "run": tag, "phase": PHASE, "scene": scene, "seed": seed, "init": init,
        "horizon": horizon, "method": METHOD, "exit_code": code, "completed": completed,
        "success": record.get("success") if completed else None, "steps": record.get("steps", 0),
        "reason": record.get("terminal_reason") if completed else ("episode_limit" if code == 124 else "runtime_error"),
        "elapsed_s": round(elapsed, 2), "init_state_sha256": record.get("init_state_sha256"),
        "project_commit": record.get("project_commit"), "artifact": artifact.name,
    }


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()


def run(args):
    root = args.output.resolve()
    summary_path = root / "summary.json"
    results_path = root / "results.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {"stopped", "deadline_reached"} or summary["phase"] != PHASE:
        raise ValueError("the previous direct sweep must first stop cleanly")
    deadline_at = summary["deadline_at"]
    if args.until_complete and deadline_at is not None:
        summary["original_deadline_at"] = deadline_at
        summary["deadline_at"] = deadline_at = None
    deadline = datetime.fromisoformat(deadline_at).timestamp() if deadline_at else float("inf")
    if deadline - time.time() < 60:
        raise ValueError("too little time remains before the original deadline")
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    completed = {row["run"] for row in rows if row["completed"]}
    blocks = list(planned_blocks())
    diagnostic_tags = {job[0] for block in blocks for job in block}
    active = {}
    concurrency = args.workers
    summary.update(status="running", pid=os.getpid(), resumed_at=datetime.now(timezone.utc).isoformat())
    summary["scheduler_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=EPISODE_SCRIPT.parent, text=True,
    ).strip()

    def save():
        groups = defaultdict(lambda: {"episodes": 0, "successes": 0, "errors": 0, "steps": 0})
        for row in rows:
            group = groups[f"{row['phase']}/{row['scene']:02d}/{row['method']}/h{row['horizon']}"]
            group["episodes"] += int(row["completed"])
            group["successes"] += int(row["success"] is True)
            group["errors"] += int(not row["completed"])
            group["steps"] += row["steps"]
        summary.update(
            groups=dict(groups), episodes=len(completed), errors=sum(not row["completed"] for row in rows),
            completed_diagnostic_blocks=sum(all(job[0] in completed for job in block) for block in blocks),
            diagnostics_completed=len(completed & diagnostic_tags), diagnostics_planned=len(diagnostic_tags),
            active_episode=next(iter(active), None), active_episodes=list(active), workers=concurrency,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        tmp.replace(summary_path)

    def record(row):
        if row["completed"]:
            if row["run"] in completed:
                raise RuntimeError(f"duplicate result for {row['run']}")
            completed.add(row["run"])
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        rows.append(row)
        print(json.dumps(row), flush=True)

    def launch(job):
        tag, scene, seed, init, horizon = job
        suffix = 0
        while True:
            artifact = root / (tag if suffix == 0 else f"{tag}-retry{suffix}")
            log = artifact.with_suffix(".log")
            if not artifact.exists() and not log.exists():
                break
            suffix += 1
        command = [
            sys.executable, "-u", str(EPISODE_SCRIPT), "--workspace", str(args.workspace),
            "--roboenv", str(args.roboenv), "--sim-python", str(args.sim_python),
            "--output", str(artifact), "--scene", str(scene), "--seed", str(seed),
            "--init-index", str(init), "--chunk-steps", str(horizon), "--method", METHOD,
        ]
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active[tag] = (process, job, artifact, time.monotonic())
        save()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    # A worker can finish writing its evidence just before the old scheduler stops.
    existing_artifacts = {row.get("artifact", row["run"]) for row in rows}
    try:
        for block in blocks:
            for job in block:
                tag = job[0]
                if tag in completed:
                    continue
                for artifact in (root / tag, *sorted(root.glob(f"{tag}-retry[0-9]*"))):
                    if artifact.name in existing_artifacts or not (artifact / "episode.json").is_file():
                        continue
                    row = result_for(job, artifact, 0, 0)
                    if row["completed"]:
                        row["recovered_after_stop"] = True
                        record(row)
                        break
        save()
        errors = 0
        for block in blocks:
            pending = [job for job in block if job[0] not in completed]
            while pending or active:
                remaining = deadline - time.time()
                while pending and len(active) < concurrency and remaining >= 60 and errors < 3:
                    launch(pending.pop(0))
                for tag, (process, job, artifact, start) in list(active.items()):
                    code = process.poll()
                    elapsed = time.monotonic() - start
                    if code is None and (elapsed >= 360 or remaining <= 5):
                        stop(process)
                        code = 124
                    if code is not None:
                        del active[tag]
                        row = result_for(job, artifact, code, elapsed)
                        record(row)
                        errors = 0 if row["completed"] else errors + 1
                        if not row["completed"]:
                            concurrency = 1
                            if deadline_at is None:
                                pending.insert(0, job)
                        save()
                if not active and (remaining < 60 or errors >= 3):
                    summary["status"] = "deadline_reached" if remaining < 60 else "stopped_after_repeated_runtime_errors"
                    return
                if active:
                    time.sleep(0.2)
        summary["status"] = "plan_completed"
    except KeyboardInterrupt:
        summary["status"] = "stopped"
    except Exception:
        summary["status"] = "orchestrator_error"
        raise
    finally:
        for process, *_ in active.values():
            stop(process)
        active.clear()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--roboenv", required=True, type=Path)
    parser.add_argument("--sim-python", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--until-complete", action="store_true", help="Finish all planned diagnostic items without a batch deadline.")
    run(parser.parse_args())
