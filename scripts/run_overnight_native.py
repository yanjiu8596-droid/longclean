"""Run a bounded visual smoke and paired native OSC diagnostics on one GPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SCENES = (1, 2, 8)
HORIZONS = (5, 10, 25, 50)
METHODS = ("direct_vla", "fixed_checklist_vla", "closed_loop_vla")
TARGETS = {
    "alphabet soup can": "alphabet_soup_1", "tomato sauce": "tomato_sauce_1",
    "cream cheese box": "cream_cheese_1", "butter": "butter_1",
}
EPISODE_SCRIPT = Path(__file__).with_name("run_native_osc.py")


def visual_audit(output: Path) -> dict:
    evidence = output / "visual.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()] if evidence.exists() else []
    mismatches = 0
    positives = 0
    false_positives = 0
    for row in rows:
        actual = next((g["satisfied"] for g in row["audit_goals"] if g["predicate"][1] == TARGETS[row["target"]]), None)
        if actual is None or (row["status"] == "in_basket") != actual:
            mismatches += 1
        positives += actual is True and row["status"] == "in_basket"
        false_positives += actual is False and row["status"] == "in_basket"
    return {
        "inventory": sum(row["stage"] == "inventory" for row in rows),
        "progress": sum(row["stage"] == "progress" for row in rows),
        "unparsed": sum(not row["parsed"] for row in rows),
        "abstentions": sum(row["status"] == "unknown" for row in rows),
        "basket_mismatches": mismatches,
        "confirmed_positives": positives,
        "false_positives": false_positives,
    }


def run(args) -> None:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + args.hours * 3600
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EPISODE_SCRIPT.parent, text=True).strip()
    summary = {
        "schema": "tableclear/overnight-diagnostic/1", "pid": os.getpid(), "project_commit": commit,
        "started_at": started.isoformat(), "deadline_at": (started + timedelta(hours=args.hours)).isoformat(),
        "status": "running", "phase": "visual_smoke", "episodes": 0, "errors": 0,
        "completed_diagnostic_blocks": 0, "groups": {}, "formal_matrix_started": False,
    }
    groups = defaultdict(lambda: {"episodes": 0, "successes": 0, "errors": 0, "steps": 0})
    active = None

    def save():
        summary["groups"] = dict(groups)
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = root / "summary.tmp"
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        tmp.replace(root / "summary.json")

    def stop_episode(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()

    def on_stop(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_stop)

    def episode(scene, seed, init, horizon, method):
        nonlocal active
        remaining = deadline - time.monotonic()
        if remaining < 60:
            return None
        tag = f"{summary['phase']}-{scene:02d}-seed{seed}-init{init:02d}-h{horizon}-{method}"
        output = root / tag
        summary["active_episode"] = tag
        save()
        command = [
            sys.executable, "-u", str(EPISODE_SCRIPT), "--workspace", str(args.workspace),
            "--roboenv", str(args.roboenv), "--sim-python", str(args.sim_python),
            "--output", str(output), "--scene", str(scene), "--seed", str(seed),
            "--init-index", str(init), "--chunk-steps", str(horizon), "--method", method,
        ]
        launch = time.monotonic()
        with (root / f"{tag}.log").open("x", encoding="utf-8") as log:
            active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = active.wait(timeout=min(remaining, 360))
            except subprocess.TimeoutExpired:
                stop_episode(active)
                code = 124
        active = None
        data = output / "episode.json"
        record = json.loads(data.read_text()) if data.exists() else {}
        ok = code == 0 and isinstance(record.get("success"), bool)
        row = {
            "run": tag, "phase": summary["phase"], "scene": scene, "seed": seed, "init": init,
            "horizon": horizon, "method": method, "exit_code": code, "completed": ok,
            "success": record.get("success") if ok else None, "steps": record.get("steps", 0),
            "reason": record.get("terminal_reason") if ok else ("deadline_or_episode_limit" if code == 124 else "runtime_error"),
            "elapsed_s": round(time.monotonic() - launch, 2),
            "init_state_sha256": record.get("init_state_sha256"),
        }
        if method != "direct_vla":
            row["visual"] = visual_audit(output) if ok else {}
        with (root / "results.jsonl").open("a", encoding="utf-8") as results:
            results.write(json.dumps(row) + "\n")
        group = groups[f"{summary['phase']}/{scene:02d}/{method}/h{horizon}"]
        group["episodes"] += int(ok)
        group["successes"] += int(ok and row["success"])
        group["errors"] += int(not ok)
        group["steps"] += row["steps"]
        summary["episodes"] += int(ok)
        summary["errors"] += int(not ok)
        summary["active_episode"] = None
        save()
        print(json.dumps(row), flush=True)
        return row

    save()
    try:
        smoke = episode(1, 0, 0, 10, "closed_loop_vla")
        summary["visual_smoke"] = smoke
        audit = smoke["visual"] if smoke else {}
        visual_gate = bool(
            smoke and smoke["completed"] and audit["inventory"] == 2 and audit["progress"] > 0
            and audit["unparsed"] == 0 and audit["abstentions"] == 0 and audit["basket_mismatches"] == 0
        )
        summary["visual_smoke_gate"] = visual_gate
        if visual_gate:
            summary["phase"] = "paired_method_pilot"
            for init in (15, 16, 17):
                for scene in SCENES:
                    for method in METHODS:
                        episode(scene, 0, init, 10, method)

        summary["phase"] = "direct_horizon_diagnostics"
        errors = 0
        for seed in range(3):
            for init in range(15, 50):
                block = []
                # Rotate the condition order so the wall limit does not always favor one horizon.
                shift = (init + seed) % len(HORIZONS)
                for horizon in (*HORIZONS[shift:], *HORIZONS[:shift]):
                    for scene in SCENES:
                        row = episode(scene, seed, init, horizon, "direct_vla")
                        if row is None:
                            summary["status"] = "deadline_reached"
                            return
                        block.append(row)
                        errors = 0 if row["completed"] else errors + 1
                        if errors == 3:
                            summary["status"] = "stopped_after_repeated_runtime_errors"
                            return
                summary["completed_diagnostic_blocks"] += int(all(row["completed"] for row in block))
                save()
        summary["status"] = "plan_completed"
    except KeyboardInterrupt:
        summary["status"] = "stopped"
    except Exception:
        summary["status"] = "orchestrator_error"
        raise
    finally:
        if active is not None:
            stop_episode(active)
        summary["active_episode"] = None
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--roboenv", required=True, type=Path)
    parser.add_argument("--sim-python", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=5)
    args = parser.parse_args()
    if not 0.05 <= args.hours <= 5:
        parser.error("--hours must be between 0.05 and 5")
    run(args)
