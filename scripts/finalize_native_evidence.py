"""Finalize the best examples once the native diagnostic job releases its lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path

from archive_tableclear_evidence import archive, best_examples
from replay_native_video import replay
from summarize_horizon_diagnostics import markdown, summarize
from zipfile import ZipFile


def finalize(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "waiting_for_diagnostic_lock", "run": str(args.run), "videos": [], "replay_errors": []}

    def save():
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    save()
    try:
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            completed = [r["run"] for r in rows if r["phase"] == "direct_horizon_diagnostics" and r["completed"]]
            record["source_status"] = summary["status"]
            if summary["status"] != "plan_completed" or summary["active_episodes"]:
                raise RuntimeError("diagnostic scheduler has not reached plan_completed")
            if len(set(completed)) != len(completed) or len(completed) != summary["diagnostics_planned"]:
                raise RuntimeError("diagnostic result index does not match the completed plan")

            record["diagnostics_completed"] = len(completed)
            record["status"] = "replaying"
            save()
            replays = output / "replays"
            replays.mkdir()
            for row in best_examples(rows):
                name = row.get("artifact", row["run"])
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError(f"invalid artifact directory: {name}")
                try:
                    video = replay(args.run / name, replays / name, args.sim_python, args.roboenv)
                except Exception as exc:
                    record["replay_errors"].append({"artifact": name, "error": str(exc)})
                else:
                    record["videos"].append(video)
                save()

            target = output / "final-best.zip"
            archive(args.run, target, final=True, replays=replays)
            with ZipFile(target) as packed:
                manifest = json.loads(packed.read("archive-manifest.json"))
            report = summarize(rows, manifest)
            (output / "horizon-metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (output / "动作块配对结果.md").write_text(markdown(report), encoding="utf-8")
            record.update(status="completed", archive=str(target), archive_sha256=sha256(target.read_bytes()).hexdigest())
    except Exception as exc:
        record.update(status="error", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--sim-python", required=True, type=Path)
    parser.add_argument("--roboenv", required=True, type=Path)
    finalize(parser.parse_args())
