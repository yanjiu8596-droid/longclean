"""Archive a TableClear run without modifying its source records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EPISODE_FILES = (
    "episode.json", "actions.jsonl", "visual.jsonl", "simulator.log",
    "initial-camera1.png", "initial-camera2.png", "final-camera1.png", "final-camera2.png",
)
HISTORICAL = {
    "gpu-device-20261107T014118Z": (
        "device-smoke.json", "stop-owner.json", "project-commit.txt",
    ),
    "kinex-native-20261107T015621Z": (
        "device-smoke.json", "parity.json", "kinex-osc/initial.json", "kinex-osc/actions.jsonl",
    ),
    "native-osc-01-20261107T014440Z": ("episode.json", "actions.jsonl"),
}


def archive(run: Path, output: Path, *, final: bool, replays: Path | None) -> dict:
    summary_bytes = (run / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    if final and summary["status"] == "running":
        raise ValueError("the final archive requires a stopped or completed run")
    result_lines = (run / "results.jsonl").read_bytes().splitlines(keepends=True)
    result_bytes = b"".join(line for line in result_lines if line.endswith(b"\n"))
    rows = [json.loads(line) for line in result_bytes.splitlines()]
    completed = [row for row in rows if row["completed"]]
    if len({row["run"] for row in completed}) != len(completed):
        raise ValueError("duplicate completed runs in the result index")

    examples = {}
    for row in completed:
        if row["phase"] == "direct_horizon_diagnostics" and row["success"] is True:
            current = examples.get(row["scene"])
            if current is None or (row["steps"], row["run"]) < (current["steps"], current["run"]):
                examples[row["scene"]] = row
    selected = [examples[scene] for scene in sorted(examples)]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run": run.name,
        "status": summary["status"], "final_archive": final, "indexed_rows": len(rows),
        "completed_rows": len(completed), "scheduler_commit": summary.get("scheduler_commit"),
        "source_project_commit": summary.get("project_commit"),
        "selection": "one simulator-confirmed success per scene with the fewest action steps; ties by run ID",
        "examples": selected, "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=4) as zip_file:
        def add(name, data):
            manifest["files"][name] = sha256(data).hexdigest()
            zip_file.writestr(name, data)

        def add_file(root, relative, prefix):
            path = root / relative
            if path.is_file():
                add(f"{prefix}/{relative}", path.read_bytes())

        add("index/summary.json", summary_bytes)
        add("index/results.jsonl", result_bytes)
        episode_metrics = []
        for row in completed:
            name = row.get("artifact", row["run"])
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid episode directory: {name}")
            path = run / name / "episode.json"
            if path.is_file():
                episode = json.loads(path.read_bytes())
                keys = ("project_commit", "bddl_sha256", "vla_calls", "visual_calls", "started_at", "finished_at", "control_hz")
                episode_metrics.append(json.dumps({"run": row["run"], **{key: episode.get(key) for key in keys}}) + "\n")
        add("index/episode-metrics.jsonl", "".join(episode_metrics).encode())
        for name in ("concurrency-cutover.json", "completion-cutover.json", "scaling-cutover.json"):
            add_file(run, name, "index")
        for name, files in HISTORICAL.items():
            for file in files:
                add_file(run.parent / name, file, f"milestones/{name}")
        artifacts = {row.get("artifact", row["run"]) for row in selected}
        for name in sorted(artifacts):
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid episode directory: {name}")
            for file in EPISODE_FILES:
                add_file(run / name, file, f"episodes/{name}")
        for row in rows:
            if row["phase"] == "visual_smoke":
                add_file(run / row["run"], "visual.jsonl", "index")
        if replays is not None:
            for name in sorted(artifacts):
                for file in ("replay.mp4", "replay.json"):
                    add_file(replays / name, file, f"videos/{name}")
        zip_file.writestr("archive-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in manifest.items() if key not in {"files", "examples"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final", action="store_true", help="Require a terminal batch before archiving.")
    parser.add_argument("--replays", type=Path)
    args = parser.parse_args()
    print(json.dumps(archive(args.run, args.output, final=args.final, replays=args.replays)))
