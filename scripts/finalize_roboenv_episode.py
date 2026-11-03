"""Convert direct-episode output and sealed RoboEnv evidence to episode_end."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_episode_end(
    *,
    episodes: list[dict[str, Any]],
    terminal: dict[str, Any],
    task_id: str,
    method: str,
    seed: int,
    episode: int,
    leftover_count: int,
) -> dict[str, Any]:
    chunks = [
        chunk
        for record in episodes
        for chunk in record.get("chunks", [])
    ]
    successful_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict) and isinstance(chunk.get("subtask"), dict)
    ]
    vla_calls = len(successful_chunks)
    steps = sum(
        int(chunk["subtask"].get("actions_submitted", 0))
        for chunk in successful_chunks
    )
    started = [
        record.get("started_at")
        for record in episodes
        if isinstance(record.get("started_at"), str)
    ]
    finished = [
        record.get("finished_at")
        for record in episodes
        if isinstance(record.get("finished_at"), str)
    ]
    if started and finished:
        runtime_s = (
            datetime.fromisoformat(finished[-1])
            - datetime.fromisoformat(started[0])
        ).total_seconds()
    else:
        runtime_s = 0.0

    status = terminal.get("status")
    verdict = terminal.get("verdict")
    if not isinstance(status, dict) or not isinstance(verdict, dict):
        raise ValueError("terminal evidence lacks status or verdict")
    success = bool(status.get("success"))
    simulator_reason = str(status.get("reason", "unknown"))
    reason = (
        "simulator_success"
        if success
        else (
            "manual_stop_after_chunk_budget"
            if simulator_reason == "stopped"
            else simulator_reason
        )
    )
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "event_type": "episode_end",
        "run_id": status.get("session"),
        "task_id": task_id,
        "method": method,
        "seed": seed,
        "episode": episode,
        "success": success,
        "leftover_count": leftover_count,
        "repeated_actions": 0,
        "terminal_reason": reason,
        "steps": steps,
        "agent_calls": 0,
        "vla_calls": vla_calls,
        "runtime_s": runtime_s,
        "simulator_score": float(verdict.get("raw_score", 0.0)),
        "metadata": {
            "simulator_reason": simulator_reason,
            "budget": status.get("budget"),
            "control_equivalence": verdict.get("detail"),
            "submitted": verdict.get("submitted"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--method", default="direct_vla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--leftover-count", type=int, default=0)
    args = parser.parse_args()

    record = build_episode_end(
        episodes=[load_object(path) for path in args.episode],
        terminal=load_object(args.terminal),
        task_id=args.task_id,
        method=args.method,
        seed=args.seed,
        episode=args.episode_index,
        leftover_count=args.leftover_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
