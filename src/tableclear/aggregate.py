from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class MethodSummary:
    method: str
    episodes: int
    successes: int
    success_rate: float
    mean_leftover: float
    mean_repeated_actions: float
    mean_steps: float
    mean_vla_calls: float
    mean_runtime_s: float


def aggregate_episode_records(
    records: Iterable[dict[str, Any]],
) -> tuple[MethodSummary, ...]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("event_type") != "episode_end":
            continue
        method = record.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("episode_end record is missing method")
        grouped[method].append(record)

    summaries = []
    for method, rows in sorted(grouped.items()):
        episodes = len(rows)
        successes = sum(bool(row.get("success", False)) for row in rows)
        summaries.append(
            MethodSummary(
                method=method,
                episodes=episodes,
                successes=successes,
                success_rate=successes / episodes,
                mean_leftover=_mean(rows, "leftover_count"),
                mean_repeated_actions=_mean(rows, "repeated_actions"),
                mean_steps=_mean(rows, "steps"),
                mean_vla_calls=_mean(rows, "vla_calls"),
                mean_runtime_s=_mean(rows, "runtime_s"),
            )
        )
    return tuple(summaries)


def write_summary_csv(
    summaries: Iterable[MethodSummary],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "episodes",
        "successes",
        "success_rate",
        "mean_leftover",
        "mean_repeated_actions",
        "mean_steps",
        "mean_vla_calls",
        "mean_runtime_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    field: getattr(summary, field)
                    for field in fieldnames
                }
            )
    return path


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(key, 0.0)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        values.append(float(value))
    return mean(values) if values else 0.0

