from __future__ import annotations

import csv
import unittest
from pathlib import Path

from tableclear.aggregate import (
    aggregate_episode_records,
    write_summary_csv,
)


class AggregateTests(unittest.TestCase):
    def test_groups_episode_end_by_method(self) -> None:
        records = [
            {"event_type": "episode_start", "method": "ignored"},
            {
                "event_type": "episode_end",
                "method": "closed_loop_vla",
                "success": True,
                "leftover_count": 0,
                "repeated_actions": 1,
                "steps": 20,
                "vla_calls": 2,
                "runtime_s": 10.0,
            },
            {
                "event_type": "episode_end",
                "method": "closed_loop_vla",
                "success": False,
                "leftover_count": 2,
                "repeated_actions": 3,
                "steps": 40,
                "vla_calls": 4,
                "runtime_s": 30.0,
            },
            {
                "event_type": "episode_end",
                "method": "direct_vla",
                "success": False,
                "leftover_count": 2,
                "repeated_actions": 0,
                "steps": 30,
                "vla_calls": 1,
                "runtime_s": 15.0,
            },
        ]

        summaries = aggregate_episode_records(records)

        self.assertEqual([item.method for item in summaries], ["closed_loop_vla", "direct_vla"])
        closed_loop = summaries[0]
        self.assertEqual(closed_loop.episodes, 2)
        self.assertEqual(closed_loop.successes, 1)
        self.assertEqual(closed_loop.success_rate, 0.5)
        self.assertEqual(closed_loop.mean_leftover, 1.0)

    def test_writes_csv(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifacts = root / ".test-artifacts"
        artifacts.mkdir(exist_ok=True)
        output = artifacts / "summary.csv"
        output.unlink(missing_ok=True)
        summaries = aggregate_episode_records(
            [
                {
                    "event_type": "episode_end",
                    "method": "direct_vla",
                    "success": True,
                    "leftover_count": 0,
                }
            ]
        )

        try:
            written = write_summary_csv(summaries, output)
            with written.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        finally:
            output.unlink(missing_ok=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "direct_vla")
        self.assertEqual(rows[0]["success_rate"], "1.0")


if __name__ == "__main__":
    unittest.main()

