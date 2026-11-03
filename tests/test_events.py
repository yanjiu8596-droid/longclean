from __future__ import annotations

import unittest
from pathlib import Path

from tableclear.events import JsonlEventWriter, read_jsonl


class EventLogTests(unittest.TestCase):
    def test_round_trip_jsonl(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifact_root = root / ".test-artifacts"
        artifact_root.mkdir(exist_ok=True)
        path = artifact_root / "episode-test.jsonl"
        path.unlink(missing_ok=True)
        try:
            with JsonlEventWriter(path) as writer:
                writer.write("episode_start", {"task_id": "demo", "seed": 0})
                writer.write("cleanup_attempt", {"target_id": "object_1"})

            records = list(read_jsonl(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event_type"], "episode_start")
        self.assertEqual(records[1]["target_id"], "object_1")
        self.assertIn("timestamp_utc", records[0])


if __name__ == "__main__":
    unittest.main()

