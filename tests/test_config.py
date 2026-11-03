from __future__ import annotations

import unittest
from pathlib import Path

from tableclear.config import expand_matrix, load_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v1_eval.toml"


class EvaluationConfigTests(unittest.TestCase):
    def test_v1_matrix_has_expected_size(self) -> None:
        config = load_evaluation_config(CONFIG)
        jobs = expand_matrix(config)

        self.assertEqual(config.episode_count, 135)
        self.assertEqual(len(jobs), 135)
        self.assertEqual(len({job.run_id for job in jobs}), 135)
        self.assertEqual(
            [scene.source_task for scene in config.scenes],
            [
                "libero/libero-10-01",
                "libero/libero-10-02",
                "libero/libero-10-08",
            ],
        )

    def test_paired_methods_share_initial_state(self) -> None:
        config = load_evaluation_config(CONFIG)
        jobs = expand_matrix(config)
        grouped: dict[tuple[str, int, int], set[int]] = {}
        for job in jobs:
            key = (job.scene_id, job.seed, job.episode)
            grouped.setdefault(key, set()).add(job.init_state_index)

        self.assertTrue(all(len(indices) == 1 for indices in grouped.values()))
        self.assertEqual(
            [job.init_state_index for job in jobs[:9]],
            list(range(9)),
        )


if __name__ == "__main__":
    unittest.main()

