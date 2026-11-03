from __future__ import annotations

import unittest

from tableclear.interfaces import FinalVerdict, Observation, SkillExecution


class InterfaceModelTests(unittest.TestCase):
    def test_observation_and_execution_are_plain_data(self) -> None:
        observation = Observation(
            images={"head": "head.png", "wrist": "wrist.png"},
            robot_state=(0.0, 1.0),
            sim_time_s=1.5,
        )
        execution = SkillExecution(
            status="completed",
            observation=observation,
            metadata={"actions_submitted": 10},
        )
        verdict = FinalVerdict(success=True, score=1.0, reason="task_success")

        self.assertEqual(execution.observation.images["head"], "head.png")
        self.assertTrue(verdict.success)


if __name__ == "__main__":
    unittest.main()

