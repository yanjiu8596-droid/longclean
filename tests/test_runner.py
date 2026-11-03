from __future__ import annotations

import unittest

from tableclear.controller import CleanupController
from tableclear.interfaces import (
    FinalVerdict,
    Observation,
    SkillExecution,
)
from tableclear.models import (
    CleanupVerdict,
    ObjectStatus,
    Subtask,
    VerdictStatus,
)
from tableclear.runner import CleanupEpisodeRunner, EpisodeContext
from tableclear.state import TaskState


class FakeEnvironment:
    def __init__(self, success: bool = True) -> None:
        self.success = success
        self.reset_called = False
        self.closed = False
        self.instructions: list[str] = []
        self.time_s = 0.0

    def reset(self, *, init_state_index: int) -> Observation:
        self.reset_called = True
        self.time_s = float(init_state_index)
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            images={"head": f"frame-{self.time_s}.png"},
            robot_state=(0.0,),
            sim_time_s=self.time_s,
        )

    def execute(self, subtask: Subtask) -> SkillExecution:
        self.instructions.append(subtask.instruction)
        self.time_s += 1.0
        return SkillExecution(
            status="completed",
            observation=self.observe(),
            metadata={"steps": 10},
        )

    def score(self) -> FinalVerdict:
        return FinalVerdict(
            success=self.success,
            score=1.0 if self.success else 0.0,
            reason="test",
        )

    def close(self) -> None:
        self.closed = True


class SuccessVerifier:
    def verify(
        self,
        before: Observation,
        after: Observation,
        target_id: str,
        instruction: str,
    ) -> CleanupVerdict:
        return CleanupVerdict(
            target_id=target_id,
            status=VerdictStatus.IN_BASKET,
            progress=1.0,
            confidence=1.0,
        )


class FailOnceVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(
        self,
        before: Observation,
        after: Observation,
        target_id: str,
        instruction: str,
    ) -> CleanupVerdict:
        self.calls += 1
        if self.calls == 1:
            return CleanupVerdict(
                target_id=target_id,
                status=VerdictStatus.NO_PROGRESS,
                progress=0.0,
                confidence=1.0,
            )
        return CleanupVerdict(
            target_id=target_id,
            status=VerdictStatus.IN_BASKET,
            progress=1.0,
            confidence=1.0,
        )


class CleanupEpisodeRunnerTests(unittest.TestCase):
    def test_clears_every_object_before_requesting_finish(self) -> None:
        state = TaskState.from_labels("demo", ("mug", "bowl"))
        controller = CleanupController(state, max_attempts=3)
        environment = FakeEnvironment(success=True)
        runner = CleanupEpisodeRunner(
            environment=environment,
            verifier=SuccessVerifier(),
            controller=controller,
        )

        result = runner.run(
            EpisodeContext(
                run_id="run-1",
                task_id="demo",
                method="closed_loop_vla",
                seed=0,
                episode=0,
                init_state_index=0,
            )
        )

        self.assertTrue(environment.reset_called)
        self.assertTrue(environment.closed)
        self.assertEqual(len(environment.instructions), 2)
        self.assertTrue(result.success)
        self.assertEqual(result.leftover_count, 0)
        self.assertEqual(
            state.get("object_1").status,
            ObjectStatus.IN_BASKET,
        )
        self.assertEqual(
            state.get("object_2").status,
            ObjectStatus.IN_BASKET,
        )

    def test_retry_is_counted_even_after_eventual_success(self) -> None:
        state = TaskState.from_labels("demo", ("mug",))
        runner = CleanupEpisodeRunner(
            environment=FakeEnvironment(success=True),
            verifier=FailOnceVerifier(),
            controller=CleanupController(state, max_attempts=2),
        )

        result = runner.run(
            EpisodeContext(
                run_id="run-retry",
                task_id="demo",
                method="closed_loop_vla",
                seed=0,
                episode=0,
                init_state_index=0,
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.vla_calls, 2)
        self.assertEqual(result.repeated_actions, 1)

    def test_simulator_verdict_is_authoritative(self) -> None:
        state = TaskState.from_labels("demo", ("mug",))
        runner = CleanupEpisodeRunner(
            environment=FakeEnvironment(success=False),
            verifier=SuccessVerifier(),
            controller=CleanupController(state),
        )

        result = runner.run(
            EpisodeContext(
                run_id="run-2",
                task_id="demo",
                method="closed_loop_vla",
                seed=0,
                episode=1,
                init_state_index=1,
            )
        )

        self.assertTrue(state.is_cleared())
        self.assertFalse(result.success)
        self.assertEqual(result.terminal_reason, "simulator_rejected_clear")


if __name__ == "__main__":
    unittest.main()

