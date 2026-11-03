from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .controller import CleanupController
from .interfaces import EnvironmentAdapter, FinalVerdict, Observation, ProgressVerifier
from .models import (
    CleanupVerdict,
    EpisodeResult,
    VerdictStatus,
)
from .state import NoActionableObjectError, TaskState


class EventSink(Protocol):
    def write(self, event_type: str, payload: dict[str, object]) -> None:
        ...


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    run_id: str
    task_id: str
    method: str
    seed: int
    episode: int
    init_state_index: int


class CleanupEpisodeRunner:
    """Run one complete object-level cleanup episode."""

    def __init__(
        self,
        *,
        environment: EnvironmentAdapter,
        verifier: ProgressVerifier,
        controller: CleanupController,
        max_rounds: int = 100,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.environment = environment
        self.verifier = verifier
        self.controller = controller
        self.max_rounds = max_rounds

    def run(
        self,
        context: EpisodeContext,
        *,
        events: EventSink | None = None,
    ) -> EpisodeResult:
        started = time.monotonic()
        state = self.controller.state
        observation = self.environment.reset(
            init_state_index=context.init_state_index
        )
        vla_calls = 0
        steps = 0
        terminal_reason = "round_limit"

        if events is not None:
            events.write(
                "episode_start",
                {
                    "run_id": context.run_id,
                    "task_id": context.task_id,
                    "method": context.method,
                    "seed": context.seed,
                    "episode": context.episode,
                    "init_state_index": context.init_state_index,
                    "objects": [obj.label for obj in state.objects],
                },
            )

        try:
            for _ in range(self.max_rounds):
                if self.controller.is_cleared():
                    terminal_reason = "cleared"
                    break

                try:
                    subtask = self.controller.next_subtask()
                except NoActionableObjectError:
                    terminal_reason = "no_actionable_objects"
                    break

                before = observation
                if events is not None:
                    events.write(
                        "subtask_start",
                        {
                            "run_id": context.run_id,
                            "target_id": subtask.target_id,
                            "instruction": subtask.instruction,
                            "attempt": state.get(subtask.target_id).attempts,
                        },
                    )

                execution = self.environment.execute(subtask)
                vla_calls += 1
                steps += int(execution.metadata.get("steps", 0))

                if execution.status != "completed":
                    verdict = CleanupVerdict(
                        target_id=subtask.target_id,
                        status=VerdictStatus.NO_PROGRESS,
                        progress=0.0,
                        confidence=1.0,
                        evidence=execution.error_code or "execution_failed",
                    )
                else:
                    after = execution.observation
                    if after is None:
                        after = self.environment.observe()
                    verdict = self.verifier.verify(
                        before=before,
                        after=after,
                        target_id=subtask.target_id,
                        instruction=subtask.instruction,
                    )
                    observation = after

                status = self.controller.apply_verdict(verdict)
                if events is not None:
                    events.write(
                        "subtask_end",
                        {
                            "run_id": context.run_id,
                            "target_id": subtask.target_id,
                            "verdict": verdict.status.value,
                            "object_status": status.value,
                            "object_progress": verdict.progress,
                            "verifier_confidence": verdict.confidence,
                            "evidence": verdict.evidence,
                            "remaining": list(state.remaining_ids()),
                        },
                    )
            else:
                terminal_reason = "round_limit"
        finally:
            final = self.environment.score()
            self.environment.close()

        if not isinstance(final, FinalVerdict):
            raise TypeError("environment.score() must return FinalVerdict")
        if final.success:
            terminal_reason = "simulator_success"
        elif state.is_cleared():
            terminal_reason = "simulator_rejected_clear"

        repeated_actions = sum(
            max(0, obj.attempts - 1)
            for obj in state.objects
        )
        result = EpisodeResult(
            run_id=context.run_id,
            task_id=context.task_id,
            method=context.method,
            seed=context.seed,
            episode=context.episode,
            success=final.success,
            leftover_count=len(state.remaining_ids()),
            repeated_actions=repeated_actions,
            terminal_reason=terminal_reason,
            steps=steps,
            agent_calls=0,
            vla_calls=vla_calls,
            runtime_s=time.monotonic() - started,
            metadata={
                "simulator_score": final.score,
                "simulator_reason": final.reason,
            },
        )
        if events is not None:
            events.write(
                "episode_end",
                {
                    "run_id": context.run_id,
                    "task_id": context.task_id,
                    "method": context.method,
                    "seed": context.seed,
                    "episode": context.episode,
                    "success": result.success,
                    "leftover_count": result.leftover_count,
                    "repeated_actions": result.repeated_actions,
                    "terminal_reason": result.terminal_reason,
                    "steps": result.steps,
                    "vla_calls": result.vla_calls,
                    "runtime_s": result.runtime_s,
                    "simulator_score": final.score,
                },
            )
        return result


def observation_from_parts(
    images: dict[str, str],
    robot_state: tuple[float, ...],
    sim_time_s: float,
) -> Observation:
    """Small convenience factory for adapters and tests."""

    return Observation(
        images=images,
        robot_state=robot_state,
        sim_time_s=sim_time_s,
    )


__all__ = [
    "CleanupEpisodeRunner",
    "EpisodeContext",
    "EventSink",
    "observation_from_parts",
]

