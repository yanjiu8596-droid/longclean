from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import ActionChunk, CleanupVerdict, Subtask


@dataclass(frozen=True, slots=True)
class Observation:
    images: dict[str, str]
    robot_state: tuple[float, ...]
    sim_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillExecution:
    status: str
    observation: Observation | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinalVerdict:
    success: bool
    score: float
    reason: str


class EnvironmentAdapter(Protocol):
    """Contract for LIBERO/RoboEnv or a future real-robot adapter."""

    def reset(self, *, init_state_index: int) -> Observation:
        ...

    def observe(self) -> Observation:
        ...

    def execute(self, subtask: Subtask) -> SkillExecution:
        ...

    def score(self) -> FinalVerdict:
        ...

    def close(self) -> None:
        ...


class VLAAdapter(Protocol):
    """Replaceable execution policy, initially implemented by SmolVLA."""

    def infer(
        self,
        observation: Observation,
        instruction: str,
        max_actions: int,
    ) -> ActionChunk:
        ...


class ProgressVerifier(Protocol):
    """Visual verifier used by the closed-loop method."""

    def verify(
        self,
        before: Observation,
        after: Observation,
        target_id: str,
        instruction: str,
    ) -> CleanupVerdict:
        ...

