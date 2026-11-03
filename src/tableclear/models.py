from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObjectStatus(str, Enum):
    PENDING = "pending"
    GRASPED = "grasped"
    IN_BASKET = "in_basket"
    MISSING = "missing"
    LOST_GRIP = "lost_grip"
    BLOCKED = "blocked"
    UNREACHABLE = "unreachable"


class VerdictStatus(str, Enum):
    ON_TABLE = "on_table"
    GRASPED = "grasped"
    IN_BASKET = "in_basket"
    MISSING = "missing"
    NO_PROGRESS = "no_progress"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ObjectRecord:
    id: str
    label: str
    status: ObjectStatus = ObjectStatus.PENDING
    attempts: int = 0
    last_seen_region: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("object id must not be empty")
        if not self.label.strip():
            raise ValueError("object label must not be empty")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")


@dataclass(slots=True)
class CleanupVerdict:
    target_id: str
    status: VerdictStatus
    progress: float
    confidence: float
    evidence: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class Subtask:
    target_id: str
    instruction: str
    max_actions: int = 10

    def __post_init__(self) -> None:
        if self.max_actions < 1:
            raise ValueError("max_actions must be positive")


@dataclass(frozen=True, slots=True)
class ActionChunk:
    actions: tuple[tuple[float, ...], ...]
    action_space: str = "pose_delta"
    gripper_index: int = 6
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("action chunk must not be empty")
        width = len(self.actions[0])
        if width == 0:
            raise ValueError("actions must not be empty vectors")
        if any(len(action) != width for action in self.actions):
            raise ValueError("all actions must have the same dimensions")
        if not 0 <= self.gripper_index < width:
            raise ValueError("gripper_index is outside the action vector")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")


@dataclass(slots=True)
class EpisodeResult:
    run_id: str
    task_id: str
    method: str
    seed: int
    episode: int
    success: bool
    leftover_count: int
    repeated_actions: int
    terminal_reason: str
    steps: int
    agent_calls: int
    vla_calls: int
    runtime_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.leftover_count < 0:
            raise ValueError("leftover_count must be non-negative")
        if self.repeated_actions < 0:
            raise ValueError("repeated_actions must be non-negative")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.agent_calls < 0:
            raise ValueError("agent_calls must be non-negative")
        if self.vla_calls < 0:
            raise ValueError("vla_calls must be non-negative")
        if self.runtime_s < 0:
            raise ValueError("runtime_s must be non-negative")

