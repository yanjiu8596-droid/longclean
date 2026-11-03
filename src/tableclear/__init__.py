"""TableClear V1 core package."""

from .models import (
    ActionChunk,
    CleanupVerdict,
    EpisodeResult,
    ObjectRecord,
    ObjectStatus,
    Subtask,
    VerdictStatus,
)
from .interfaces import (
    EnvironmentAdapter,
    FinalVerdict,
    Observation,
    ProgressVerifier,
    SkillExecution,
    VLAAdapter,
)
from .aggregate import MethodSummary, aggregate_episode_records
from .state import TaskState
from .runner import CleanupEpisodeRunner, EpisodeContext

__all__ = [
    "ActionChunk",
    "CleanupVerdict",
    "CleanupEpisodeRunner",
    "EnvironmentAdapter",
    "EpisodeContext",
    "EpisodeResult",
    "FinalVerdict",
    "MethodSummary",
    "ObjectRecord",
    "ObjectStatus",
    "Observation",
    "ProgressVerifier",
    "SkillExecution",
    "Subtask",
    "TaskState",
    "VLAAdapter",
    "VerdictStatus",
    "aggregate_episode_records",
]

__version__ = "0.1.0"

