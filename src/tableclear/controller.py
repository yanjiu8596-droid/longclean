from __future__ import annotations

from .models import CleanupVerdict, ObjectStatus, Subtask
from .state import NoActionableObjectError, TaskState


class CleanupController:
    """Deterministic V1 controller around the object-level task state."""

    def __init__(
        self,
        state: TaskState,
        max_attempts: int = 3,
        max_actions: int = 10,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.state = state
        self.max_attempts = max_attempts
        self.max_actions = max_actions

    def next_subtask(self) -> Subtask:
        if self.state.is_cleared():
            raise NoActionableObjectError("table is already clear")
        obj = self.state.next_actionable(max_attempts=self.max_attempts)
        self.state.record_attempt(obj.id)
        subtask = self.state.instruction_for(obj.id)
        subtask.max_actions = self.max_actions
        return subtask

    def apply_verdict(self, verdict: CleanupVerdict) -> ObjectStatus:
        return self.state.apply_verdict(
            verdict=verdict,
            max_attempts=self.max_attempts,
        )

    def mark_unreachable(self, object_id: str) -> None:
        self.state.get(object_id).status = ObjectStatus.UNREACHABLE

    def can_finish(self) -> bool:
        return self.state.can_request_finish()

    def is_cleared(self) -> bool:
        return self.state.is_cleared()

