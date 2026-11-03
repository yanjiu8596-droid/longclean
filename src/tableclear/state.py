from __future__ import annotations

from collections.abc import Iterable, Mapping

from .models import (
    CleanupVerdict,
    ObjectRecord,
    ObjectStatus,
    Subtask,
    VerdictStatus,
)


class NoActionableObjectError(RuntimeError):
    """Raised when the controller cannot select another cleanup action."""


class TaskState:
    """Object-level state for one table-clearing episode."""

    def __init__(self, task_id: str, objects: Mapping[str, ObjectRecord]) -> None:
        if not task_id.strip():
            raise ValueError("task_id must not be empty")
        if not objects:
            raise ValueError("task must contain at least one object")
        self.task_id = task_id
        self._objects = dict(objects)
        if len(self._objects) != len(objects):
            raise ValueError("object ids must be unique")

    @classmethod
    def from_labels(cls, task_id: str, labels: Iterable[str]) -> TaskState:
        records: dict[str, ObjectRecord] = {}
        for index, label in enumerate(labels, start=1):
            object_id = f"object_{index}"
            records[object_id] = ObjectRecord(id=object_id, label=label)
        return cls(task_id=task_id, objects=records)

    @property
    def objects(self) -> tuple[ObjectRecord, ...]:
        return tuple(self._objects.values())

    def get(self, object_id: str) -> ObjectRecord:
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown object id: {object_id}") from exc

    def pending_objects(self) -> tuple[ObjectRecord, ...]:
        return tuple(
            obj
            for obj in self._objects.values()
            if obj.status
            in {
                ObjectStatus.PENDING,
                ObjectStatus.GRASPED,
                ObjectStatus.LOST_GRIP,
                ObjectStatus.MISSING,
            }
        )

    def next_actionable(self, max_attempts: int = 2) -> ObjectRecord:
        for obj in self._objects.values():
            if obj.status in {
                ObjectStatus.PENDING,
                ObjectStatus.GRASPED,
                ObjectStatus.LOST_GRIP,
                ObjectStatus.MISSING,
            } and obj.attempts < max_attempts:
                return obj
        raise NoActionableObjectError("no actionable objects remain")

    def instruction_for(self, object_id: str) -> Subtask:
        obj = self.get(object_id)
        if obj.status == ObjectStatus.GRASPED:
            instruction = f"Place the {obj.label} into the basket."
        elif obj.status == ObjectStatus.MISSING:
            instruction = (
                f"Locate the {obj.label} on the table and place it into the basket."
            )
        else:
            instruction = (
                f"Pick up the {obj.label} and place it into the basket."
            )
        return Subtask(target_id=obj.id, instruction=instruction)

    def record_attempt(self, object_id: str) -> None:
        obj = self.get(object_id)
        obj.attempts += 1

    def apply_verdict(
        self,
        verdict: CleanupVerdict,
        max_attempts: int = 2,
    ) -> ObjectStatus:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        obj = self.get(verdict.target_id)

        if verdict.status == VerdictStatus.IN_BASKET:
            obj.status = ObjectStatus.IN_BASKET
        elif verdict.status == VerdictStatus.GRASPED:
            obj.status = ObjectStatus.GRASPED
        elif verdict.status == VerdictStatus.MISSING:
            obj.status = ObjectStatus.MISSING
            if obj.attempts >= max_attempts:
                obj.status = ObjectStatus.UNREACHABLE
        else:
            obj.status = ObjectStatus.LOST_GRIP
            if obj.attempts >= max_attempts:
                obj.status = ObjectStatus.UNREACHABLE

        return obj.status

    def remaining_ids(self) -> tuple[str, ...]:
        return tuple(
            obj.id
            for obj in self._objects.values()
            if obj.status != ObjectStatus.IN_BASKET
        )

    def successful_ids(self) -> tuple[str, ...]:
        return tuple(
            obj.id
            for obj in self._objects.values()
            if obj.status == ObjectStatus.IN_BASKET
        )

    def is_cleared(self) -> bool:
        return all(
            obj.status == ObjectStatus.IN_BASKET
            for obj in self._objects.values()
        )

    def can_request_finish(self) -> bool:
        return not self.pending_objects()


__all__ = ["NoActionableObjectError", "TaskState"]

