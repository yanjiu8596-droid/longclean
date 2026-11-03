from __future__ import annotations

import unittest

from tableclear.controller import CleanupController
from tableclear.models import CleanupVerdict, ObjectStatus, VerdictStatus
from tableclear.state import NoActionableObjectError, TaskState


class TaskStateTests(unittest.TestCase):
    def test_cleanup_requires_every_object(self) -> None:
        state = TaskState.from_labels("demo", ("mug", "bowl"))
        state.get("object_1").status = ObjectStatus.IN_BASKET

        self.assertFalse(state.is_cleared())
        self.assertEqual(state.remaining_ids(), ("object_2",))

    def test_successful_verdict_clears_object(self) -> None:
        state = TaskState.from_labels("demo", ("mug",))
        controller = CleanupController(state, max_attempts=2)
        subtask = controller.next_subtask()

        controller.apply_verdict(
            CleanupVerdict(
                target_id=subtask.target_id,
                status=VerdictStatus.IN_BASKET,
                progress=1.0,
                confidence=0.9,
            )
        )

        self.assertTrue(state.is_cleared())
        self.assertEqual(state.remaining_ids(), ())

    def test_retry_exhaustion_marks_object_unreachable(self) -> None:
        state = TaskState.from_labels("demo", ("mug",))
        controller = CleanupController(state, max_attempts=2)

        first = controller.next_subtask()
        controller.apply_verdict(
            CleanupVerdict(
                target_id=first.target_id,
                status=VerdictStatus.NO_PROGRESS,
                progress=0.0,
                confidence=0.9,
            )
        )
        self.assertEqual(state.get("object_1").status, ObjectStatus.LOST_GRIP)

        second = controller.next_subtask()
        controller.apply_verdict(
            CleanupVerdict(
                target_id=second.target_id,
                status=VerdictStatus.NO_PROGRESS,
                progress=0.0,
                confidence=0.9,
            )
        )

        self.assertEqual(state.get("object_1").status, ObjectStatus.UNREACHABLE)
        self.assertFalse(state.is_cleared())
        with self.assertRaises(NoActionableObjectError):
            controller.next_subtask()

    def test_grasped_object_gets_place_instruction(self) -> None:
        state = TaskState.from_labels("demo", ("mug",))
        controller = CleanupController(state, max_attempts=3)
        first = controller.next_subtask()
        controller.apply_verdict(
            CleanupVerdict(
                target_id=first.target_id,
                status=VerdictStatus.GRASPED,
                progress=0.5,
                confidence=0.9,
            )
        )

        second = controller.next_subtask()
        self.assertIn("Place the mug", second.instruction)
        self.assertEqual(state.get("object_1").attempts, 2)


if __name__ == "__main__":
    unittest.main()

