from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from native_method_schedule import MethodSchedule
from run_native_methods import jobs_for
from tableclear.state import TaskState


class NativeMethodScheduleTests(unittest.TestCase):
    def schedule(self, method):
        return MethodSchedule(method, TaskState.from_labels("scene", ["can", "box"]), 520, 10)

    def test_each_method_can_use_full_budget_when_simulator_never_succeeds(self):
        for method in ("direct_vla", "fixed_checklist_vla", "closed_loop_vla"):
            with self.subTest(method=method):
                schedule = self.schedule(method)
                targets = []
                for steps in range(0, 520, 10):
                    target, _, count = schedule.action(steps, "full instruction")
                    self.assertEqual(count, 10)
                    targets.append(target)
                    if schedule.needs_visual_update(steps + count, False):
                        schedule.apply_visual("unknown", steps + count)
                self.assertIsNone(schedule.action(520, "full instruction"))
                if method == "fixed_checklist_vla":
                    self.assertEqual(targets, ["object_1"] * 26 + ["object_2"] * 26)
                if method == "closed_loop_vla":
                    self.assertEqual(targets, (["object_1"] * 13 + ["object_2"] * 13) * 2)

    def test_confirmed_progress_changes_target_and_preserves_remaining_budget(self):
        schedule = self.schedule("closed_loop_vla")
        for steps in range(0, 50, 10):
            self.assertEqual(schedule.action(steps, "")[0], "object_1")
        self.assertTrue(schedule.needs_visual_update(50, False))
        schedule.apply_visual("in_basket", 50)
        for steps in range(50, 520, 10):
            self.assertEqual(schedule.action(steps, "")[0], "object_2")
        self.assertIsNone(schedule.action(520, ""))

    def test_formal_jobs_share_45_conditions_and_reserve_pilot_indices(self):
        formal = jobs_for("formal")
        self.assertEqual(len(formal), 135)
        self.assertEqual(len({job[:3] for job in formal}), 45)
        self.assertEqual({job[2] for job in formal}, set(range(15)))
        self.assertTrue(all(job[2] >= 15 for job in jobs_for("pilot")))
