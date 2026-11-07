from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from native_method_schedule import MethodSchedule, accepted_visual_provider
from native_temporal_vision import release_steps
from native_visual import VisualPlanner
from run_native_methods import jobs_for
from tableclear.state import TaskState


class NativeMethodScheduleTests(unittest.TestCase):
    def test_release_detection_requires_sustained_close_and_open(self):
        commands = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        self.assertEqual(release_steps(commands), [11, 18])

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

    def test_temporal_invalidation_reopens_a_retired_target(self):
        schedule = self.schedule("closed_loop_vla")
        schedule.action(0, "")
        schedule.apply_visual("in_basket", 50)
        schedule.action(50, "")
        schedule.apply_visual("in_basket", 100)
        self.assertIsNone(schedule.action(100, ""))
        schedule.reopen("object_1")
        self.assertEqual(schedule.action(100, "")[0], "object_1")

    def test_formal_jobs_share_45_conditions_and_reserve_pilot_indices(self):
        formal = jobs_for("formal")
        self.assertEqual(len(formal), 135)
        self.assertEqual(len({job[:3] for job in formal}), 45)
        self.assertEqual({job[2] for job in formal}, set(range(15)))
        self.assertTrue(all(job[2] >= 15 for job in jobs_for("pilot")))

    def test_failed_visual_gate_can_only_be_used_in_a_runtime_probe(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "validation.json"
            report.write_text(json.dumps({
                "status": "holdout_failed", "gate_passed": False,
                "selected": "qwen3_vl_4b_instruct/basket_zoom",
                "holdout": {
                    "strategy": "basket_zoom",
                    "model": {
                        "repository": "Qwen/Qwen3-VL-4B-Instruct",
                        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                    },
                },
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                accepted_visual_provider(report, project, root)
            _, provider = accepted_visual_provider(report, project, root, runtime_probe=True)
            self.assertEqual(provider["strategy"], "basket_zoom")

    def test_consensus_only_uses_reference_for_a_disagreement(self):
        planner = VisualPlanner.__new__(VisualPlanner)
        planner.strategy = "basket_consensus"
        called = []
        statuses = {"basket_zoom": "in_basket", "basket_wrist": "on_table", "example_zoom": "in_basket"}

        def inspect(_images, _target, _label, strategy, _margin):
            called.append(strategy)
            return {"status": statuses[strategy], "raw": "test", "parsed": True}

        planner.inspect_with_strategy = inspect
        self.assertEqual(planner.inspect_images([], "object_1", "can")["status"], "in_basket")
        self.assertEqual(called, ["basket_zoom", "basket_wrist", "example_zoom"])
        called.clear()
        statuses["basket_zoom"] = "on_table"
        self.assertEqual(planner.inspect_images([], "object_1", "can")["status"], "on_table")
        self.assertEqual(called, ["basket_zoom", "basket_wrist"])
