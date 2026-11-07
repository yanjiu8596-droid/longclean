from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from native_method_schedule import MethodSchedule, accepted_visual_provider
from native_temporal_vision import TemporalVision, release_steps
from native_rgbd_placement import RGBD_RELEASE, has_deposit
from native_visual import VisualPlanner
from run_native_methods import jobs_for
from run_native_osc import ReleaseSampling
from run_overnight_native import visual_audit
from tableclear.state import TaskState


class NativeMethodScheduleTests(unittest.TestCase):
    def test_uncolored_depth_change_cannot_confirm_a_shifted_basket(self):
        evidence = {"fixed_foreground_pixels": 259, "target_color_pixels": [[0, 0], [0, 0]]}
        self.assertFalse(has_deposit(evidence, upright=False))
        self.assertTrue(has_deposit(evidence, upright=True))

    def test_release_detection_requires_sustained_close_and_open(self):
        commands = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        self.assertEqual(release_steps(commands), [11, 18])

    def test_release_window_waits_for_observed_opening(self):
        history = TemporalVision()
        history.observe(0, [], [], [], [0.04, -0.04])
        history.observe(10, [], [1] * 10, [], [0.021, -0.021])
        history.observe(20, [], [1] * 4 + [-1] * 6, [], [0.023, -0.024])
        self.assertEqual(list(history.release_windows()), [])
        history.observe(30, [], [-1] * 10, [], [0.039, -0.039])
        self.assertEqual([(command, before.step, after.step) for command, before, after in history.release_windows()], [(15, 10, 30)])

    def test_rgbd_release_detects_meaningful_opening_before_full_aperture(self):
        history = TemporalVision()
        history.observe(0, [], [], [], [0.0385, -0.0385])
        history.observe(10, [], [1] * 10, [], [0.021, -0.021])
        history.observe(20, [], [1] * 4 + [-1] * 6, [], [0.0234, -0.0235])
        self.assertEqual(list(history.release_windows(rgbd=True)), [])
        history.observe(30, [], [-1] * 10, [], [0.0265, -0.0265])
        self.assertEqual([(before.step, after.step) for _, before, after in history.release_windows(rgbd=True)], [(10, 30)])

    def test_dense_sampling_starts_on_first_open_and_has_a_bounded_window(self):
        sampling = ReleaseSampling()
        commands = [-1, 1, 1, -1, 1, 1, 1] + [-1] * 14 + [1, 1, 1, -1]
        self.assertEqual([step for step, value in enumerate(commands, 1) if sampling.observe(value)],
                         [*range(8, 20), 25])

    def test_position_positive_without_release_leaves_target_active(self):
        history = TemporalVision()
        history.observe(0, [], [], [], [0.04, -0.04])
        history.observe(10, [], [1] * 10, [], [0.02, -0.02])
        history.observe(12, [], [-1, -1], [], [0.022, -0.022])
        planner = Mock()
        planner.inspect_consensus.return_value = {"status": "in_basket", "inferences": 1, "parsed": True}
        with (patch("native_temporal_vision.basket_difference", return_value=0),
              patch("native_temporal_vision.wall_still_visible", return_value=True),
              patch("native_temporal_vision.entering_basket", return_value={"supported": True}),
              patch.object(history, "current_hold", return_value=(None, 0)),
              patch.object(history, "vote_grasp", return_value=({"label": "tomato sauce", "parsed": True}, 2))):
            decision = history.inspect(planner, "object_1", "tomato sauce", RGBD_RELEASE)
        self.assertTrue(decision["in_basket_region"])
        self.assertFalse(decision["release_confirmed"])
        self.assertIsNone(decision["confirmed_frame"])
        schedule = self.schedule("closed_loop_vla")
        schedule.action(0, "")
        schedule.apply_visual(decision["status"], 12)
        self.assertEqual(schedule.action(12, "")[0], "object_1")

    def test_runtime_audit_separates_position_and_release(self):
        base = {"target": "tomato sauce", "stage": "progress", "parsed": True, "in_basket_region": True,
                "audit_goals": [{"predicate": ["in", "tomato_sauce_1", "basket"], "satisfied": True}]}
        rows = [{**base, "status": "grasped", "release_confirmed": False},
                {**base, "status": "in_basket", "release_confirmed": True},
                {**base, "status": "in_basket", "release_confirmed": False}]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "visual.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            scores = visual_audit(output)
        self.assertEqual(scores["basket_mismatches"], 0)
        self.assertEqual(scores["confirmed_positives"], 1)
        self.assertEqual(scores["invalid_completions"], 1)

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
        self.assertEqual({job[:3] for job in jobs_for("smoke")}, {(1, 0, 15), (2, 0, 15), (8, 0, 15)})

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
