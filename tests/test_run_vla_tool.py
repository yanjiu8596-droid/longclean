from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / ".kinex" / "tools" / "run_vla_subtask" / "tool.py"


def load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_vla_subtask_tool", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_vla_subtask tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunVLAToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = load_tool_module()
        self.read_count = 0
        self.stop_count = 0
        self.run_policy_args: dict[str, Any] = {}

    def device(self, name: str, action: str = "read", args: Any = None) -> dict[str, Any]:
        if action == "read" and name in {"camera", "camera_wrist"}:
            self.read_count += 1
            return {"file": f"{name}-{self.read_count}.png"}
        if action == "wait" and name == "world":
            return {"waited_s": args["duration_s"]}
        if action == "read" and name == "arm":
            return {"positions": [0.0] * 7}
        if action == "fk" and name == "arm":
            self.assertNotIn("reference_frame", args)
            return {"pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]}
        if action == "read" and name == "gripper":
            return {"aperture_m": 0.04}
        raise AssertionError(f"unexpected device call: {name} {action} {args}")

    def profile(self, name: str) -> dict[str, Any]:
        self.assertEqual(name, "arm")
        return {
            "extrinsics": {
                "mount": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }

    def call(self, tool: str, **kwargs: Any) -> dict[str, Any]:
        self.assert_tool_name(tool)
        self.run_policy_args = kwargs
        return {
            "actions_submitted": kwargs["max_actions"],
            "duration_s": 0.5,
            "execution_ids": ["exec-1"],
            "ik_failure": False,
        }

    def assert_tool_name(self, tool: str) -> None:
        if tool != "run_policy":
            raise AssertionError(tool)

    def stop(self) -> None:
        self.stop_count += 1

    def test_executes_bounded_policy_and_returns_fresh_evidence(self) -> None:
        result = self.tool.execute(
            {
                "instruction": "Pick up the mug and place it into the basket.",
                "max_actions": 10,
            },
            device=self.device,
            call=self.call,
            stop_owned=self.stop,
            profile=self.profile,
        )

        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["actions_submitted"], 10)
        self.assertEqual(result["before_images"], ["camera-1.png", "camera_wrist-2.png"])
        self.assertEqual(result["after_images"], ["camera-3.png", "camera_wrist-4.png"])
        self.assertEqual(result["observation_state"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -0.02, 0.02])
        self.assertEqual(self.run_policy_args["observation"]["state"], result["observation_state"])
        self.assertEqual(self.run_policy_args["command"][0]["space"], "pose_delta")
        self.assertEqual(
            self.run_policy_args["command"][0]["scale"],
            [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        )
        self.assertEqual(
            self.run_policy_args["command"][1]["clip"],
            [-1.0, 1.0],
        )
        self.assertEqual(self.stop_count, 1)

    def test_rejects_invalid_action_budget(self) -> None:
        with self.assertRaises(self.tool.VLAToolError):
            self.tool.execute(
                {
                    "instruction": "Pick up the mug.",
                    "max_actions": 51,
                },
                device=self.device,
                call=self.call,
                stop_owned=self.stop,
                profile=self.profile,
            )

    def test_quaternion_conversion_uses_xyzw_order(self) -> None:
        half_angle = math.pi / 4.0
        axis_angle = self.tool._quaternion_xyzw_to_axis_angle(
            [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        )

        self.assertAlmostEqual(axis_angle[0], 0.0)
        self.assertAlmostEqual(axis_angle[1], 0.0)
        self.assertAlmostEqual(axis_angle[2], math.pi / 2.0)

    def test_composes_fk_pose_with_arm_mount(self) -> None:
        pose = self.tool._compose_mount(
            [0.5, 0.0, 0.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
            {
                "translation_m": [-0.5, 1.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )

        self.assertAlmostEqual(pose[0], 0.0)
        self.assertAlmostEqual(pose[1], 1.0)
        self.assertAlmostEqual(pose[2], 0.0)


if __name__ == "__main__":
    unittest.main()

