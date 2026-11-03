from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / ".kinex" / "tools" / "cleanup_state" / "tool.py"


def load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cleanup_state_tool", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cleanup_state tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanupStateToolTests(unittest.TestCase):
    def test_init_update_and_resume(self) -> None:
        tool = load_tool_module()
        workspace = ROOT / ".test-artifacts" / "kinex-tool"
        state_path = workspace / ".tableclear" / "state.json"
        state_path.unlink(missing_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)

        initialized = tool.execute(
            {
                "action": "init",
                "task_id": "tableclear-test",
                "objects": ["mug", "bowl"],
            },
            workspace,
        )
        self.assertEqual(len(initialized["objects"]), 2)
        self.assertEqual(initialized["remaining"], ["object_1", "object_2"])

        updated = tool.execute(
            {
                "action": "update",
                "target_id": "object_1",
                "status": "in_basket",
                "evidence": "mug visible in basket",
            },
            workspace,
        )
        self.assertEqual(updated["remaining"], ["object_2"])
        self.assertEqual(updated["in_basket"], ["object_1"])

        resumed = tool.execute({"action": "get"}, workspace)
        self.assertEqual(resumed["objects"]["object_1"]["status"], "in_basket")
        self.assertEqual(resumed["remaining"], ["object_2"])
        state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

