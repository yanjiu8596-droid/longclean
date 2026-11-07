"""Action allocation for paired native OSC method experiments."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import tomllib

from tableclear.models import ObjectStatus, VerdictStatus
from tableclear.state import TaskState


class MethodSchedule:
    def __init__(self, method: str, state: TaskState, budget: int, chunk: int) -> None:
        self.method = method
        self.state = state
        self.budget = budget
        self.chunk = chunk
        self.targets = deque(obj.id for obj in state.objects)
        self.target = None
        self.slot_end = 0
        self.check_at = 0

    def action(self, steps: int, default_instruction: str) -> tuple[str | None, str, int] | None:
        if steps >= self.budget:
            return None
        if self.method == "direct_vla":
            return None, default_instruction, min(self.chunk, self.budget - steps)
        if self.method == "fixed_checklist_vla":
            count = len(self.targets)
            target_index = min(steps * count // self.budget, count - 1)
            self.target = self.targets[target_index]
            self.slot_end = ((target_index + 1) * self.budget + count - 1) // count
        elif self.target is None or steps >= self.slot_end:
            self.target = None
            while self.targets:
                target = self.targets.popleft()
                if self.state.get(target).status != ObjectStatus.IN_BASKET:
                    self.targets.append(target)
                    self.target = target
                    self.state.record_attempt(target)
                    # A retry changes focus without resetting the simulation.
                    self.slot_end = min(steps + 13 * self.chunk, self.budget)
                    self.check_at = steps + 5 * self.chunk
                    break
            if self.target is None:
                return None
        subtask = self.state.instruction_for(self.target)
        return self.target, subtask.instruction, min(self.chunk, self.slot_end - steps)

    def needs_visual_update(self, steps: int, terminal: bool) -> bool:
        return self.method == "closed_loop_vla" and (terminal or steps >= min(self.slot_end, self.check_at))

    def apply_visual(self, status: str, steps: int) -> None:
        verdict = VerdictStatus(status)
        target = self.state.get(self.target)
        self.check_at = steps + 5 * self.chunk
        if verdict == VerdictStatus.IN_BASKET:
            target.status = ObjectStatus.IN_BASKET
            self.target = None
        elif verdict == VerdictStatus.GRASPED:
            target.status = ObjectStatus.GRASPED
        elif verdict == VerdictStatus.ON_TABLE:
            target.status = ObjectStatus.PENDING


def accepted_visual_provider(report_path: Path, project: Path, workspace: Path) -> tuple[Path, str, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("gate_passed") is not True:
        raise ValueError("closed-loop evaluation requires a passed held-out visual validation")
    model_id = report["selected"].split("/", 1)[0]
    with (project / "versions.lock").open("rb") as handle:
        pin = tomllib.load(handle)["models"][model_id]
    provider = report["holdout"]
    if any(provider["model"][key] != pin[key] for key in ("repository", "revision")):
        raise ValueError("validated vision provider does not match the locked revision")
    path = workspace / "models" / model_id
    revision_file = path / "model-revision.txt"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() != pin["revision"]:
        raise ValueError("installed vision provider does not match the locked revision")
    return path, provider["strategy"], provider["model"]
