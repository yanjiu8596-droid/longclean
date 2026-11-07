"""Offline visual inventory and progress for the native OSC pilot."""

from __future__ import annotations

from pathlib import Path

from tableclear.controller import CleanupController
from tableclear.models import CleanupVerdict, ObjectStatus, VerdictStatus
from tableclear.state import NoActionableObjectError, TaskState


STATUSES = {
    "1": VerdictStatus.IN_BASKET,
    "2": VerdictStatus.ON_TABLE,
    "3": VerdictStatus.GRASPED,
    "4": VerdictStatus.UNKNOWN,
}


class VisualPlanner:
    def __init__(self, model_path: Path, task_id: str, labels: list[str], *, fixed: bool) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
        ).to("cuda").eval()
        self.fixed = fixed
        self.controller = CleanupController(
            TaskState.from_labels(task_id, labels), max_attempts=1 if fixed else 3, max_actions=10,
        )

    def inspect(self, observation: dict, target_id: str) -> dict:
        from PIL import Image

        target = self.controller.state.get(target_id)
        images = []
        for camera in ("camera1", "camera2"):
            frame = observation[f"observation.images.{camera}"][0].permute(1, 2, 0)
            images.append(Image.fromarray(frame.mul(255).round().byte().numpy()))
        question = (
            f"These are two simultaneous views of a robot table. Where is the {target.label}? "
            "Reply with exactly one digit. 1 = visibly inside the basket. "
            "2 = visible on the table, outside the basket. 3 = held by the robot gripper. "
            "4 = not visible or cannot tell. Use 4 if unsure."
        )
        text = self.answer(images, question)
        token = text.strip(". \n\t")
        status = STATUSES.get(token, VerdictStatus.UNKNOWN)
        return {"target_id": target_id, "target": target.label, "status": status.value, "raw": text, "parsed": token in STATUSES}

    def answer(self, images: list, question: str) -> str:
        messages = [{"role": "user", "content": [
            *({"type": "image"} for _ in images), {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        batch = self.processor(text=prompt, images=images, return_tensors="pt").to("cuda", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=12, do_sample=False)
        return self.processor.decode(result[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def inventory(self, observation: dict) -> list[dict]:
        results = []
        for target in self.controller.state.objects:
            result = self.inspect(observation, target.id)
            results.append(result)
            if result["status"] == VerdictStatus.IN_BASKET:
                target.status = ObjectStatus.IN_BASKET
            elif result["status"] == VerdictStatus.GRASPED:
                target.status = ObjectStatus.GRASPED
            elif result["status"] == VerdictStatus.UNKNOWN:
                target.status = ObjectStatus.UNREACHABLE if self.fixed else ObjectStatus.MISSING
        return results

    def next_subtask(self):
        try:
            return self.controller.next_subtask()
        except NoActionableObjectError:
            return None

    def update(self, observation: dict, target_id: str) -> dict:
        result = self.inspect(observation, target_id)
        status = VerdictStatus(result["status"])
        self.controller.apply_verdict(CleanupVerdict(
            target_id=target_id, status=status, progress=float(status == VerdictStatus.IN_BASKET),
            confidence=0.0, evidence=result["raw"],
        ))
        return result
