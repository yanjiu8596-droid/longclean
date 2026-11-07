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
LOCATION_WORDS = {
    "basket": VerdictStatus.IN_BASKET,
    "table": VerdictStatus.ON_TABLE,
    "gripper": VerdictStatus.GRASPED,
    "unknown": VerdictStatus.UNKNOWN,
}


class VisualPlanner:
    def __init__(
        self, model_path: Path, task_id: str, labels: list[str], *,
        fixed: bool, strategy: str = "original",
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
        ).to("cuda").eval()
        self.fixed = fixed
        self.strategy = strategy
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
        return self.inspect_images(images, target.id, target.label)

    def inspect_images(self, images: list, target_id: str, label: str) -> dict:
        question = (
            f"These are two simultaneous views of a robot table. Where is the {label}? "
            "Reply with exactly one digit. 1 = visibly inside the basket. "
            "2 = visible on the table, outside the basket. 3 = held by the robot gripper. "
            "4 = not visible or cannot tell. Use 4 if unsure."
        )
        statuses = STATUSES
        if self.strategy != "original":
            if self.strategy == "external":
                images = images[:1]
            elif self.strategy != "dual":
                raise ValueError(f"unsupported visual strategy: {self.strategy}")
            question = (
                f"Locate the {label} in this robot manipulation scene. The rectangular "
                "container with woven sides and a pale fabric lining is the basket. "
                "Answer BASKET only if the named object is visibly inside the basket. "
                "Answer TABLE if it is visibly outside the basket, on the work surface. "
                "Answer GRIPPER if it is visibly held between the robot fingers. "
                "Answer UNKNOWN if you cannot locate the named object. Reply with one word."
            )
            statuses = LOCATION_WORDS
        text = self.answer(images, question)
        token = text.strip(". \n\t").lower()
        status = statuses.get(token, VerdictStatus.UNKNOWN)
        return {"target_id": target_id, "target": label, "status": status.value, "raw": text, "parsed": token in statuses}

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
