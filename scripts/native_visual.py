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
OBJECT_APPEARANCE = {
    "alphabet soup can": "short cylindrical can with a blue and yellow label and a silver lid",
    "tomato sauce": "short cylindrical can with red bands and a green stripe on its label",
    "cream cheese box": "small flat light-blue and white rectangular carton",
    "butter": "small thin orange-red and yellow rectangular carton",
}


class VisualPlanner:
    def __init__(
        self, model_path: Path, task_id: str, labels: list[str], *,
        fixed: bool, strategy: str = "original", min_basket_margin: float | None = None,
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
        self.min_basket_margin = min_basket_margin
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
        if self.strategy in {"appearance", "basket_zoom", "basket_wrist", "basket_guard"}:
            appearance = OBJECT_APPEARANCE[label]
            wrist = images[1]
            images = images[:1]
            if self.strategy != "appearance":
                from PIL import Image

                view = images[0]
                # The external camera is fixed in these three tasks.
                box = tuple(round(value * dimension / 256) for value, dimension in zip(
                    (0, 80, 82, 202), (*view.size, *view.size),
                ))
                closeup = view.crop(box).resize((336, 504), Image.Resampling.BICUBIC)
                images.append(closeup)
            if self.strategy in {"basket_wrist", "basket_guard"}:
                images.append(wrist)
            question = (
                f"Find the {appearance}. The white-lined wicker basket is on the far left of the first image. "
                + ("The second image enlarges the basket. " if self.strategy != "appearance" else "")
                + ("The third image looks down from the gripper and can distinguish an item held above "
                   "the basket from an item resting inside. " if len(images) == 3 else "")
                + "Where is the named item? Use 1 for inside the white-lined basket, 2 for outside the basket "
                "on the brown worktop, 3 for held between the robot fingers, or 4 for fully hidden. "
                "Answer with one digit."
            )
            if self.strategy == "basket_guard":
                question = (
                    f"Locate only the {appearance}. Image 1 shows the entire worktop, image 2 is a magnified "
                    "crop around the white-lined wicker basket, and image 3 is the wrist camera. "
                    "Check the full worktop for the matching item first. Answer 2 if the target is on the "
                    "worktop outside the basket. Answer 3 if it is pinched in the gripper, even directly above "
                    "the basket. Answer 1 only when its matching shape and color are visible resting inside "
                    "the white lining. A different item inside the basket does not count. If neither view "
                    "identifies the named item, answer 4. Output exactly one digit."
                )
        elif self.strategy != "original":
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
        text, margin = self.generate(images, question, score_digits=self.min_basket_margin is not None)
        token = text.strip(". \n\t").lower()
        status = statuses.get(token, VerdictStatus.UNKNOWN)
        result = {"target_id": target_id, "target": label, "status": status.value, "raw": text, "parsed": token in statuses}
        if margin is not None:
            result["basket_margin"] = margin
            if status == VerdictStatus.IN_BASKET and margin < self.min_basket_margin:
                result["status"] = VerdictStatus.UNKNOWN.value
        return result

    def answer(self, images: list, question: str) -> str:
        return self.generate(images, question, score_digits=False)[0]

    def generate(self, images: list, question: str, *, score_digits: bool) -> tuple[str, float | None]:
        messages = [{"role": "user", "content": [
            *({"type": "image"} for _ in images), {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        batch = self.processor(text=prompt, images=images, return_tensors="pt").to("cuda", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            result = self.model.generate(
                **batch, max_new_tokens=12, do_sample=False,
                return_dict_in_generate=score_digits, output_scores=score_digits,
            )
        sequences = result.sequences if score_digits else result
        text = self.processor.decode(sequences[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not score_digits:
            return text, None
        token_ids = [self.processor.tokenizer.encode(digit, add_special_tokens=False) for digit in STATUSES]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError("digit confidence requires four individual answer tokens")
        scores = result.scores[0][0, [ids[0] for ids in token_ids]].float()
        return text, (scores[0] - scores[1:].max()).item()

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
