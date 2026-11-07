"""Match RGB-D grasp crops against frozen visual exemplars with DINOv2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DINO_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


@dataclass
class Reference:
    label: str
    camera: int | None
    image: object


class ReferenceMatcher:
    def __init__(self, path: Path, references: list[Reference]):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if (path / "model-revision.txt").read_text(encoding="utf-8").strip() != DINO_REVISION:
            raise ValueError("DINOv2 model revision does not match the experiment")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to("cuda").eval()
        self.references = references
        self.labels = list(dict.fromkeys(item.label for item in references))
        self.bank = self.encode([item.image for item in references])

    def encode(self, images):
        from PIL import Image, ImageOps

        views = [ImageOps.pad(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC, color=(64, 64, 64)) for image in images]
        inputs = self.processor(images=views, do_resize=False, do_center_crop=False, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            features = self.model(**inputs).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(features.float(), dim=-1)

    def classify(self, crops, *, held_references=True, min_margin=0.04):
        if not crops:
            return {"label": None, "top_label": None, "scores": {}, "margin": None}
        matrix = (self.encode([image for _, image in crops]) @ self.bank.T).cpu().tolist()
        view_scores = []
        for (camera, _), scores in zip(crops, matrix, strict=True):
            candidates = {
                label: max(
                    score for ref, score in zip(self.references, scores, strict=True)
                    if ref.label == label and (ref.camera is None or held_references and ref.camera == camera)
                )
                for label in self.labels
            }
            view_scores.append(candidates)
        combined = {label: sum(scores[label] for scores in view_scores) / len(view_scores) for label in self.labels}
        ranked = sorted(combined, key=combined.get, reverse=True)
        margin = combined[ranked[0]] - combined[ranked[1]]
        return {
            "label": ranked[0] if margin >= min_margin else None, "top_label": ranked[0],
            "scores": combined, "margin": margin, "by_view": view_scores,
        }
