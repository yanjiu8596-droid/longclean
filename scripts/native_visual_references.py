"""Create reproducible visual exemplars from inspected calibration frames."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


SCENE_01 = "direct_horizon_diagnostics-01-seed0-init20-h5-direct_vla"
SCENE_02 = "direct_horizon_diagnostics-02-seed2-init18-h5-direct_vla"
RECIPES = {
    "alphabet soup can": (SCENE_01, (172, 118, 207, 170)),
    "tomato sauce": (SCENE_01, (99, 120, 132, 169)),
    "cream cheese box": (SCENE_02, (199, 173, 233, 216)),
    "butter": (SCENE_02, (91, 168, 118, 212)),
}


def create_references(run: Path, root: Path) -> dict:
    from PIL import Image, ImageOps

    destination = root / "references"
    destination.mkdir()
    manifest = {}
    for label, (artifact, box) in RECIPES.items():
        source = run / artifact / "initial-camera1.png"
        file = destination / f"{label.replace(' ', '-')}.png"
        with Image.open(source) as image:
            crop = image.convert("RGB").crop(box)
            reference = ImageOps.pad(crop, (224, 224), method=Image.Resampling.BICUBIC, color=(235, 235, 235))
            reference.save(file)
        manifest[label] = {
            "file": file.name, "sha256": sha256(file.read_bytes()).hexdigest(),
            "source_artifact": artifact, "source_sha256": sha256(source.read_bytes()).hexdigest(), "crop": box,
        }
    return manifest


def load_references(root: Path, manifest: dict) -> dict:
    from PIL import Image

    if set(manifest) != set(RECIPES):
        raise ValueError("visual validation reference manifest is incomplete")
    references = {}
    for label, entry in manifest.items():
        filename = entry["file"]
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("invalid reference image filename")
        path = root / "references" / filename
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"visual exemplar checksum does not match for {label}")
        with Image.open(path) as image:
            references[label] = image.convert("RGB")
    return references
