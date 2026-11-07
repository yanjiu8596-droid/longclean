"""Compare reference matching and the current VLM on labeled grasp replays."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import PROJECT, VisualPlanner, trajectory
from native_reference_identity import DINO_ID, DINO_REVISION, Reference, ReferenceMatcher
from native_rgbd_placement import held_crops, held_identity
from native_visual_references import load_references


def frame_at(folder, step):
    meta = json.loads((folder / "verified.json").read_bytes())
    history, sample_hash = trajectory(folder, meta["init_state_sha256"], step)
    return history.frames[-1], sample_hash, meta["init_state_sha256"]


def digest(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return sha256(buffer.getvalue()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    validation = json.loads(args.references.read_bytes())
    exemplars = load_references(args.references.parent, validation["references"])
    references = [Reference(label, None, image) for label, image in exemplars.items()]
    recipe = (
        ("cream cheese box", args.development / "early", (90, 100, 120), (0, 1)),
        ("butter", args.development / "butter", (220, 240), (0,)),
        ("alphabet soup can", args.development / "soup", (90, 100, 120), (0, 1)),
        ("tomato sauce", args.tomato, (220, 230, 250), (0, 1)),
    )
    gallery = []
    reference_hashes = set()
    for label, folder, steps, cameras in recipe:
        for step in steps:
            frame, sample_hash, initial_hash = frame_at(folder, step)
            reference_hashes.add(initial_hash)
            for camera, crop in held_crops(frame):
                if camera in cameras:
                    references.append(Reference(label, camera, crop))
                    gallery.append({
                        "label": label, "camera": camera, "step": step, "source": str(folder),
                        "initial_hash": initial_hash, "sample_sha256": sample_hash, "crop_sha256": digest(crop),
                    })

    matcher = ReferenceMatcher(args.dino, references)
    planner = VisualPlanner(args.qwen, "reference-probe", ["probe target"], fixed=False, references=exemplars)
    timing = json.loads((args.timing / "release-timing.json").read_bytes())
    rows = []
    for sample in timing["results"]:
        if sample["phase"] != "after_release_observation":
            continue
        folder = args.timing / Path(sample["artifact"]).name
        if sample["init_state_sha256"] in reference_hashes:
            raise ValueError("a query initial state is shared with the reference bank")
        history, sample_hash = trajectory(folder, sample["init_state_sha256"], sample["step"])
        releases = [
            (release, before) for release, before, after in history.release_windows(rgbd=True)
            if release >= sample["official_step"] - 35
        ]
        if not releases:
            raise ValueError(f"no matching release in {folder}")
        release, before = releases[-1]
        frame = history.grasp_view(release, before)
        crops = held_crops(frame)
        vlm, _ = held_identity(planner, frame)
        row = {
            "artifact": folder.name, "task_id": sample["task_id"], "expected": sample["label"],
            "frame": frame.step, "init_state_sha256": sample["init_state_sha256"], "sample_sha256": sample_hash,
            "grasp_crop_sha256": [digest(crop) for _, crop in crops], "qwen": vlm,
            "dino_static": matcher.classify(crops, held_references=False),
            "dino_held": matcher.classify(crops),
        }
        rows.append(row)
        print(json.dumps({
            "artifact": row["artifact"], "expected": row["expected"],
            "qwen": vlm["label"], "static": row["dino_static"]["label"],
            "held": row["dino_held"]["label"], "margin": row["dino_held"]["margin"],
        }), flush=True)
    scores = {
        key: {
            "correct": sum(row[key]["label"] == row["expected"] for row in rows),
            "abstain": sum(row[key]["label"] is None for row in rows),
            "total": len(rows),
        }
        for key in ("qwen", "dino_static", "dino_held")
    }
    report = {
        "scope": "seen identity regression; no closed-loop or held-out gate",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "model": DINO_ID, "revision": DINO_REVISION, "margin": 0.04,
        "gallery": gallery, "scores": scores, "records": rows,
    }
    (args.output / "reference-identity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "development", "tomato", "references", "timing", "qwen", "dino"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
