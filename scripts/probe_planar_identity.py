"""Check a depth-gated logo verifier on the existing grasp and release regressions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import PROJECT, VisualPlanner, trajectory
from native_planar_identity import LOGO_BOUNDS, PlanarLogo
from native_rgbd_placement import held_identity
from native_visual_references import load_references


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.development / "verified.json").read_bytes())
    source = json.loads(args.references.read_bytes())
    exemplars = load_references(args.references.parent, source["references"])
    images = []
    template_index = []
    for step in (90, 100, 120):
        history, sample_hash = trajectory(args.development, metadata["init_state_sha256"], step)
        template = history.frames[-1].images[1].crop(LOGO_BOUNDS)
        path = args.output / f"cream-logo-{step}.png"
        template.save(path)
        images.append(template)
        template_index.append({
            "file": path.name, "sha256": sha256(path.read_bytes()).hexdigest(), "step": step,
            "source_initial_hash": metadata["init_state_sha256"], "source_frames_sha256": sample_hash,
            "bounds": LOGO_BOUNDS,
        })
    planner = VisualPlanner(args.qwen, "logo-probe", ["probe target"], fixed=False, references=exemplars)
    planner.held_logo = PlanarLogo(images)
    identity = []
    old_identity = json.loads(args.identity.read_bytes())
    for row in old_identity["records"]:
        if row["init_state_sha256"] == metadata["init_state_sha256"]:
            raise ValueError("a query initial state matches the reference source")
        history, _ = trajectory(args.timing / row["artifact"], row["init_state_sha256"], row["frame"])
        prediction, _ = held_identity(planner, history.frames[-1])
        identity.append({**row, "sift_and_qwen": prediction})
        print(json.dumps({
            "artifact": row["artifact"], "expected": row["expected"],
            "qwen": row["qwen"]["label"], "sift_and_qwen": prediction["label"],
            "evidence": prediction.get("local_identity"),
        }), flush=True)

    timing = []
    for row in json.loads((args.timing / "release-timing.json").read_bytes())["results"]:
        folder = args.timing / Path(row["artifact"]).name
        history, _ = trajectory(folder, row["init_state_sha256"], row["step"])
        answer = history.inspect(planner, "probe_target", row["label"], "rgbd_hybrid")
        prediction = None if answer["status"] == "unknown" else answer["status"] == "in_basket"
        timing.append({**row, "sift_and_qwen_prediction": prediction, "sift_and_qwen_answer": answer})
    scores = {"identity": {
        "cases": len(identity), "qwen": sum(row["qwen"]["label"] == row["expected"] for row in identity),
        "sift_and_qwen": sum(row["sift_and_qwen"]["label"] == row["expected"] for row in identity),
    }}
    for phase in ("before_release", "after_release_observation"):
        group = [row for row in timing if row["phase"] == phase]
        scores[phase] = {
            "cases": len(group), "baseline_correct": sum(row["prediction"] is row["expected"] for row in group),
            "correct": sum(row["sift_and_qwen_prediction"] is row["expected"] for row in group),
            "false_positive": sum(row["sift_and_qwen_prediction"] is True and not row["expected"] for row in group),
        }
    report = {
        "scope": "seen regression, with optional OpenCV logo verifier; formal gate not evaluated",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "opencv_version": planner.held_logo.cv.__version__, "templates": template_index,
        "scores": scores, "identity": identity, "timing": timing,
    }
    (args.output / "planar-identity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "development", "references", "timing", "qwen", "identity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
