"""Compare placement history on development trajectories, then verify one rule once."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from native_temporal_vision import TEMPORAL_STRATEGIES, TemporalVision
from native_visual_references import create_references, load_references
from replay_native_frames import DEVELOPMENT, HOLDOUT, replay_panel, scene_labels
from validate_native_vision import PROJECT, VisualPlanner, model_pin, panel, qualified, summarize


MODEL = "qwen3_vl_4b_instruct"


def trajectory(path: Path, expected_init: str) -> tuple[TemporalVision, str]:
    import numpy as np
    from PIL import Image

    verified = json.loads((path / "verified.json").read_bytes())
    sample_path = path / "frames.npz"
    sample_hash = sha256(sample_path.read_bytes()).hexdigest()
    if verified["init_state_sha256"] != expected_init or verified["frames_sha256"] != sample_hash:
        raise ValueError(f"replay evidence failed verification: {path}")
    history = TemporalVision()
    with np.load(sample_path, allow_pickle=False) as samples:
        commands = samples["gripper_commands"].tolist()
        last = 0
        for step, views, hashes in zip(samples["steps"], samples["images"], samples["image_hashes"], strict=True):
            step = int(step)
            history.observe(step, [Image.fromarray(view) for view in views], commands[last:step], hashes.tolist())
            last = step
    return history, sample_hash


def evaluate(planner, cases, frames_dir, strategies, split):
    records = {strategy: [] for strategy in strategies}
    for (scene, label, expected), samples in sorted(cases.items()):
        for row, folder in samples.values():
            history, sample_hash = trajectory(frames_dir / folder.name, row["init_state_sha256"])
            for strategy in strategies:
                answer = history.inspect(planner, "probe_target", label, strategy)
                prediction = None if answer["status"] == "unknown" else answer["status"] == "in_basket"
                records[strategy].append({
                    "scene": scene, "target": label, "run": row["run"], "artifact": folder.name,
                    "seed": row["seed"], "init": row["init"], "init_state_sha256": row["init_state_sha256"],
                    "sample_sha256": sample_hash, "answer": answer,
                    "expected_in_basket": expected, "predicted_in_basket": prediction,
                })
                print(json.dumps({
                    "split": split, "strategy": strategy, "scene": scene, "init": row["init"],
                    "target": label, "prediction": prediction, "reason": answer["raw"],
                }), flush=True)
    return {strategy: {"score": summarize(rows), "records": rows} for strategy, rows in records.items()}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = create_references(args.run, args.output)
    pin = model_pin(args.model, MODEL)
    report = {
        "source": str(args.run), "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
        ).strip(),
        "calibration_indices": [DEVELOPMENT.start, DEVELOPMENT.stop - 1],
        "holdout_indices": [HOLDOUT.start, HOLDOUT.stop - 1],
        "references": manifest, "status": "preparing_calibration", "gate_passed": False,
    }

    def save():
        temporary = args.output / "visual-validation.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output / "visual-validation.json")

    save()
    scenes = scene_labels()
    calibration = panel(args.run, scenes, DEVELOPMENT, 2)
    replay_panel(calibration, args.frames, args.sim_python, args.roboenv, args.workers)
    planner = VisualPlanner(
        args.model, "temporal-validation", ["probe target"], fixed=False,
        references=load_references(args.output, manifest),
    )
    report["calibration"] = evaluate(planner, calibration, args.frames, TEMPORAL_STRATEGIES, "calibration")
    candidates = {
        key: item["score"]["correct"] for key, item in report["calibration"].items()
        if qualified(item["score"], holdout=False)
    }
    if not candidates:
        report["status"] = "calibration_failed"
        save()
        return
    strategy = max(candidates, key=candidates.get)
    report.update(status="validating_holdout", selected=f"{MODEL}/{strategy}")
    save()
    holdout = panel(args.run, scenes, HOLDOUT, 4)
    heldout_frames = args.output / "holdout-frames"
    replay_panel(holdout, heldout_frames, args.sim_python, args.roboenv, args.workers)
    result = evaluate(planner, holdout, heldout_frames, (strategy,), "holdout")[strategy]
    report["holdout"] = {**result, "model": pin, "strategy": strategy}
    report["gate_passed"] = qualified(result["score"], holdout=True)
    report["status"] = "passed" if report["gate_passed"] else "holdout_failed"
    save()
    print(json.dumps({"status": report["status"], "score": result["score"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "frames", "sim-python", "roboenv", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=8)
    run(parser.parse_args())
