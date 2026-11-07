"""Calibrate RGB-D placement decisions before evaluating fresh trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, evaluate, model_pin, panel, qualified, trajectory
from native_rgbd_placement import RGBD_STRATEGIES
from native_visual_references import create_references, load_references
from replay_native_frames import DEVELOPMENT, scene_labels


def calibrate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    references = create_references(args.run, args.output)
    planner = VisualPlanner(
        args.model, "rgbd-calibration", ["probe target"], fixed=False,
        references=load_references(args.output, references),
    )
    cases = panel(args.run, scene_labels(), DEVELOPMENT, 3)
    report = {
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "model": model_pin(args.model, MODEL), "references": references, "gate_passed": False,
        "calibration": evaluate(planner, cases, args.frames, RGBD_STRATEGIES, "calibration"), "regression": {},
    }
    metadata = json.loads((args.regression / "verified.json").read_bytes())
    for strategy in RGBD_STRATEGIES:
        answers = []
        for step, expected in ((130, False), (140, True), (150, True)):
            history, sample_hash = trajectory(args.regression, metadata["init_state_sha256"], step)
            answer = history.inspect(planner, "probe_target", "cream cheese box", strategy)
            answers.append({"step": step, "expected_in_basket": expected, "sample_sha256": sample_hash, "answer": answer})
        report["regression"][strategy] = answers
    candidates = [
        strategy for strategy, trial in report["calibration"].items()
        if qualified(trial["score"], holdout=False)
        and report["regression"][strategy][0]["answer"]["status"] != "in_basket"
        and report["regression"][strategy][-1]["answer"]["status"] == "in_basket"
    ]
    report["status"] = "calibrated" if candidates else "calibration_failed"
    if candidates:
        report["selected"] = f"{MODEL}/{max(candidates, key=lambda key: report['calibration'][key]['score']['correct'])}"
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": report.get("selected"),
        "calibration": {key: trial["score"] for key, trial in report["calibration"].items()},
        "regression": {key: [(row["step"], row["answer"]["status"]) for row in rows] for key, rows in report["regression"].items()},
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "frames", "regression", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    calibrate(parser.parse_args())
