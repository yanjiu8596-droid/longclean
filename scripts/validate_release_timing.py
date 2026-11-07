"""Test pre-release safety and delayed confirmation on fresh trajectories."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, model_pin, trajectory
from native_visual_references import load_references
from native_visual_windows import audit
from probe_native_vision import TARGETS
from replay_native_frames import replay


def cases_for(runs, used):
    targets = {value: label for label, value in TARGETS.items()}
    for root in runs:
        for path in sorted(root.glob("scene*/episode.json")):
            episode = json.loads(path.read_bytes())
            key = episode["init_state_sha256"]
            if key in used or not episode["success"] or episode["init_state_index"] < 15:
                continue
            trace = [json.loads(line) for line in (path.parent / "actions.jsonl").read_bytes().splitlines()]
            if len(trace) < 2 or trace[-1]["actions"][-1][-1] >= -0.5:
                continue
            previous = audit(trace[-2])
            entering = [target for target, value in audit(episode).items() if value and not previous[target]]
            if len(entering) != 1:
                continue
            target = entering[0]
            negative = max((
                row["start_step"] + row["executed"] for row in trace[:-1]
                if (row["start_step"] + row["executed"]) % 10 == 0 and not audit(row)[target]
            ), default=0)
            if negative == 0 or episode["steps"] - negative > 30:
                continue
            used.add(key)
            yield {
                "artifact": str(path.parent), "target": target, "label": targets[target],
                "before": negative, "official_step": episode["steps"], "init_state_sha256": key,
                "seed": episode["seed"], "task_id": episode["task_id"], "init": episode["init_state_index"],
            }


def run(args):
    calibration = json.loads(args.calibration.read_bytes())
    prior = json.loads(args.previous.read_bytes())
    used = {
        row["init_state_sha256"]
        for split in ("calibration", "diagnostic") for trial in calibration[split].values() for row in trial["records"]
    }
    used.update(row["init_state_sha256"] for row in prior["holdout"]["records"])
    used.update(json.loads(path.read_bytes())["init_state_sha256"] for path in args.exclude)
    strategy = prior["selected"].split("/", 1)[1]
    pin = model_pin(args.model, MODEL)
    if pin != prior["holdout"]["model"]:
        raise ValueError("visual weights differ from the frozen candidate")
    subprocess.run([
        "git", "diff", "--quiet", prior["project_commit"], "HEAD", "--",
        "scripts/native_temporal_vision.py", "scripts/native_rgbd_placement.py", "scripts/native_visual.py",
    ], cwd=PROJECT, check=True)

    cases = list(cases_for(args.run, used))
    if not cases:
        raise ValueError("no untouched trajectories have a sampled pre-release observation")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")

    def capture(case):
        path = Path(case["artifact"])
        replay(path, args.output / path.name, args.sim_python, args.roboenv, rgbd=True, release_tail=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(capture, cases))
    planner = VisualPlanner(
        args.model, "release-timing", ["probe target"], fixed=False,
        references=load_references(args.previous.parent, prior["references"]),
    )
    results = []
    for case in cases:
        artifact = args.output / Path(case["artifact"]).name
        metadata = json.loads((artifact / "verified.json").read_bytes())
        after = next(record for record in metadata["diagnostic_tail"] if record["offset"] == 10)
        for phase, step, expected in (
            ("before_release", case["before"], False),
            ("after_release_observation", case["official_step"] + 10, audit(after)[case["target"]]),
        ):
            history, hash_value = trajectory(artifact, case["init_state_sha256"], step)
            answer = history.inspect(planner, "probe_target", case["label"], strategy)
            prediction = None if answer["status"] == "unknown" else answer["status"] == "in_basket"
            results.append({
                **case, "phase": phase, "step": step, "expected": expected, "prediction": prediction,
                "gripper_width_m": history.frames[-1].gripper_width, "frames_sha256": hash_value, "answer": answer,
            })
        print(json.dumps({"trajectory": artifact.name, "decisions": [row["prediction"] for row in results[-2:]]}), flush=True)
    counts = {}
    for phase in ("before_release", "after_release_observation"):
        group = [row for row in results if row["phase"] == phase]
        counts[phase] = {
            "cases": len(group), "correct": sum(row["prediction"] is row["expected"] for row in group),
            "false_positive": sum(row["prediction"] is True and not row["expected"] for row in group),
            "false_negative": sum(row["prediction"] is False and row["expected"] for row in group),
            "abstain": sum(row["prediction"] is None for row in group),
        }
    report = {
        "scope": "release timing diagnostic; original official terminal frames and method gate are separate",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "candidate_sha256": sha256(args.previous.read_bytes()).hexdigest(), "model": pin,
        "counts": counts, "results": results,
    }
    (args.output / "release-timing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "sim-python", "roboenv", "model", "calibration", "previous"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 11))
    run(parser.parse_args())
