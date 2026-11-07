"""Calibrate RGB-D placement decisions before evaluating fresh trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, evaluate, model_pin, panel, qualified, trajectory
from native_rgbd_placement import RGBD_STRATEGIES
from native_visual_references import create_references, load_references
from native_visual_windows import audit
from probe_native_vision import TARGETS
from replay_native_frames import DEVELOPMENT, replay_panel, scene_labels


def calibrate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    references = create_references(args.run, args.output)
    cases = panel(args.run, scene_labels(), DEVELOPMENT, 3)
    replay_panel(cases, args.frames, args.sim_python, args.roboenv, args.workers, rgbd=True)
    planner = VisualPlanner(
        args.model, "rgbd-calibration", ["probe target"], fixed=False,
        references=load_references(args.output, references),
    )
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


def holdout_cases(runs, excluded):
    candidates = defaultdict(dict)
    scenes = scene_labels()
    for run in runs:
        if (run / "status.txt").read_text(encoding="utf-8").strip() != "completed":
            raise ValueError(f"incomplete holdout trajectories: {run}")
        for path in sorted(run.glob("scene*/episode.json")):
            episode = json.loads(path.read_bytes())
            scene = int(episode["task_id"].rsplit("-", 1)[1])
            initial_hash = episode["init_state_sha256"]
            if initial_hash in excluded or type(episode["success"]) is not bool or episode["init_state_index"] < 15:
                continue
            trace = [json.loads(line) for line in (path.parent / "actions.jsonl").read_bytes().splitlines()]
            timeline = [(row["start_step"] + row["executed"], audit(row)) for row in trace]
            for label in scenes[scene]:
                target = TARGETS[label]
                first_positive = next((step for step, truth in timeline if truth[target]), None)
                for step, truth in timeline:
                    if step % 10 and step != episode["steps"] or step < 50:
                        continue
                    positive = truth[target]
                    if positive:
                        priority = 0 if step == episode["steps"] else 1
                    elif first_positive is not None and 0 < first_positive - step <= 20:
                        priority = 0
                    elif any(value for key, value in truth.items() if key != target):
                        priority = 1
                    elif step == episode["steps"]:
                        priority = 2
                    else:
                        continue
                    rank = priority, path.parent.name, -step
                    group = candidates[scene, label, positive]
                    if initial_hash in group and group[initial_hash][0] <= rank:
                        continue
                    group[initial_hash] = rank, {
                        "run": path.parent.name, "seed": episode["seed"], "init": episode["init_state_index"],
                        "init_state_sha256": initial_hash, "decision_step": step,
                    }, path.parent
    selected = {}
    for scene, labels in scenes.items():
        for label in labels:
            for truth in (False, True):
                key = scene, label, truth
                chosen = sorted(candidates[key].values(), key=lambda row: row[0])[:4]
                if len(chosen) != 4:
                    raise ValueError(f"fresh trajectory coverage for {key}: {len(chosen)}/4")
                selected[key] = {index: (row, path) for index, (_, row, path) in enumerate(chosen)}
    return selected


def accept(args):
    calibration_bytes = args.calibration.read_bytes()
    report = json.loads(calibration_bytes)
    if report["status"] != "calibrated":
        raise ValueError("the RGB-D candidate did not pass calibration")
    classifier = [
        "scripts/native_temporal_vision.py", "scripts/native_rgbd_placement.py",
        "scripts/native_visual.py", "scripts/validate_temporal_vision.py",
    ]
    subprocess.run(["git", "diff", "--quiet", report["project_commit"], "HEAD", "--", *classifier], cwd=PROJECT, check=True)
    if model_pin(args.model, MODEL) != report["model"]:
        raise ValueError("visual weights differ from calibration")
    strategy = report["selected"].split("/", 1)[1]
    excluded = {row["init_state_sha256"] for trial in report["calibration"].values() for row in trial["records"]}
    excluded.add(json.loads((args.regression / "verified.json").read_bytes())["init_state_sha256"])
    cases = holdout_cases(args.holdout_run, excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    references = create_references(args.run, args.output)
    if references != report["references"]:
        raise ValueError("visual reference images differ from calibration")
    result = {
        "status": "validating_holdout", "gate_passed": False, "selected": report["selected"], "references": references,
        "calibration_sha256": sha256(calibration_bytes).hexdigest(), "calibration_commit": report["project_commit"],
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "sources": [str(run) for run in args.holdout_run],
        "holdout_cases": [
            {**row, "scene": scene, "target": label, "expected_in_basket": truth, "artifact": path.name}
            for (scene, label, truth), group in cases.items() for row, path in group.values()
        ],
    }
    def save():
        temporary = args.output / "visual-validation.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output / "visual-validation.json")
    save()
    replay_panel(cases, args.frames, args.sim_python, args.roboenv, args.workers, rgbd=True)
    planner = VisualPlanner(args.model, "rgbd-holdout", ["probe target"], fixed=False, references=load_references(args.output, references))
    scored = evaluate(planner, cases, args.frames, (strategy,), "holdout")[strategy]
    result["holdout"] = {**scored, "model": report["model"], "strategy": strategy}
    result["gate_passed"] = qualified(scored["score"], holdout=True)
    result["status"] = "passed" if result["gate_passed"] else "holdout_failed"
    save()
    print(json.dumps({"status": result["status"], "score": scored["score"], "baseline": scored["same_cases_static_score"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "frames", "regression", "model", "sim-python", "roboenv"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=8)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--holdout-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    if bool(args.calibration) != bool(args.holdout_run):
        parser.error("--calibration and --holdout-run must be supplied together")
    (accept if args.calibration else calibrate)(args)
