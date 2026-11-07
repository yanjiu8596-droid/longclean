"""Calibrate RGB-D placement decisions before evaluating fresh trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, evaluate, model_pin, panel, qualified, trajectory
from native_rgbd_placement import RGBD_RELEASE
from native_planar_identity import load_logo
from native_visual_references import create_references, load_references
from native_visual_windows import audit
from probe_native_vision import TARGETS
from replay_native_frames import DEVELOPMENT, replay_panel, scene_labels
from validate_planar_identity import prior_hashes


BASE_STRATEGIES = ("rgbd_placement", "rgbd_hybrid")


def freeze_logo(source, destination, templates):
    if not templates:
        return None
    logo = load_logo(source, templates)
    for row in templates:
        copyfile(source / row["file"], destination / row["file"])
    return logo


def calibrate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    templates = json.loads(args.logo_report.read_bytes())["templates"] if args.logo_report else []
    logo = freeze_logo(args.logo_report.parent, args.output, templates) if templates else None
    references = create_references(args.run, args.output)
    cases = panel(args.run, scene_labels(), DEVELOPMENT, 3)
    strategies = ("rgbd_hybrid", RGBD_RELEASE) if args.dense_release else BASE_STRATEGIES
    replay_panel(cases, args.frames, args.sim_python, args.roboenv, args.workers, rgbd=True,
                 dense_release=args.dense_release)
    planner = VisualPlanner(
        args.model, "rgbd-calibration", ["probe target"], fixed=False,
        references=load_references(args.output, references),
    )
    report = {
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "model": model_pin(args.model, MODEL), "references": references, "gate_passed": False,
        "calibration": evaluate(planner, cases, args.frames, strategies, "calibration", logo=logo), "regression": {},
        "logo_templates": templates,
    }
    if args.diagnostic_report:
        previous = json.loads(args.diagnostic_report.read_bytes())
        groups = defaultdict(dict)
        for row in previous["holdout"]["records"]:
            key = row["scene"], row["target"], row["expected_in_basket"]
            folder = next(Path(source) / row["artifact"] for source in previous["sources"] if (Path(source) / row["artifact"]).is_dir())
            groups[key][len(groups[key])] = row, folder
        report["diagnostic"] = evaluate(planner, groups, args.diagnostic_frames, strategies, "seen-regression", logo=logo)
    metadata = json.loads((args.regression / "verified.json").read_bytes())
    for strategy in strategies:
        planner.held_logo = logo if strategy == RGBD_RELEASE else None
        answers = []
        for step, expected in ((130, False), (140, True), (150, True)):
            history, sample_hash = trajectory(args.regression, metadata["init_state_sha256"], step)
            answer = history.inspect(planner, "probe_target", "cream cheese box", strategy)
            answers.append({"step": step, "expected_in_basket": expected, "sample_sha256": sample_hash, "answer": answer})
        report["regression"][strategy] = answers
    candidates = [
        strategy for strategy, trial in report["calibration"].items()
        if (not args.dense_release or strategy == RGBD_RELEASE)
        and qualified(trial["score"], holdout=False)
        and (
            not args.diagnostic_report or
            report["diagnostic"][strategy]["score"]["correct"] >= 40
            and report["diagnostic"][strategy]["score"]["false_positive"] == 0
        )
        and report["regression"][strategy][0]["answer"]["status"] != "in_basket"
        and report["regression"][strategy][-1]["answer"]["status"] == "in_basket"
    ]
    report["status"] = "calibrated" if candidates else "calibration_failed"
    if candidates:
        selected = max(candidates, key=lambda key: (
            report.get("diagnostic", report["calibration"])[key]["score"]["correct"],
            report["calibration"][key]["score"]["correct"],
        ))
        report["selected"] = f"{MODEL}/{selected}"
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": report.get("selected"),
        "calibration": {key: trial["score"] for key, trial in report["calibration"].items()},
        "seen_regression": {key: trial["score"] for key, trial in report.get("diagnostic", {}).items()},
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
    strategy = report["selected"].split("/", 1)[1]
    dense = strategy == RGBD_RELEASE
    if dense:
        classifier.extend(("scripts/run_native_osc.py", "scripts/replay_native_frames.py", "scripts/native_planar_identity.py",
                           "scripts/validate_rgbd_placement.py"))
    subprocess.run(["git", "diff", "--quiet", report["project_commit"], "HEAD", "--", *classifier], cwd=PROJECT, check=True)
    if model_pin(args.model, MODEL) != report["model"]:
        raise ValueError("visual weights differ from calibration")
    excluded = {row["init_state_sha256"] for trial in report["calibration"].values() for row in trial["records"]}
    excluded.update(row["init_state_sha256"] for trial in report.get("diagnostic", {}).values() for row in trial["records"])
    excluded.add(json.loads((args.regression / "verified.json").read_bytes())["init_state_sha256"])
    templates = report.get("logo_templates", [])
    excluded.update(row["source_initial_hash"] for row in templates)
    for previous in args.exclude_report:
        excluded.update(prior_hashes(json.loads(previous.read_bytes())))
    cases = holdout_cases(args.holdout_run, excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    logo = freeze_logo(args.calibration.parent, args.output, templates)
    references = create_references(args.run, args.output)
    if json.dumps(references, sort_keys=True) != json.dumps(report["references"], sort_keys=True):
        raise ValueError("visual reference images differ from calibration")
    result = {
        "status": "validating_holdout", "gate_passed": False, "selected": report["selected"], "references": references,
        "logo_templates": templates,
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
    replay_panel(cases, args.frames, args.sim_python, args.roboenv, args.workers, rgbd=True, dense_release=dense)
    planner = VisualPlanner(args.model, "rgbd-holdout", ["probe target"], fixed=False, references=load_references(args.output, references))
    strategies = ("rgbd_hybrid", strategy) if dense else (strategy,)
    paired = evaluate(planner, cases, args.frames, strategies, "holdout", logo=logo)
    scored = paired[strategy]
    if dense:
        result["same_cases_rgbd_hybrid"] = paired["rgbd_hybrid"]
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
    parser.add_argument("--diagnostic-report", type=Path)
    parser.add_argument("--diagnostic-frames", type=Path)
    parser.add_argument("--dense-release", action="store_true")
    parser.add_argument("--logo-report", type=Path)
    parser.add_argument("--exclude-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if bool(args.calibration) != bool(args.holdout_run):
        parser.error("--calibration and --holdout-run must be supplied together")
    if bool(args.diagnostic_report) != bool(args.diagnostic_frames):
        parser.error("--diagnostic-report and --diagnostic-frames must be supplied together")
    if args.logo_report and (args.calibration or not args.dense_release):
        parser.error("--logo-report applies only to dense release calibration")
    (accept if args.calibration else calibrate)(args)
