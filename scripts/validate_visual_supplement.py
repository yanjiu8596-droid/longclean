"""Score a frozen visual candidate on the remaining untouched official states."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, evaluate, model_pin, summarize
from native_planar_identity import load_logo
from native_rgbd_placement import RGBD_RELEASE
from native_visual_references import load_references
from native_visual_windows import audit
from probe_native_vision import TARGETS
from replay_native_frames import replay_panel, scene_labels


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source_episodes(run, extra, coverage):
    index = run / "results.jsonl"
    if digest(index) != coverage["source_results_sha256"]:
        raise ValueError("original diagnostic index differs from the coverage audit")
    for row in map(json.loads, index.read_bytes().splitlines()):
        if row.get("phase") == "direct_horizon_diagnostics" and row["completed"]:
            name = row.get("artifact", row["run"])
            if Path(name).name != name:
                raise ValueError(f"invalid diagnostic artifact name: {name}")
            yield run / name, row["init_state_sha256"]
    if (extra / "status.txt").read_text(encoding="utf-8").strip() != "completed":
        raise ValueError("supplemental trajectory collection has not completed")
    for path in sorted(extra.glob("scene*/episode.json")):
        yield path.parent, None


def choose_cases(run, extra, coverage):
    scenes = scene_labels()
    untouched = {
        (int(scene), state["init"]): state["sha256"]
        for scene, states in coverage["unseen_official_initial_states"].items() for state in states
    }
    if any(init < 15 or digest_value in coverage["excluded_initial_hashes"]
           for (_, init), digest_value in untouched.items()):
        raise ValueError("the coverage audit includes a reserved or previously used state")
    allowed = set(untouched.values())
    choices = defaultdict(list)
    for folder, indexed_hash in source_episodes(run, extra, coverage):
        if indexed_hash is not None and indexed_hash not in allowed:
            continue
        episode = read(folder / "episode.json")
        scene = int(episode["task_id"].rsplit("-", 1)[1])
        init, initial_hash = episode["init_state_index"], episode["init_state_sha256"]
        if untouched.get((scene, init)) != initial_hash or type(episode["success"]) is not bool:
            raise ValueError(f"trajectory differs from the untouched state manifest: {folder}")
        trace = [json.loads(line) for line in (folder / "actions.jsonl").read_bytes().splitlines()]
        timeline = [(row["start_step"] + row["executed"], audit(row)) for row in trace]
        if not timeline or timeline[-1][0] != episode["steps"]:
            raise ValueError(f"missing terminal action audit: {folder}")
        for label in scenes[scene]:
            target = TARGETS[label]
            first = next((step for step, truth in timeline if truth[target]), None)
            observations = {}
            for step, truth in timeline:
                terminal = step == episode["steps"]
                if step < 50 or (step % 10 and not terminal):
                    continue
                positive = truth[target]
                if positive:
                    rank = (not terminal, step)
                elif first is not None and 0 < first - step <= 20:
                    rank = (0, -step)
                elif any(value for key, value in truth.items() if key != target):
                    rank = (1, -step)
                elif terminal:
                    rank = (2, -step)
                else:
                    continue
                if positive not in observations or rank < observations[positive][0]:
                    observations[positive] = rank, step
            if observations:
                row = {"run": folder.name, "seed": episode["seed"], "init": init, "init_state_sha256": initial_hash}
                choices[scene, init, label].append((folder, row, observations))

    cases, plan, missing = defaultdict(dict), [], []
    for (scene, init), initial_hash in sorted(untouched.items()):
        for label in scenes[scene]:
            options = choices[scene, init, label]
            paired = [item for item in options if len(item[2]) == 2]
            for truth in (False, True):
                group = scene, label, truth
                key = {"scene": scene, "target": label, "expected_in_basket": truth,
                       "init": init, "init_state_sha256": initial_hash}
                eligible = paired or [item for item in options if truth in item[2]]
                if not eligible:
                    missing.append(key)
                    continue
                folder, row, observations = min(eligible, key=lambda item: (
                    item[2][False][0][0] if paired else item[2][truth][0][0],
                    item[2][True][0][0] if paired else 0,
                    item[0].parent == extra, item[0].name,
                ))
                row = {**row, "decision_step": observations[truth][1]}
                cases[group][init] = row, folder
                plan.append({**key, **row, "source": str(folder), "actions_sha256": digest(folder / "actions.jsonl")})
    return cases, plan, missing


def run(args):
    calibration = read(args.calibration)
    coverage = read(args.coverage)
    if calibration.get("status") != "calibrated" or calibration["selected"] != f"{MODEL}/{RGBD_RELEASE}":
        raise ValueError("a frozen RGB-D release candidate is required")
    subprocess.run(["git", "diff", "--quiet", calibration["project_commit"], "HEAD", "--",
                    "scripts/native_temporal_vision.py", "scripts/native_rgbd_placement.py", "scripts/native_visual.py",
                    "scripts/validate_temporal_vision.py", "scripts/run_native_osc.py", "scripts/replay_native_frames.py",
                    "scripts/native_planar_identity.py", "scripts/validate_rgbd_placement.py"], cwd=PROJECT, check=True)
    if model_pin(args.model, MODEL) != calibration["model"]:
        raise ValueError("installed visual model differs from the frozen candidate")
    cases, rows, missing = choose_cases(args.run, args.trajectories, coverage)
    if not rows:
        raise ValueError("no untouched observations are available")
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = args.output / "plan.json"
    plan = {"cases": rows, "missing": missing, "calibration_sha256": digest(args.calibration),
            "coverage_sha256": digest(args.coverage), "collection_plan_sha256": digest(args.trajectories / "plan.tsv")}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    frames = args.output / "frames"
    replay_panel(cases, frames, args.sim_python, args.roboenv, args.workers, rgbd=True, dense_release=True)
    planner = VisualPlanner(args.model, "supplemental-visual", ["probe target"], fixed=False,
                            references=load_references(args.calibration.parent, calibration["references"]))
    logo = load_logo(args.calibration.parent, calibration["logo_templates"])
    results = evaluate(planner, cases, frames, ("rgbd_hybrid", RGBD_RELEASE), "supplemental", logo=logo)
    for trial in results.values():
        trial["by_scene"] = {scene: summarize([row for row in trial["records"] if row["scene"] == scene])
                             for scene in sorted({row["scene"] for row in rows})}
    report = {
        "status": "supplemental_completed", "gate_passed": False,
        "scope": "disjoint official initial states, supplemental coverage below the original holdout requirement",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "candidate_commit": calibration["project_commit"], "model": calibration["model"],
        "plan_sha256": digest(plan_path), "states": len({row["init_state_sha256"] for row in rows}),
        "observations": len(rows), "missing_observations": missing, "methods": results,
    }
    (args.output / "supplemental.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value["score"] for key, value in results.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "trajectories", "coverage", "calibration", "output", "sim-python", "roboenv", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=8)
    run(parser.parse_args())
