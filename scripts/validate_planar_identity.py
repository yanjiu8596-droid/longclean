"""Compare a frozen logo verifier and the visual baseline on new replayed grasps."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from validate_temporal_vision import MODEL, PROJECT, VisualPlanner, model_pin, trajectory
from native_planar_identity import CREAM_LABEL, PlanarLogo
from native_rgbd_placement import held_identity
from native_visual_references import load_references
from native_visual_windows import audit
from probe_native_vision import TARGETS


LABELS = ("alphabet soup can", CREAM_LABEL)


def read(path):
    return json.loads(path.read_bytes())


def prior_hashes(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"init_state_sha256", "source_initial_hash"}:
                yield item
            else:
                yield from prior_hashes(item)
    if isinstance(value, list):
        for item in value:
            yield from prior_hashes(item)


def paired_cases(args, used):
    cases, omitted = [], []
    for replay in sorted(args.frames.glob("*/verified.json")):
        source = args.run / replay.parent.name
        episode = read(source / "episode.json")
        key = episode["init_state_sha256"]
        if key in used or not episode["success"] or set(audit(episode)) != {TARGETS[label] for label in LABELS}:
            raise ValueError(f"expected a distinct, successful two-target scene 08 trajectory: {source}")
        used.add(key)
        history, frames_hash = trajectory(replay.parent, key, episode["steps"])
        trace = [json.loads(line) for line in (source / "actions.jsonl").read_bytes().splitlines()]
        checkpoints = [(row["start_step"] + row["executed"], audit(row)) for row in trace]
        for label in LABELS:
            target = TARGETS[label]
            base = {"artifact": source.name, "init_state_sha256": key, "frames_sha256": frames_hash, "expected": label}
            last_false = max(step for step, truth in checkpoints if not truth[target])
            first_true = next(step for step, truth in checkpoints if step > last_false and truth[target])
            # A release can precede the final negative checkpoint while the object is settling.
            releases = [window for window in history.release_windows(rgbd=True) if last_false - 6 <= window[0] <= first_true]
            if len(releases) != 1 or not any(frame.step == first_true for frame in history.frames):
                omitted.append({**base, "last_false": last_false, "first_true": first_true, "releases": len(releases)})
                continue
            released, before, opened = releases[0]
            grasp = history.grasp_view(released, before)
            cases.append({**base, "last_false": last_false, "first_true": first_true, "release": released,
                          "measured_open": opened.step, "grasp": grasp.step, "grasp_image_sha256": grasp.hashes})
    return cases, omitted


def run(args):
    from PIL import Image

    report = read(args.templates)
    excluded = [args.templates, args.references, *args.exclude]
    used = {key for path in excluded for key in prior_hashes(read(path))}
    cases, omitted = paired_cases(args, used)
    if not cases:
        raise ValueError("no matched fresh trajectories")
    images = []
    for row in report["templates"]:
        path = args.templates.parent / row["file"]
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"logo template failed verification: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    verifier = PlanarLogo(images)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"cases": cases, "omitted": omitted, "inputs": {path.name: sha256(path.read_bytes()).hexdigest() for path in excluded}}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    planner = VisualPlanner(args.model, "logo-independent", ["probe target"], fixed=False,
                            references=load_references(args.references.parent, read(args.references)["references"]))
    results = []
    for case in cases:
        artifact = args.frames / case["artifact"]
        history, _ = trajectory(artifact, case["init_state_sha256"], case["grasp"])
        frame = history.frames[-1]
        planner.held_logo = None
        baseline, baseline_calls = held_identity(planner, frame)
        planner.held_logo = verifier
        candidate, candidate_calls = held_identity(planner, frame)

        # Use separate histories so the second method cannot reuse the first one's cached decisions.
        placement = {}
        for method, logo in (("qwen", None), ("logo_and_qwen", verifier)):
            placement_history, _ = trajectory(artifact, case["init_state_sha256"], case["first_true"])
            planner.held_logo = logo
            placement[method] = placement_history.inspect(planner, "probe_target", case["expected"], "rgbd_hybrid")
        results.append({**case, "qwen": baseline, "logo_and_qwen": candidate,
                        "identity_model_calls": [baseline_calls, candidate_calls], "placement": placement})
        print(json.dumps({"artifact": case["artifact"], "label": case["expected"], "qwen": baseline["label"],
                          "logo_and_qwen": candidate["label"],
                          "placement": {method: value["status"] for method, value in placement.items()}}), flush=True)

    scores = {method: {
        "identity_correct": sum(row[method]["label"] == row["expected"] for row in results),
        "identity_abstained": sum(row[method]["label"] is None for row in results),
        "confirmed_official_positive": sum(row["placement"][method]["status"] == "in_basket" for row in results),
        "cream_confusions_on_soup": sum(row["expected"] != CREAM_LABEL and row[method]["label"] == CREAM_LABEL for row in results),
    } for method in ("qwen", "logo_and_qwen")}
    result = {
        "scope": "new successful scene 08 trajectories; placement checks are positives only; full gate not evaluated",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "model": model_pin(args.model, MODEL), "opencv": verifier.cv.__version__, "scores": scores,
        "cases": len(results), "omitted": omitted, "results": results,
    }
    (args.output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "frames", "run", "model", "templates", "references"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    run(parser.parse_args())
