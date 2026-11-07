"""Compare visual providers on calibration images, then score one on a sealed split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from probe_native_vision import PROJECT, TARGETS, VisualPlanner


def panel(run: Path, scenes: dict, indices: range, count: int) -> dict:
    selected = defaultdict(dict)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in sorted(rows, key=lambda r: (r["init"], r["seed"], r["run"])):
        if row["phase"] != "direct_horizon_diagnostics" or not row["completed"] or row["init"] not in indices:
            continue
        scene = row["scene"]
        name = row.get("artifact", row["run"])
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid artifact directory {name}")
        folder = run / name
        episode = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        audit = {goal["predicate"][1]: goal["satisfied"] for goal in episode["audit_goals"]}
        for label in scenes[scene]:
            truth = audit[TARGETS[label]]
            group = selected[scene, label, truth]
            if len(group) < count:
                group.setdefault(row["init"], (row, folder))
        if len(selected) == 2 * sum(map(len, scenes.values())) and all(len(group) == count for group in selected.values()):
            return selected
    raise ValueError(f"insufficient independent samples in indices {indices.start}–{indices.stop - 1}")


def evaluate(planner, cases: dict, strategy: str) -> tuple[dict, list]:
    from PIL import Image

    planner.strategy = strategy
    totals = Counter(cases=0, correct=0, abstain=0, false_positive=0, false_negative=0, unparsed=0)
    evidence = []
    for (scene, label, expected), samples in sorted(cases.items()):
        for row, folder in samples.values():
            paths = [folder / f"final-camera{i}.png" for i in (1, 2)]
            images = []
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            answer = planner.inspect_images(images, "probe_target", label)
            prediction = None if answer["status"] == "unknown" else answer["status"] == "in_basket"
            totals["cases"] += 1
            totals["unparsed"] += not answer["parsed"]
            if prediction is None:
                totals["abstain"] += 1
            elif prediction == expected:
                totals["correct"] += 1
            else:
                totals["false_positive" if prediction else "false_negative"] += 1
            evidence.append({
                "scene": scene, "target": label, "run": row["run"], "artifact": folder.name,
                "seed": row["seed"], "init": row["init"], "init_state_sha256": row["init_state_sha256"],
                "image_sha256": [sha256(path.read_bytes()).hexdigest() for path in paths],
                "answer": answer, "expected_in_basket": expected, "predicted_in_basket": prediction,
            })
    return dict(totals), evidence


def qualified(score: dict, *, holdout: bool) -> bool:
    minimum_accuracy = 0.90 if holdout else 0.75
    return (
        score["false_positive"] == 0 and score["unparsed"] == 0
        and score["correct"] >= minimum_accuracy * score["cases"]
    )


def model_pin(model: Path, identifier: str) -> dict:
    with (PROJECT / "versions.lock").open("rb") as handle:
        pinned = tomllib.load(handle)["models"][identifier]
    revision_file = model / "model-revision.txt"
    if revision_file.exists() and revision_file.read_text(encoding="utf-8").strip() != pinned["revision"]:
        raise ValueError(f"weight revision disagrees with versions.lock: {model}")
    return {"repository": pinned["repository"], "revision": pinned["revision"]}


def run(args) -> None:
    import torch

    with (PROJECT / "configs" / "v1_eval.toml").open("rb") as handle:
        scenes = {int(s["source_task"].rsplit("-", 1)[1]): s["objects"] for s in tomllib.load(handle)["scenes"]}
    args.output.mkdir(parents=True, exist_ok=False)
    calibration = panel(args.run, scenes, range(15, 25), 2)
    report = {
        "source": str(args.run), "calibration_indices": [15, 24], "holdout_indices": [25, 49],
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "calibration": {}, "gate_passed": False,
    }
    if args.grounded:
        models = [("qwen2_5_vl_3b_instruct", args.qwen, ("appearance", "basket_zoom"))]
    else:
        models = [("smolvlm2_500m_video_instruct", args.smol, ("original", "external", "dual"))]
        if args.qwen is not None:
            models.append(("qwen2_5_vl_3b_instruct", args.qwen, ("external", "dual")))

    def save():
        path = args.output / "visual-validation.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)

    candidates = {}
    for model_id, path, strategies in models:
        pin = model_pin(path, model_id)
        planner = VisualPlanner(path, "vision-validation", ["probe target"], fixed=False)
        for strategy in strategies:
            key = f"{model_id}/{strategy}"
            score, records = evaluate(planner, calibration, strategy)
            report["calibration"][key] = {"model": pin, "strategy": strategy, "score": score, "records": records}
            print(json.dumps({"split": "calibration", "candidate": key, "score": score}), flush=True)
            save()
            if qualified(score, holdout=False):
                candidates[key] = (score["correct"], model_id, path, strategy)
        del planner
        torch.cuda.empty_cache()
    if not candidates:
        report["status"] = "calibration_failed"
        save()
        return
    winner = max(candidates, key=lambda key: candidates[key][0])
    _, model_id, path, strategy = candidates[winner]
    report.update(status="validating_holdout", selected=winner)
    save()
    holdout = panel(args.run, scenes, range(25, 50), 4)
    planner = VisualPlanner(path, "vision-validation", ["probe target"], fixed=False)
    score, records = evaluate(planner, holdout, strategy)
    report["holdout"] = {"model": model_pin(path, model_id), "strategy": strategy, "score": score, "records": records}
    report["gate_passed"] = qualified(score, holdout=True)
    report["status"] = "passed" if report["gate_passed"] else "holdout_failed"
    print(json.dumps({"split": "holdout", "candidate": winner, "score": score, "passed": report["gate_passed"]}), flush=True)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "smol", "qwen"):
        parser.add_argument(f"--{name}", type=Path, required=name != "qwen")
    parser.add_argument("--grounded", action="store_true")
    args = parser.parse_args()
    if args.grounded and args.qwen is None:
        parser.error("--grounded requires --qwen")
    run(args)
