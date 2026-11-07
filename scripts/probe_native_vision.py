"""Audit the local visual model on simulator-labeled basket examples."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from native_visual import VisualPlanner
from run_overnight_native import TARGETS
from tableclear.state import TaskState


def cases_for(run, scenes):
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = {}
    for row in sorted(rows, key=lambda item: item["run"]):
        if row["phase"] != "direct_horizon_diagnostics" or not row["completed"]:
            continue
        scene = row["scene"]
        name = row.get("artifact", row["run"])
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid artifact directory {name}")
        folder = run / name
        episode = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        audit = {goal["predicate"][1]: goal["satisfied"] for goal in episode["audit_goals"]}
        for label in scenes[scene]:
            expected = audit[TARGETS[label]]
            selected.setdefault((scene, label, expected), (row, folder))
        if len(selected) == 2 * sum(map(len, scenes.values())):
            break
    return selected


def probe(args):
    import numpy as np
    from PIL import Image
    import torch

    with (PROJECT / "configs" / "v1_eval.toml").open("rb") as config:
        scenes = {int(scene["source_task"].rsplit("-", 1)[1]): scene["objects"] for scene in tomllib.load(config)["scenes"]}
    panel = cases_for(args.run, scenes)
    args.output.mkdir(parents=True, exist_ok=False)
    details = []
    planner = None
    for scene, labels in scenes.items():
        if planner is None:
            planner = VisualPlanner(args.model, f"libero/libero-10-{scene:02d}", labels, fixed=False)
        else:
            planner.controller.state = TaskState.from_labels(f"libero/libero-10-{scene:02d}", labels)
        target_ids = {obj.label: obj.id for obj in planner.controller.state.objects}
        for (case_scene, label, expected), (row, folder) in panel.items():
            if case_scene != scene:
                continue
            paths = [folder / f"final-camera{i}.png" for i in (1, 2)]
            images = []
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            observation = {
                f"observation.images.camera{i}": torch.from_numpy(np.array(image)).permute(2, 0, 1)
                .unsqueeze(0).float().div(255)
                for i, image in enumerate(images, 1)
            }
            result = planner.inspect(observation, target_ids[label])
            baseline = None if result["status"] == "unknown" else result["status"] == "in_basket"
            question = (
                f"Is the {label} inside the woven basket in this robot scene? "
                "Answer with one digit: 1 yes, 2 no, 3 cannot see enough to tell."
            )
            outputs = [("original_dual_four_way", result["raw"], baseline)]
            for variant, views in (("binary_dual", images), ("binary_head", images[:1])):
                raw = planner.answer(views, question)
                outputs.append((variant, raw, {"1": True, "2": False}.get(raw.strip(". \n\t"))))
            for variant, raw, prediction in outputs:
                details.append({
                    "scene": scene, "target": label, "run": row["run"], "artifact": folder.name,
                    "image_sha256": [sha256(path.read_bytes()).hexdigest() for path in paths],
                    "variant": variant, "raw": raw, "expected_in_basket": expected, "predicted_in_basket": prediction,
                })

    scores = {}
    for row in details:
        count = scores.setdefault(row["variant"], Counter())
        count["cases"] += 1
        expected, prediction = row["expected_in_basket"], row["predicted_in_basket"]
        if prediction is None:
            count["abstain"] += 1
        elif prediction == expected:
            count["correct"] += 1
        else:
            count["false_positive" if prediction else "false_negative"] += 1
    report = {"source": str(args.run), "model_path": str(args.model), "project_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
    ).strip(), "labeled_cases": len(panel), "scores": scores, "records": details}
    (args.output / "visual-probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("labeled_cases", "scores")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    probe(parser.parse_args())
