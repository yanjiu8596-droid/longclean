"""Pair formal methods and preserve compact, replay-verified evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

from archive_tableclear_evidence import EPISODE_FILES
from replay_native_video import replay
from run_native_methods import METHODS, jobs_for, read_jsonl


CONDITION_FIELDS = ("init_state_sha256", "bddl_sha256", "model", "step_budget", "chunk_horizon", "supplemental_camera")


def summarize(rows):
    planned = {(scene, seed, init) for scene, seed, init, _ in jobs_for("formal")}
    groups = defaultdict(dict)
    errors = Counter()
    for row in rows:
        if not row["completed"]:
            errors[str(row["error"] or row["terminal_reason"] or row["exit_code"])] += 1
            continue
        condition = row["scene"], row["seed"], row["init"]
        method = row["method"]
        if condition not in planned or method not in METHODS or method in groups[condition]:
            raise ValueError(f"unexpected or duplicate completed condition {condition}/{method}")
        groups[condition][method] = row

    matched, incomplete, mismatched = {}, [], []
    for condition in sorted(planned):
        group = groups[condition]
        if set(group) != set(METHODS):
            incomplete.append({"condition": condition, "missing_methods": sorted(set(METHODS) - set(group))})
        elif len({tuple(row[field] for field in CONDITION_FIELDS) for row in group.values()}) != 1:
            mismatched.append(condition)
        else:
            matched[condition] = group

    def scores(cases):
        metrics = {}
        for method in METHODS:
            results = [case[method] for case in cases]
            successes = [row for row in results if row["success"]]
            metrics[method] = {
                "episodes": len(results), "successes": len(successes),
                "success_rate": len(successes) / len(results) if results else None,
                "median_steps": median(row["steps"] for row in results) if results else None,
                "median_success_steps": median(row["steps"] for row in successes) if successes else None,
                "mean_vla_calls": mean(row["vla_calls"] for row in results) if results else None,
                "mean_visual_calls": mean(row["visual_calls"] for row in results) if results else None,
                "median_elapsed_s": median(row["elapsed_s"] for row in results) if results else None,
                "terminal_reasons": dict(Counter(row["terminal_reason"] for row in results)),
            }
        comparisons = {}
        for method, baseline in ((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])):
            wins = sum(case[method]["success"] and not case[baseline]["success"] for case in cases)
            losses = sum(case[baseline]["success"] and not case[method]["success"] for case in cases)
            step_deltas = [
                case[method]["steps"] - case[baseline]["steps"]
                for case in cases if case[method]["success"] and case[baseline]["success"]
            ]
            comparisons[f"{method}_vs_{baseline}"] = {
                "wins": wins, "losses": losses, "both_succeeded": len(step_deltas),
                "success_delta_pp": 100 * (wins - losses) / len(cases) if cases else None,
                "median_step_delta_both_succeeded": median(step_deltas) if step_deltas else None,
            }
        return {"methods": metrics, "paired_comparisons": comparisons}

    return {
        "paired_conditions": len(matched), "incomplete_pairs": incomplete, "condition_mismatches": mismatched,
        "infrastructure_failures": dict(errors), "overall": scores(list(matched.values())),
        "by_scene": {scene: scores([v for k, v in matched.items() if k[0] == scene]) for scene in sorted({k[0] for k in planned})},
    }


def finalize(args):
    import fcntl

    with (args.run / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((args.run / "summary.json").read_bytes())
        if status["status"] != "completed" or status["active"]:
            raise ValueError("formal scheduler has not completed")
        visual_bytes = args.visual_validation.read_bytes()
        if sha256(visual_bytes).hexdigest() != status["visual_validation_sha256"]:
            raise ValueError("visual validation report does not match the formal run")
        formal = args.run / "formal"
        rows = read_jsonl(formal / "results.jsonl")
        report = summarize(rows)
        if report["paired_conditions"] != len(jobs_for("formal")) // len(METHODS):
            raise ValueError("formal result index does not cover the paired matrix")
        args.output.mkdir(parents=True, exist_ok=False)

        archive = args.output / "v1-formal.zip"
        pending = archive.with_suffix(".pending.zip")
        manifest = {"source": str(args.run), "project_commit": status["project_commit"], "files": {}, "examples": [], "replay_errors": []}
        with ZipFile(pending, "x", compression=ZIP_DEFLATED, compresslevel=4) as packed:
            def add(name, data):
                packed.writestr(name, data)
                manifest["files"][name] = sha256(data).hexdigest()

            add("index/summary.json", (args.run / "summary.json").read_bytes())
            for phase in ("smoke", "pilot", "formal"):
                index = args.run / phase / "results.jsonl"
                add(f"index/{phase}-results.jsonl", index.read_bytes())
                for row in read_jsonl(index):
                    if row["method"] != "closed_loop_vla" or not row["completed"]:
                        continue
                    artifact = row["artifact"]
                    if Path(artifact).name != artifact or artifact in {".", ".."}:
                        raise ValueError(f"invalid artifact directory {artifact}")
                    add(f"visual/{phase}/{artifact}.jsonl", (args.run / phase / artifact / "visual.jsonl").read_bytes())
            add("index/visual-validation.json", visual_bytes)
            add("index/paired-metrics.json", (json.dumps(report, indent=2) + "\n").encode())
            for scene in report["by_scene"]:
                successful = [row for row in rows if row["scene"] == scene and row["completed"] and row["success"]]
                if not successful:
                    continue
                example = min(successful, key=lambda row: (row["steps"], row["run"]))
                artifact = example["artifact"]
                if Path(artifact).name != artifact or artifact in {".", ".."}:
                    raise ValueError(f"invalid artifact directory {artifact}")
                source = formal / artifact
                selected = {field: example[field] for field in ("run", "artifact", "steps", "method", "scene", "seed", "init")}
                manifest["examples"].append(selected)
                camera = example["supplemental_camera"]
                views = (f"initial-{camera}.png", f"final-{camera}.png") if camera else ()
                for name in (*EPISODE_FILES, *views):
                    if (source / name).is_file():
                        add(f"episodes/{artifact}/{name}", (source / name).read_bytes())
                output = args.output / "replays" / artifact
                try:
                    selected["video"] = replay(source, output, args.sim_python, args.roboenv)
                except Exception as exc:
                    manifest["replay_errors"].append({"artifact": artifact, "error": str(exc)})
                else:
                    for name in ("replay.mp4", "replay.json"):
                        add(f"episodes/{artifact}/{name}", (output / name).read_bytes())
            packed.writestr("archive-manifest.json", json.dumps(manifest, indent=2) + "\n")
        pending.replace(archive)
        result = {
            "status": "archive_with_replay_errors" if manifest["replay_errors"] else "completed",
            "paired_conditions": report["paired_conditions"], "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
            "examples": manifest["examples"], "replay_errors": manifest["replay_errors"],
        }
        (args.output / "status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "visual-validation", "sim-python", "roboenv"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    finalize(parser.parse_args())
