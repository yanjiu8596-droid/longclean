"""Plan or run the small, paired SmolVLA and Pi0.5 native OSC comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
import tomllib

from native_pi05 import PIN, PROJECT, model_directory, weights_path


POLICIES = ("smolvla", "pi05")
SCENES = (1, 2, 8)
BUDGET = 520
CHUNK = 10


def make_plan() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    with (PROJECT / "versions.lock").open("rb") as stream:
        smolvla = tomllib.load(stream)["models"]["smolvla_libero"]
    return {
        "project_commit": commit, "step_budget": BUDGET, "chunk_horizon": CHUNK,
        "smolvla": smolvla, "pi05": PIN, "conditions": [
            {"scene": scene, "seed": seed, "init": init, "policies": list(POLICIES[order % 2:] + POLICIES[:order % 2])}
            for order, (scene, seed, init) in enumerate(
                (scene, seed, init) for seed, init in enumerate((0, 5, 10)) for scene in SCENES
            )
        ],
    }


def read_episode(output, plan, case, policy):
    path = output / "episode.json"
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (not isinstance(row, dict) or row.get("project_commit") != plan["project_commit"]
            or row.get("step_budget") != BUDGET or row.get("chunk_horizon") != CHUNK
            or row.get("method") != "direct_vla" or row.get("task_id") != f"libero/libero-10-{case['scene']:02d}"
            or row.get("seed") != case["seed"] or row.get("init_state_index") != case["init"]
            or row.get("controller") != "OSC_POSE" or row.get("visual_calls") != 0
            or row.get("terminal_reason") not in ("simulator_success", "simulator_done", "step_budget")
            or not isinstance(row.get("success"), bool) or type(row.get("steps")) is not int
            or not 1 <= row["steps"] <= BUDGET):
        return None
    if policy == "pi05" and (row["model"] != PIN["repository"] or row["model_artifacts"] != [PIN]):
        raise ValueError(f"Unexpected Pi0.5 model provenance in {path}")
    if policy == "smolvla" and row["model"] != f"{plan[policy]['repository']}@{plan[policy]['revision']}":
        raise ValueError(f"Unexpected SmolVLA model provenance in {path}")
    return {**row, "policy": policy, "artifact": output.name}


def run_case(args, plan, case, policy):
    key = f"{case['scene']:02d}-seed{case['seed']}-init{case['init']:02d}-{policy}"
    for suffix in ("", "-retry1"):
        output = args.output / f"{key}{suffix}"
        log_path = output.with_suffix(".log")
        if output.exists() or log_path.exists():
            row = read_episode(output, plan, case, policy)
            if row is not None:
                return row
            continue
        command = [
            sys.executable, "-u", str(PROJECT / "scripts" / "run_native_osc.py"),
            "--workspace", str(args.workspace), "--roboenv", str(args.roboenv), "--sim-python", str(args.sim_python),
            "--output", str(output), "--policy", policy, "--steps", str(BUDGET), "--chunk-steps", str(CHUNK),
            "--scene", str(case["scene"]), "--seed", str(case["seed"]), "--init-index", str(case["init"]),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Episode exited with status {process.returncode}: {log_path}")
        row = read_episode(output, plan, case, policy)
        if row is None:
            raise ValueError(f"Invalid episode evidence: {output}")
        return row
    raise RuntimeError(f"Both attempts failed for {key}; see preserved episode logs")


def summarize(pairs):
    successes = {policy: sum(pair[policy]["success"] for pair in pairs) for policy in POLICIES}
    return {
        "paired_conditions": len(pairs), "successes": successes,
        "pi05_only": sum(pair["pi05"]["success"] and not pair["smolvla"]["success"] for pair in pairs),
        "smolvla_only": sum(pair["smolvla"]["success"] and not pair["pi05"]["success"] for pair in pairs),
        "median_steps": {policy: median(pair[policy]["steps"] for pair in pairs) if pairs else None for policy in POLICIES},
        "terminal_reasons": {policy: Counter(pair[policy]["terminal_reason"] for pair in pairs) for policy in POLICIES},
    }


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args, plan):
    import fcntl
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the comparison; omit --run to inspect the plan")
    weights_path(model_directory(args.workspace), verify=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.output / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_bytes()) != plan:
            raise ValueError("Resume requires the original comparison plan and project commit")
        write_json(plan_path, plan)
        pairs = []
        started = time.monotonic()
        for case in plan["conditions"]:
            pair = {policy: run_case(args, plan, case, policy) for policy in case["policies"]}
            for key in ("init_state_sha256", "bddl_sha256", "instruction", "observation_sha256", "sim_state_sha256"):
                if key not in pair["smolvla"] or key not in pair["pi05"] or pair["smolvla"][key] != pair["pi05"][key]:
                    raise ValueError(f"Unpaired {key} for {case}")
            pairs.append(pair)
            report = summarize(pairs)
            report["status"] = "completed" if len(pairs) == len(plan["conditions"]) else "running"
            report["session_elapsed_s"] = round(time.monotonic() - started, 2)
            report["by_scene"] = {
                str(scene): summarize([row for row in pairs if row["pi05"]["task_id"].endswith(f"-{scene:02d}")])
                for scene in SCENES
            }
            write_json(args.output / "pairs.json", pairs)
            write_json(args.output / "summary.json", report)
            print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    for name in ("workspace", "roboenv", "sim-python", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    plan = make_plan()
    if args.run:
        if not all((args.workspace, args.roboenv, args.sim_python, args.output)):
            parser.error("--run requires --workspace, --roboenv, --sim-python, and --output")
        run(args, plan)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
