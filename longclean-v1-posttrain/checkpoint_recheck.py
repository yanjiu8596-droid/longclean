"""Re-pair Scene 08 checkpoints on one replacement GPU using the original challenge seeds."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys

from checkpoint_control import PAIR_METRICS, gpu_snapshot, wait_until_idle
from common import Lab, file_sha, read_json, write_json
from score import audit_episode, paired, per_arm


STEPS = (400, 600)
ARMS = ("natural", "corrected")


def freeze(inputs, out, lab, uuid):
    from evaluate import attempt_regular

    original_path = inputs / "test-plan.json"
    original = read_json(original_path)
    records = []
    lab.add_imports()
    for case in original["cases"]:
        if case["cohort"] != "challenge":
            continue
        layout = case["layout_seed"]
        target = out / "reset" / str(layout)
        target.mkdir(parents=True, exist_ok=True)
        result = attempt_regular(lab, target, layout)
        state = result["initial"]
        if result["goal_at_reset"] or state["bddl_sha256"] != case["initial"]["bddl_sha256"]:
            raise ValueError(f"Replacement setup altered the task or started at goal for {layout}")
        records.append({"original_initial": case["initial"], "new_case": {**case, "initial": state}})
        print("RESET", layout, {key: value == case["initial"][key] for key, value in state.items()}, flush=True)
    if len(records) != len({record["new_case"]["initial"]["raw_sha256"] for record in records}):
        raise ValueError("Replacement setup produced duplicate initial states")
    plan = {"schema": "v1-scene08-matched-steps-recheck/1", "challenge_resets": records,
            "cases": [record["new_case"] for record in records], "gpu_uuid": uuid,
            "source_plan_sha256": file_sha(original_path), "source_gpu_uuid": original["gpu_uuid"]}
    path = out / "recheck-plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Existing replacement setup differs from the new initial states")
    write_json(path, plan)


def configure(inputs, out, plan):
    for step in STEPS:
        folder = out / f"step-{step:04d}"
        for arm in ARMS:
            original = inputs / "train" / arm
            source = original / f"step-{step:04d}"
            state = read_json(source / "state.json")
            chosen = read_json(original / "selected.json")
            if (state["arm"], state["step"]) != (arm, step) or any(
                state[key] != chosen[key] for key in ("data_sha256", "frozen_sha256")
            ):
                raise ValueError(f"Replacement checkpoint has a different assignment: {source}")
            adapter = Path(f"step-{step:04d}/adapter")
            target = folder / "train" / arm / adapter
            target.mkdir(parents=True, exist_ok=True)
            for file in ("adapter_model.safetensors", "adapter_config.json"):
                if not (target / file).exists():
                    copyfile(source / "adapter" / file, target / file)
                if file_sha(target / file) != file_sha(source / "adapter" / file):
                    raise ValueError(f"Copied adapter mismatch: {target / file}")
            if file_sha(target / "adapter_model.safetensors") != state["adapter_sha256"]:
                raise ValueError(f"Replacement adapter SHA-256 differs: {target}")
            write_json(folder / "train" / arm / "selected.json",
                       {**state, "selected_path": adapter.as_posix()})
        write_json(folder / "test-plan.json", {**plan, "training_step": step})


def summarize(out):
    plan = read_json(out / "recheck-plan.json")
    cases = plan["cases"]
    results = {}
    for step in STEPS:
        folder = out / f"step-{step:04d}"
        path = folder / "test-plan.json"
        if read_json(path) != {**plan, "training_step": step}:
            raise ValueError(f"Replacement plan mismatch at {step}")
        digest = file_sha(path)
        adapters = {arm: read_json(folder / "train" / arm / "selected.json")["adapter_sha256"]
                    for arm in ARMS}
        pairs = [{arm: audit_episode(folder, case, seed, arm, digest, plan["gpu_uuid"], adapters)
                  for arm in ARMS} for case in cases for seed in case["inference_seeds"]]
        results[str(step)] = {
            "arms": {arm: per_arm([pair[arm] for pair in pairs]) for arm in ARMS},
            "paired_corrected_vs_natural": {metric: paired(pairs, "corrected", "natural", metric)
                                          for metric in PAIR_METRICS},
            "pairs": pairs,
        }
    return {"episodes": len(ARMS) * sum(len(data["pairs"]) for data in results.values()),
            "challenge_layouts": len(cases), "test_plan_sha256": file_sha(out / "recheck-plan.json"),
            "gpu_uuid": plan["gpu_uuid"], "matched_steps": results}


def run_experiment(inputs, out, lab, source, uuid):
    for line in (inputs / "source.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if file_sha(source / filename) != digest:
            raise ValueError(f"Evaluation source differs: {filename}")
    (out / "logs").mkdir(parents=True, exist_ok=True)

    def stage(label, *command):
        before = wait_until_idle(uuid)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "MUJOCO_EGL_DEVICE_ID": str(before["index"]),
               "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
        with (out / "logs" / f"{label}.log").open("a", encoding="utf-8") as log:
            print("GPU_BEFORE", uuid, json.dumps(before), file=log, flush=True)
            try:
                subprocess.run(["timeout", "--kill-after=30s", "25m", sys.executable, "-B", "-u", *command],
                               cwd=out, env=env, stdout=log, stderr=log, check=True)
            finally:
                print("GPU_AFTER", uuid, json.dumps(gpu_snapshot(uuid)), file=log, flush=True)

    print("FREEZING_REPLACEMENT_RESETS", uuid, flush=True)
    if not (out / "recheck-plan.json").exists():
        stage("reset", __file__, "freeze", "--inputs", str(inputs), "--out", str(out),
              "--lab", str(lab.root), "--gpu-uuid", uuid)
    plan = read_json(out / "recheck-plan.json")
    if plan["gpu_uuid"] != uuid or plan["source_plan_sha256"] != file_sha(inputs / "test-plan.json"):
        raise ValueError("Replacement plan uses a different GPU or source")
    configure(inputs, out, plan)
    for step in STEPS:
        for arm in ARMS:
            print("RECHECK", arm, step, flush=True)
            stage(f"{arm}-{step}", str(source / "evaluate.py"), arm, "--lab", str(lab.root),
                  "--run", str(out / f"step-{step:04d}"), "--gpu-uuid", uuid)
    result = summarize(out)
    write_json(out / "summary.json", result)
    print("REPLACEMENT_CHECK_COMPLETE", result["episodes"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    for name in ("inputs", "source", "lab"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "score":
        print(json.dumps(summarize(out), indent=2))
        return
    if not (args.lab and args.inputs and args.gpu_uuid and args.gpu_uuid.startswith("GPU-")):
        parser.error("Specify the personal lab, verified inputs and replacement GPU UUID")
    lab = Lab(args.lab.resolve())
    inputs = args.inputs.resolve()
    if not out.is_relative_to(lab.root) or not inputs.is_relative_to(lab.root):
        parser.error("Keep the study under the personal lab directory")
    if args.stage == "freeze":
        freeze(inputs, out, lab, args.gpu_uuid)
        return
    if args.source is None:
        parser.error("--source must point to the frozen primary source")
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_experiment(inputs, out, lab, args.source.resolve(), args.gpu_uuid)


if __name__ == "__main__":
    main()
