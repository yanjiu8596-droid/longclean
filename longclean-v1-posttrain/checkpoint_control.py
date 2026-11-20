"""Compare the two Scene 08 LoRAs at matched training steps on the frozen challenge cohort."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from time import monotonic, sleep
from xml.etree import ElementTree

from common import file_sha, read_json, write_json
from score import audit_episode, paired, per_arm


ADDED_STEPS = {"natural": 400, "corrected": 600}
SELECTED_STEPS = {"natural": 600, "corrected": 400}
PAIR_METRICS = ("success", "release")


def plan_for(primary, checkpoints):
    original = primary / "test-plan.json"
    plan = read_json(original)
    cases = [case for case in plan["cases"] if case["cohort"] == "challenge"]
    if not cases or any(len(case["inference_seeds"]) != 1 for case in cases):
        raise ValueError("Expected the complete single-seed challenge cohort")
    return {
        "schema": "v1-scene08-matched-training-steps/1",
        "cases": cases,
        "gpu_uuid": plan["gpu_uuid"],
        "rule_sha256": plan["rule_sha256"],
        "primary_plan_sha256": file_sha(original),
        "primary_summary_sha256": file_sha(primary / "summary.json"),
        "checkpoints": checkpoints,
    }


def prepare(primary, run):
    checkpoints = {}
    for arm, step in ADDED_STEPS.items():
        trained = primary / "train" / arm
        selected = read_json(trained / "selected.json")
        source = trained / f"step-{step:04d}"
        state = read_json(source / "state.json")
        if selected["step"] != SELECTED_STEPS[arm] or (state["arm"], state["step"]) != (arm, step):
            raise ValueError(f"Unexpected training assignment for {arm}")
        if any(state[key] != selected[key] for key in ("data_sha256", "frozen_sha256")):
            raise ValueError(f"Checkpoint data or frozen base differs for {arm}")
        config = source / "adapter/adapter_config.json"
        model = source / "adapter/adapter_model.safetensors"
        if file_sha(model) != state["adapter_sha256"]:
            raise ValueError(f"Checkpoint adapter checksum differs for {arm}")
        checkpoints[arm] = {**state, "adapter_config_sha256": file_sha(config),
                            "selected_path": f"step-{step:04d}/adapter"}
    path = run / "test-plan.json"
    plan = plan_for(primary, checkpoints)
    if path.exists() and read_json(path) != plan:
        raise ValueError("Existing matched-step plan differs")
    for arm, record in checkpoints.items():
        rel = Path("train") / arm / record["selected_path"]
        for name, key in (("adapter_config.json", "adapter_config_sha256"),
                          ("adapter_model.safetensors", "adapter_sha256")):
            target = run / rel / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                pending = target.with_suffix(target.suffix + ".pending")
                copyfile(primary / rel / name, pending)
                pending.replace(target)
            if file_sha(target) != record[key]:
                raise ValueError(f"Copied checkpoint checksum differs: {target}")
        write_json(run / "train" / arm / "selected.json", record)
    write_json(path, plan)
    return plan


def summarize(primary, run):
    path = run / "test-plan.json"
    plan = read_json(path)
    checkpoints = {arm: read_json(run / "train" / arm / "selected.json") for arm in ADDED_STEPS}
    if plan != plan_for(primary, checkpoints):
        raise ValueError("Matched-step plan differs from its frozen primary evidence")
    if {arm: row["step"] for arm, row in checkpoints.items()} != ADDED_STEPS:
        raise ValueError("Matched-step checkpoints differ from the preassigned update counts")
    original = read_json(primary / "summary.json")
    if original["test_plan_sha256"] != plan["primary_plan_sha256"]:
        raise ValueError("Primary summary does not match its test plan")
    prior = {(p["base"]["layout_seed"], p["base"]["policy_seed"]): p for p in original["pairs"]}
    adapter_shas = {arm: row["adapter_sha256"] for arm, row in checkpoints.items()}
    rows = []
    for case in plan["cases"]:
        layout, seed = case["layout_seed"], case["inference_seeds"][0]
        trial = {f"{arm}_{step}": prior[layout, seed][arm] for arm, step in SELECTED_STEPS.items()}
        for arm, step in ADDED_STEPS.items():
            trial[f"{arm}_{step}"] = audit_episode(run, case, seed, arm, file_sha(path),
                                                   plan["gpu_uuid"], adapter_shas)
        rows.append(trial)
    contrasts = {}
    for step in (400, 600):
        natural, corrected = f"natural_{step}", f"corrected_{step}"
        contrasts[str(step)] = {
            "natural": per_arm([trial[natural] for trial in rows]),
            "corrected": per_arm([trial[corrected] for trial in rows]),
            "paired_corrected_vs_natural": {metric: paired(rows, corrected, natural, metric)
                                          for metric in PAIR_METRICS},
        }
    return {"supplemental_episodes": 2 * len(rows), "challenge_pairs": len(rows),
            "test_plan_sha256": file_sha(path), "matched_steps": contrasts, "pairs": rows}


def gpu_snapshot(uuid):
    cmd = ["nvidia-smi", "-i", uuid, "--query-gpu=index,uuid,memory.used,utilization.gpu",
           "--format=csv,noheader,nounits"]
    row = subprocess.check_output(cmd, text=True, timeout=15).strip()
    index, actual, memory, utilization = (part.strip() for part in row.split(","))
    xml = subprocess.check_output(["nvidia-smi", "-i", uuid, "-q", "-x"], timeout=15)
    gpu = ElementTree.fromstring(xml).find("gpu")
    if gpu is None or gpu.findtext("uuid") != uuid or actual != uuid:
        raise ValueError("Could not verify the frozen GPU identity")
    processes = [(item.findtext("type"), item.findtext("process_name"))
                 for item in gpu.findall("processes/process_info")]
    non_display_processes = sum(item != ("G", "/usr/lib/xorg/Xorg") for item in processes)
    return {"index": int(index), "memory_mib": int(memory), "utilization": int(utilization),
            "non_display_processes": non_display_processes}


def is_idle(status):
    return status["memory_mib"] <= 64 and status["utilization"] == 0 and not status["non_display_processes"]


def wait_until_idle(uuid):
    deadline = monotonic() + 15 * 60
    while True:
        status = gpu_snapshot(uuid)
        if is_idle(status):
            sleep(2)
            later = gpu_snapshot(uuid)
            if is_idle(later) and status["index"] == later["index"]:
                return later
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("The frozen GPU did not become idle")
        print("WAITING_FOR_GPU", json.dumps(status), flush=True)
        sleep(min(60, remaining))


def run_experiment(primary, run, lab, source):
    if (primary / "controller.exit").read_text().strip() != "0":
        raise ValueError("The primary experiment did not exit successfully")
    for line in (primary / "logs/source.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if file_sha(source / filename) != digest:
            raise ValueError(f"Primary evaluation source changed: {filename}")
    plan = prepare(primary, run)
    uuid = plan["gpu_uuid"]
    for name in ("tmp", "cache", "logs"):
        (run / name).mkdir(exist_ok=True)
    cache = run / "cache"
    env = {
        **os.environ, "TABLECLEAR_LAB": str(lab), "CUDA_VISIBLE_DEVICES": uuid,
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(primary / "python-extra"), "TMPDIR": str(run / "tmp"),
        "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache / "xdg"),
        "TORCH_HOME": str(cache / "torch"), "LIBERO_CONFIG_PATH": str(cache / "libero"),
        "ROBOENV_ASSET_CACHE": str(lab / "roboenv-assets"),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for arm, step in ADDED_STEPS.items():
        print("CHECKPOINT", arm, step, file_sha(run / "test-plan.json"), flush=True)
        before = wait_until_idle(uuid)
        log_path = run / "logs" / f"{arm}-{step}.log"
        with log_path.open("a", encoding="utf-8") as log:
            print("GPU_BEFORE", uuid, json.dumps(before), file=log, flush=True)
            try:
                subprocess.run([
                    "timeout", "--kill-after=30s", "25m", sys.executable, "-B", "-u",
                    str(source / "evaluate.py"), arm, "--lab", str(lab),
                    "--run", str(run), "--gpu-uuid", uuid,
                ], env={**env, "MUJOCO_EGL_DEVICE_ID": str(before["index"])},
                    cwd=run, stdout=log, stderr=log, check=True)
            finally:
                print("GPU_AFTER", uuid, json.dumps(gpu_snapshot(uuid)), file=log, flush=True)
    result = summarize(primary, run)
    write_json(run / "summary.json", result)
    print("MATCHED_STEPS_COMPLETE", result["supplemental_episodes"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "score"))
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lab", type=Path)
    args = parser.parse_args()
    primary, out = args.primary.resolve(), args.out.resolve()
    if primary == out:
        parser.error("Use a separate output directory for the matched-step control")
    if args.stage == "score":
        result = summarize(primary, out)
        print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
        return
    if args.lab is None:
        parser.error("--lab is required for simulation")
    lab = args.lab.resolve()
    if not primary.is_relative_to(lab) or not out.is_relative_to(primary):
        parser.error("Keep the supplemental result inside the primary run on the personal data disk")
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_experiment(primary, out, lab, primary / "src-torch-state")


if __name__ == "__main__":
    main()
