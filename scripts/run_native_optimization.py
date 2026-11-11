"""Run the fixed TableClear backbone pilot, chunk sweep and confirmation study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from threading import Lock
import tomllib

from compare_native_backbones import BUDGET, SCENES, make_plan, read_episode, run_case, write_json
from native_backbone_metrics import audit_trace, select_horizon, summarize, verify_pair, verify_split_assets
from native_pi05 import PROJECT, model_directory, weights_path


DEVELOPMENT_INDICES = (18, 20, 23, 27, 33, 46)
CONFIRMATION_INDICES = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
BASELINES = ("smolvla-10", "pi05-10")
PHASES = ("preflight", "pilot", "development", "confirmation")


@dataclass(frozen=True)
class Device:
    uuid: str
    egl: int

    def environment(self) -> dict[str, str]:
        return {**os.environ, "CUDA_VISIBLE_DEVICES": self.uuid, "MUJOCO_EGL_DEVICE_ID": str(self.egl),
                "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@dataclass(frozen=True)
class Condition:
    phase: str
    case: dict
    variants: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.case['scene']:02d}-seed{self.case['seed']}-init{self.case['init']:02d}"


def study_plan() -> dict:
    original = make_plan()
    pilot = [
        {"scene": case["scene"], "seed": case["seed"], "init": case["init"],
         "variants": [f"{policy}-10" for policy in case["policies"]]}
        for case in original.pop("conditions")
    ]
    with (PROJECT / "versions.lock").open("rb") as stream:
        upstreams = tomllib.load(stream)["upstreams"]
    return {
        **original, "schema": "tableclear/native-optimization/1", "upstreams": upstreams,
        "pilot": pilot,
        "development": [{"scene": scene, "seed": 0, "init": index} for index in DEVELOPMENT_INDICES for scene in SCENES],
        "confirmation": [{"scene": scene, "seed": index // 5, "init": index}
                         for index in CONFIRMATION_INDICES for scene in SCENES],
        "development_variants": ["pi05-5", "pi05-10", "pi05-25"],
        "confirmation_baselines": list(BASELINES),
        "planned_episodes": {"pilot": 18, "development": 54, "confirmation": [72, 108]},
    }


def conditions(plan: dict, phase: str, selected_chunk: int = 10) -> list[Condition]:
    variants = tuple(plan["development_variants"]) if phase == "development" else BASELINES
    if phase == "confirmation" and selected_chunk != 10:
        variants += (f"pi05-{selected_chunk}",)
    result = []
    for index, planned in enumerate(plan[phase]):
        offset = index % len(variants)
        case = {key: planned[key] for key in ("scene", "seed", "init")}
        result.append(Condition(phase, case, tuple(planned.get("variants", variants[offset:] + variants[:offset]))))
    return result


def device_argument(value: str) -> Device:
    match = re.fullmatch(r"(GPU-[a-fA-F0-9-]{36})@([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("expected full NVIDIA GPU UUID@EGL_INDEX, e.g. GPU-...@0")
    return Device(match[1], int(match[2]))


def checkout(path: Path, expected: str) -> None:
    git = ["git", "-C", str(path)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != expected or changes:
        raise ValueError(f"Expected clean checkout {expected} at {path}; found {revision}")


def prepare_runtime(args, plan) -> dict:
    checkout(PROJECT, plan["project_commit"])
    checkout(args.roboenv, plan["upstreams"]["roboenv"]["commit"])
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the pinned LeRobot checkout in the model Python")
    checkout(Path(spec.origin).parent, plan["upstreams"]["lerobot"]["commit"])
    weights_path(model_directory(args.workspace), verify=True)
    sim_versions = subprocess.check_output([
        str(args.sim_python), "-c", "import importlib.metadata as m, importlib.util as u, json, sys; "
        "libero = u.find_spec('libero'); assert libero is not None, 'LIBERO is absent'; "
        "print(json.dumps({'python': sys.version.split()[0], 'mujoco': m.version('mujoco'), "
        "'robosuite': m.version('robosuite'), 'libero_package': list(libero.submodule_search_locations or [])}))",
    ], text=True, env={**os.environ, "PYOPENGL_PLATFORM": "egl", "MUJOCO_GL": "egl"})
    return {"model_packages": {name: importlib.metadata.version(name) for name in ("torch", "lerobot", "transformers")},
            "simulator_packages": json.loads(sim_versions.splitlines()[-1])}


def card_inventory(devices: list[Device]) -> dict:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    available = {}
    for line in raw.splitlines():
        uuid, name, driver, total, free = map(str.strip, line.split(","))
        available[uuid] = {"name": name, "driver_version": driver, "total_mib": int(total), "free_mib": int(free)}
    if any(device.uuid not in available for device in devices):
        raise ValueError("A requested GPU UUID is absent from nvidia-smi")
    return {device.uuid: {**available[device.uuid], "egl_device_id": device.egl} for device in devices}


def launch(args, plan, case, variant, device, directory, lock_fd, *, budget=BUDGET) -> dict:
    policy, chunk = variant.split("-")
    worker_args = argparse.Namespace(**{**vars(args), "output": directory})
    row = run_case(worker_args, plan, case, policy, chunk=int(chunk), budget=budget,
                   gpu_uuid=device.uuid, env=device.environment(), lock_fd=lock_fd)
    artifact = directory / row["artifact"]
    return {**row, **audit_trace(artifact, row), "artifact": artifact.relative_to(args.output).as_posix()}


def load_group(root, directory, plan, condition):
    report = directory / "pair.json"
    if not report.exists():
        return None
    group = json.loads(report.read_bytes())
    if group["case"] != condition.case or set(group["episodes"]) != set(condition.variants):
        raise ValueError(f"Saved group differs from the frozen plan: {report}")
    for variant, saved in group["episodes"].items():
        policy, chunk = variant.split("-")
        artifact = root / saved["artifact"]
        if artifact.parent != directory:
            raise ValueError(f"Saved artifact is outside its group: {artifact}")
        row = read_episode(artifact, plan, condition.case, policy, chunk=int(chunk), gpu_uuid=directory.parent.name)
        if row is None or {**row, **audit_trace(artifact, row), "artifact": saved["artifact"]} != saved:
            raise ValueError(f"Saved group no longer matches its episode evidence: {artifact}")
    verify_pair(group["episodes"])
    return group


def run_groups(args, plan, jobs, devices, lock_fd) -> dict:
    pending = []
    results = {}
    for job in jobs:
        matches = list((args.output / job.phase).glob(f"GPU-*/{job.name}"))
        if len(matches) > 1:
            raise ValueError(f"Duplicate GPU assignments for {job.phase}/{job.name}")
        directory = matches[0] if matches else None
        if directory:
            completed = load_group(args.output, directory, plan, job)
            if completed is not None:
                results[job.phase, job.name] = completed
                continue
            if all(device.uuid != directory.parent.name for device in devices):
                raise ValueError(f"Resume {job.phase}/{job.name} with its original GPU {directory.parent.name}")
        pending.append((job, directory))

    mutex = Lock()
    errors = []

    def work(device, prioritize_development):
        while True:
            with mutex:
                eligible = [(job, directory) for job, directory in pending
                            if directory is None or directory.parent.name == device.uuid]
                if errors or not eligible:
                    return
                job, directory = min(eligible, key=lambda item: item[0].phase != "development") if prioritize_development else eligible[0]
                pending.remove((job, directory))
            try:
                directory = directory or args.output / job.phase / device.uuid / job.name
                directory.mkdir(parents=True, exist_ok=True)
                episodes = {name: launch(args, plan, job.case, name, device, directory, lock_fd) for name in job.variants}
                verify_pair(episodes)
                group = {"case": job.case, "gpu_uuid": device.uuid, "episodes": episodes,
                         "infrastructure_failed_attempts": len(list(directory.glob("*.log"))) - len(episodes)}
                write_json(directory / "pair.json", group)
                with mutex:
                    results[job.phase, job.name] = group
                    print(f"{job.phase}/{job.name} paired on {device.uuid} ({len(results)}/{len(jobs)})", flush=True)
            except Exception as exc:
                with mutex:
                    errors.append(exc)
                return

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(work, device, index >= 2) for index, device in enumerate(devices)]
        for future in futures:
            future.result()
    if errors:
        raise RuntimeError(f"Study stopped after {len(results)}/{len(jobs)} groups: {errors[0]}") from errors[0]
    if len(results) != len(jobs):
        raise RuntimeError("Study ended with unscheduled groups")
    by_phase = {}
    for job in jobs:
        by_phase.setdefault(job.phase, []).append(results[job.phase, job.name])
    return by_phase


def freeze(path, value) -> None:
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f"Existing evidence differs from the frozen study: {path}")
    else:
        write_json(path, value)


def report_phase(root, phase, groups) -> None:
    verify_split_assets(groups)
    output = root / phase
    baseline = "pi05-10" if phase == "development" else "smolvla-10"
    report = summarize(groups, baseline)
    report["status"] = "completed"
    report["infrastructure_failed_attempts"] = sum(group["infrastructure_failed_attempts"] for group in groups)
    report["discordant_pairs"] = {
        name: [{"case": group["case"], "winner": name if group["episodes"][name]["success"] else metric["reference"]}
               for group in groups if group["episodes"][name]["success"] != group["episodes"][metric["reference"]]["success"]]
        for name, metric in report["comparisons"].items()
    }
    write_json(output / "pairs.json", groups)
    write_json(output / "summary.json", report)
    if phase == "pilot":
        review = [{"case": group["case"], "gpu_uuid": group["gpu_uuid"], "variants": {
            name: {key: row[key] for key in (
                "artifact", "success", "gpu_placement", "first_observed_goal_steps", "peak_targets_at_chunk_boundaries",
            )}
            for name, row in group["episodes"].items()}}
            for group in groups if group["case"]["scene"] == 1]
        write_json(output / "scene-01-review.json", review)


def execute(args, plan) -> None:
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(args.output / "plan.json", plan)
        root = args.output
        status = {"status": "running", "stage": "preflight", "through": args.through, "pid": os.getpid()}
        write_json(root / "status.json", status)
        try:
            preflight = {"runtime": prepare_runtime(args, plan), "devices": card_inventory(args.gpu), "single_actions": []}
            freeze(root / "runtime.json", {"hostname": platform.node(), **preflight["runtime"],
                   "driver_versions": sorted({gpu["driver_version"] for gpu in preflight["devices"].values()})})
            case = {"scene": 1, "seed": 0, "init": 0}
            primary = args.gpu[0]
            for device in args.gpu:
                output = root / "preflight" / device.uuid / f"egl-{device.egl}" / "one-step"
                output.mkdir(parents=True, exist_ok=True)
                row = launch(args, plan, case, "pi05-10", device, output, lock.fileno(), budget=1)
                preflight["single_actions"].append(row)
            output = root / "preflight" / primary.uuid / f"egl-{primary.egl}" / "full"
            output.mkdir(parents=True, exist_ok=True)
            preflight["full_episode"] = launch(args, plan, case, "pi05-10", primary, output, lock.fileno())
            write_json(root / "preflight" / "summary.json", preflight)

            if args.through != "preflight":
                pilot_jobs = conditions(plan, "pilot")
                dev_jobs = conditions(plan, "development")
                parallel = len(args.gpu) == 3 and args.through != "pilot"
                status["stage"] = "pilot_and_development" if parallel else "pilot"
                write_json(root / "status.json", status)
                early = run_groups(args, plan, pilot_jobs + dev_jobs if parallel else pilot_jobs, args.gpu, lock.fileno())
                report_phase(root, "pilot", early["pilot"])

                if args.through != "pilot":
                    status["stage"] = "development"
                    write_json(root / "status.json", status)
                    development = early["development"] if parallel else run_groups(args, plan, dev_jobs, args.gpu, lock.fileno())["development"]
                    verify_split_assets(early["pilot"] + development)
                    report_phase(root, "development", development)
                    selection = select_horizon(development)
                    selection["development_pairs_sha256"] = sha256((root / "development" / "pairs.json").read_bytes()).hexdigest()
                    freeze(root / "selection.json", selection)

                    if args.through == "confirmation":
                        status["stage"] = "confirmation"
                        write_json(root / "status.json", status)
                        jobs = conditions(plan, "confirmation", selection["selected_chunk"])
                        confirmation = run_groups(args, plan, jobs, args.gpu, lock.fileno())["confirmation"]
                        verify_split_assets(early["pilot"] + development + confirmation)
                        report_phase(root, "confirmation", confirmation)
            status["status"] = "completed"
        except BaseException as exc:
            status.update(status="failed", error=str(exc) or type(exc).__name__)
            raise
        finally:
            write_json(root / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="launch the study; the default prints its frozen plan")
    parser.add_argument("--through", choices=PHASES, default="confirmation")
    parser.add_argument("--gpu", action="append", default=[], type=device_argument, metavar="UUID@EGL_INDEX")
    for name in ("workspace", "roboenv", "sim-python", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    plan = study_plan()
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if not all((args.workspace, args.roboenv, args.sim_python, args.output)) or not 1 <= len(args.gpu) <= 3:
        parser.error("--run requires all paths and one to three --gpu UUID@EGL_INDEX arguments")
    if len({device.uuid for device in args.gpu}) != len(args.gpu) or len({device.egl for device in args.gpu}) != len(args.gpu):
        parser.error("each GPU UUID and EGL index must be distinct")
    args.output = args.output.resolve()
    args.workspace = args.workspace.resolve()
    args.roboenv = args.roboenv.resolve()
    args.sim_python = args.sim_python.absolute()
    execute(args, plan)


if __name__ == "__main__":
    main()
