"""Freeze new layouts and run four-arm native Scene 08 evaluation."""

from argparse import ArgumentParser
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from common import (
    ARCHIVE_SHA256, ARMS, BUDGET, DEFAULT_LAB, EXECUTE, Lab, RULE_SHA256, SCENE, TASK_ID,
    file_sha, read_json, read_physics, write_json,
)
from policy import load_service, read_observation
from simulator import Simulator


COHORTS = {
    "regular": {"range": range(841000, 841096), "layouts": 12, "inference_seeds": (61, 62)},
    "challenge": {"range": range(841500, 841548), "layouts": 8, "inference_seeds": (61,)},
}


def load_rule(lab, run):
    path = run / "v1-release-observer.py"
    if not path.exists():
        if file_sha(lab.archive) != ARCHIVE_SHA256:
            raise ValueError("V1 evidence checksum differs")
        with ZipFile(lab.archive) as source:
            code = source.read("observe.py")
        if sha256(code).hexdigest() != RULE_SHA256:
            raise ValueError("Expected the V1 frozen release observer")
        path.write_bytes(code)
    if file_sha(path) != RULE_SHA256:
        raise ValueError("V1 observer checksum differs")
    spec = importlib.util.spec_from_file_location("frozen_v1_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Release:
    def __init__(self, module):
        self.module = module
        self.rule = module.FirstSoupRelease()

    def apply(self, path, actions, start):
        if self.rule.trigger is None:
            with np.load(path, allow_pickle=False) as raw:
                names = ("camera1", "camera2")
                images = [raw[camera][::-1, ::-1] for camera in names]
                depths = [raw[f"{camera}_depth"][::-1, ::-1] for camera in names]
                intrinsics = [raw[f"{camera}_intrinsic"] for camera in names]
                extrinsics = [raw[f"{camera}_extrinsic"] for camera in names]
                signals = self.module.evidence(
                    images, depths, intrinsics, extrinsics, raw["eef_pos"], raw["gripper_qpos"],
                )
                self.rule.observe(signals, depths[0], actions, start)
        return list(self.rule.override(actions, start))


def identity(initial):
    return {
        "raw_sha256": initial["init_state_sha256"],
        "settled_sha256": initial["sim_state_sha256"],
        "observation_sha256": initial["observation_sha256"],
        "bddl_sha256": initial["bddl_sha256"],
    }


def rollout(lab, service, folder, *, layout, seed, gpu, observer=None,
            expected=None, screen=False, fixed=None, video=False):
    from driver import predict_chunk
    from run_native_osc import validate_actions, verify_gpu_placement

    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(0)
    for key in ("policy", "preprocessor", "postprocessor"):
        service[key].reset()
    start_time = perf_counter()
    steps, edits = 0, []
    trace_path = folder / "actions.jsonl"
    with Simulator(lab, folder, layout, init=fixed, video=video) as worker, trace_path.open("w") as trace:
        initial = identity(worker.initial)
        if expected is not None and initial != expected:
            raise ValueError("Fresh initial state does not match the locked test plan")
        frame = folder / "observation.npz"
        task = worker.initial["instruction"]
        response = worker.initial
        placement = None
        while steps < BUDGET:
            model_actions = predict_chunk(service, read_observation(frame, task))
            if len(model_actions) != 50:
                raise ValueError("V1 model must predict 50 actions")
            proposed = model_actions[:min(EXECUTE, BUDGET - steps)]
            validate_actions(proposed)
            if placement is None:
                placement = verify_gpu_placement(gpu, worker.process.pid)
            commands = [action.copy() for action in proposed]
            override = observer.apply(frame, commands, steps) if observer is not None else []
            response = worker.step(commands)
            executed = response["executed"]
            if not 1 <= executed <= len(commands):
                raise ValueError("Unexpected count of applied native actions")
            applied = [edit for edit in override if edit["step"] <= steps + executed]
            trace.write(json.dumps({
                **response, "start_step": steps, "commands": commands[:executed],
                "proposals": proposed[:executed], "edits": applied,
            }, allow_nan=False) + "\n")
            steps += executed
            edits.extend(applied)
            if response["success"] or response.get("done") or (screen and edits):
                break
    return {
        "task": TASK_ID, "layout_seed": layout, "policy_seed": seed, "initial": initial,
        "steps": steps, "success": bool(response["success"]), "done": bool(response["done"]),
        "status": "goal" if response["success"] else "done" if response["done"] else
        "screened" if screen and edits else "budget",
        "edits": edits, "gpu": placement,
        **read_physics(folder, steps, response["success"]),
        "seconds": perf_counter() - start_time, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "actions_sha256": file_sha(trace_path), "audit_sha256": file_sha(folder / "audit.jsonl"),
        **({"video_sha256": file_sha(folder / "replay.mp4")} if video else {}),
    }


def historical(lab, run):
    sources = read_json(run / "replay-manifest.json")["episodes"]
    raw = {row["layout_sha256"] for row in sources}
    settled = {row["settled_sha256"] for row in sources}
    archive = lab.review / "integrated-pi05-final-acceptance/layouts.json"
    for row in read_json(archive)["layouts"]:
        if row["scene"] == SCENE:
            raw.add(row["raw_sha256"])
            settled.add(row["settled_sha256"])
    return raw, settled


def attempt_regular(lab, folder, layout):
    with Simulator(lab, folder, layout) as sim:
        return {"initial": identity(sim.initial), "goal_at_reset": bool(sim.initial["success"])}


def freeze(lab, run, gpu):
    target = run / "test-plan.json"
    if target.exists():
        return
    module = load_rule(lab, run)
    used_raw, used_settled = historical(lab, run)
    service = None
    cases, attempts = [], []
    for cohort, spec in COHORTS.items():
        selected = 0
        for layout in spec["range"]:
            if selected == spec["layouts"]:
                break
            folder = run / "screen" / cohort / str(layout)
            folder.mkdir(parents=True, exist_ok=True)
            marker = folder / "screen.json"
            if marker.exists():
                result = read_json(marker)
            else:
                try:
                    if cohort == "challenge":
                        if service is None:
                            service = load_service(lab)
                        result = rollout(lab, service, folder, layout=layout, seed=spec["inference_seeds"][0],
                                         gpu=gpu, observer=Release(module), screen=True)
                    else:
                        result = attempt_regular(lab, folder, layout)
                except (RuntimeError, ValueError):
                    log = (folder / "simulator.log").read_text(encoding="utf-8", errors="replace")
                    if "sampled reset collides" not in log and "already satisfies a placement goal" not in log:
                        raise
                    result = {"rejected": "native_state_duplicate_or_pre_satisfied"}
                write_json(marker, result)
            reason = result.get("rejected")
            initial = result.get("initial")
            if reason is None and (initial["raw_sha256"] in used_raw or initial["settled_sha256"] in used_settled):
                reason = "duplicate_historical_or_candidate_state"
            if initial is not None:
                used_raw.add(initial["raw_sha256"])
                used_settled.add(initial["settled_sha256"])
            if reason is None and cohort == "challenge" and not result["edits"]:
                reason = "no_applied_v1_edit"
            attempts.append({"cohort": cohort, "layout_seed": layout, "reason": reason,
                             "screen_sha256": file_sha(marker)})
            print("SCREEN", cohort, layout, reason or "accepted", flush=True)
            if reason is not None:
                continue
            case = {"cohort": cohort, "layout_seed": layout, "initial": initial,
                    "inference_seeds": spec["inference_seeds"], "representative_video": selected < 2}
            if cohort == "challenge":
                case["first_v1_override"] = result["edits"][0]["step"]
            cases.append(case)
            selected += 1
        if cohort == "regular" and selected != spec["layouts"]:
            raise RuntimeError("The independent regular cohort is incomplete")
    write_json(target, {"schema": "v1-scene08-posttrain/1", "rule_sha256": RULE_SHA256, "cases": cases,
                        "screened": attempts, "planned": {name: values["layouts"] for name, values in COHORTS.items()},
                        "replay_manifest_sha256": file_sha(run / "replay-manifest.json"), "gpu_uuid": gpu})


def adapter_path(run, arm):
    if arm in ("base", "rule"):
        return None
    folder = run / "train" / arm
    selected = read_json(folder / "selected.json")
    path = folder / selected["selected_path"]
    if file_sha(path / "adapter_model.safetensors") != selected["adapter_sha256"]:
        raise ValueError(f"Selected adapter integrity failure for {arm}")
    return path


def evaluate_arm(lab, run, arm, gpu):
    plan_path = run / "test-plan.json"
    plan = read_json(plan_path)
    if plan["gpu_uuid"] != gpu:
        raise ValueError("Same-device pairing requires the GPU used to freeze the plan")
    module = load_rule(lab, run) if arm == "rule" else None
    adapter = adapter_path(run, arm)
    service = None
    for case in plan["cases"]:
        cohort, layout = case["cohort"], case["layout_seed"]
        for seed in case["inference_seeds"]:
            target = run / "eval" / cohort / str(layout) / str(seed) / arm
            marker = target / "result.json"
            if marker.is_file():
                previous = read_json(marker)
                if (previous["test_plan_sha256"] != file_sha(plan_path)
                        or previous["actions_sha256"] != file_sha(target / "actions.jsonl")
                        or previous["audit_sha256"] != file_sha(target / "audit.jsonl")):
                    raise ValueError(f"Mismatching saved evaluation at {target}")
                continue
            if service is None:
                service = load_service(lab, adapter=adapter)
            trial = run / "eval-attempts"
            trial.mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix=f"{arm}-{layout}-{seed}-", dir=trial))
            values = rollout(lab, service, root, layout=layout, seed=seed, gpu=gpu,
                             observer=Release(module) if module is not None else None,
                             expected=case["initial"], video=case["representative_video"])
            if arm == "rule" and cohort == "challenge" and (
                not values["edits"] or values["edits"][0]["step"] != case["first_v1_override"]
            ):
                raise ValueError("Frozen rule opportunity differs between selection and evaluation")
            values.update(arm=arm, cohort=cohort, test_plan_sha256=file_sha(plan_path),
                          adapter_sha256=file_sha(adapter / "adapter_model.safetensors") if adapter else None)
            write_json(root / "result.json", values)
            target.parent.mkdir(parents=True, exist_ok=True)
            root.rename(target)
            print("EVALUATION", arm, cohort, layout, seed, values["success"], values["steps"], flush=True)


def smoke(lab, run, gpu):
    root = run / "adapter-rollout-smoke"
    if (root / "result.json").exists():
        return
    root.mkdir(exist_ok=True)
    service = load_service(lab, adapter=run / "train-smoke/step-0001/adapter")
    result = rollout(lab, service, root, layout=8, seed=8, gpu=gpu, fixed=44, video=True)
    write_json(root / "result.json", result)
    print("ADAPTER_ROLLOUT_SMOKE_OK", result["steps"], result["status"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "smoke", *ARMS))
    parser.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES"), required=False)
    args = parser.parse_args()
    if not args.gpu_uuid or not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("Specify the authorized CUDA/EGL GPU UUID")
    lab = Lab(args.lab)
    lab.add_imports()
    if args.stage == "freeze":
        freeze(lab, args.run, args.gpu_uuid)
    elif args.stage == "smoke":
        smoke(lab, args.run, args.gpu_uuid)
    else:
        evaluate_arm(lab, args.run, args.stage, args.gpu_uuid)


if __name__ == "__main__":
    main()
