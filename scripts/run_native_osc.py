"""Run one direct SmolVLA episode on official LIBERO OSC in split Python environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CAMERAS = {"camera1": "agentview_image", "camera2": "robot0_eye_in_hand_image"}


def validate_actions(actions: object) -> None:
    if not isinstance(actions, list) or not 1 <= len(actions) <= 50 or any(
        not isinstance(action, list) or len(action) != 7
        or not all(type(value) in (float, int) and math.isfinite(value) for value in action)
        for action in actions
    ):
        raise ValueError("expected 1 to 50 finite 7D actions")


def simulate(
    output: Path, roboenv: Path, seed: int, init_index: int,
    socket_path: Path | None, steps: int,
) -> None:
    # Keep the response pipe separate from native library diagnostics.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    import numpy as np

    sys.path[:0] = [str(roboenv / "src"), str(roboenv / "runtime" / "src")]
    from roboenv_runtime.adapters.libero.config import ensure_libero_config

    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    suite = get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = suite.get_task_init_states(0)
    if not 0 <= init_index < len(states):
        raise ValueError(f"init index must be between 0 and {len(states) - 1}")

    def capture(obs, **metadata):
        arrays = dict(
            **{name: obs[key] for name, key in CAMERAS.items()},
            eef_pos=obs["robot0_eef_pos"],
            eef_quat=obs["robot0_eef_quat"],
            gripper_qpos=obs["robot0_gripper_qpos"],
        )
        np.savez(output / "observation.npz", **arrays)
        return {
            **metadata,
            "sim_time": float(env.env.sim.data.time),
            "sim_state_sha256": hashlib.sha256(env.get_sim_state().tobytes()).hexdigest(),
            "observation_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in arrays.items()},
        }

    def advance(actions):
        for executed, action in enumerate(actions, start=1):
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
            success = bool(env.check_success())
            if success or done:
                break
        return capture(obs, executed=executed, success=success, done=bool(done))

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, control_freq=20,
    )
    try:
        env.reset()
        env.seed(seed)
        env.reset()
        env.set_init_state(states[init_index])
        initial_state_hash = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
        # Match the pinned LeRobot LIBERO wrapper's settling policy.
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        controller = env.env.robots[0].controller
        controller.use_delta = True
        if controller.name != "OSC_POSE" or env.env.robots[0].action_dim != 7:
            raise RuntimeError("expected a 7D native OSC_POSE action space")
        initial = capture(
            obs,
            instruction=task.language,
            controller=controller.name,
            control_hz=env.env.control_freq,
            settle_frames=10,
            seed=seed,
            init_state_index=init_index,
            init_state_sha256=initial_state_hash,
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
        )
        if socket_path is not None:
            from native_osc_session import serve

            serve(socket_path, output, initial, steps, validate_actions, advance)
            return

        print(json.dumps(initial, allow_nan=False), file=protocol)
        for line in sys.stdin:
            actions = json.loads(line)
            validate_actions(actions)
            result = advance(actions)
            print(json.dumps(result, allow_nan=False), file=protocol)
            if result["success"] or result["done"]:
                break
    finally:
        env.close()
        protocol.close()


def episode(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import yaml
    from lerobot.processor.env_processor import LiberoProcessorStep
    from PIL import Image

    sys.path.insert(0, str(PROJECT / "kinex_devices" / "smolvla"))
    from driver import open_service, predict_chunk

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "tableclear/native-osc-smoke/1",
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
        ).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "libero/libero-10-01",
        "method": "direct_vla",
        "seed": args.seed,
        "init_state_index": args.init_index,
        "step_budget": args.steps,
        "steps": 0,
        "vla_calls": 0,
        "success": None,
        "terminal_reason": "runtime_error",
    }
    torch.manual_seed(args.seed)
    processor = LiberoProcessorStep()
    worker_command = [
        str(args.sim_python), str(Path(__file__).resolve()), "--sim-worker",
        "--roboenv", str(args.roboenv), "--output", str(output),
        "--seed", str(args.seed), "--init-index", str(args.init_index),
    ]

    def read_observation(instruction, image_label=None):
        with np.load(output / "observation.npz", allow_pickle=False) as obs:
            batch = {
                f"observation.images.{name}": torch.from_numpy(obs[name]).permute(2, 0, 1)
                .unsqueeze(0).float().div(255)
                for name in CAMERAS
            }
            batch["observation.robot_state"] = {
                "eef": {key: torch.from_numpy(obs[f"eef_{key}"]).unsqueeze(0) for key in ("pos", "quat")},
                "gripper": {"qpos": torch.from_numpy(obs["gripper_qpos"]).unsqueeze(0)},
            }
        batch = processor.observation(batch)
        if image_label:
            for name in CAMERAS:
                rgb = batch[f"observation.images.{name}"][0].permute(1, 2, 0)
                Image.fromarray(rgb.mul(255).round().byte().numpy()).save(output / f"{image_label}-{name}.png")
        return {**batch, "task": instruction}

    def receive(worker):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("LIBERO worker closed its response pipe; see simulator.log")
        return json.loads(line)

    try:
        with (output / "simulator.log").open("w", encoding="utf-8") as sim_log, (
            output / "actions.jsonl"
        ).open("w", encoding="utf-8", buffering=1) as trace:
            worker = subprocess.Popen(
                worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=sim_log, text=True, encoding="utf-8",
            )
            try:
                initial = receive(worker)
                record.update(initial)
                instruction = initial["instruction"]
                obs = read_observation(instruction, "initial")
                record["initial_observation_state"] = obs["observation.state"][0].tolist()
                artifact_root = args.workspace / ".kinex" / "devices" / "smolvla"
                config = yaml.safe_load((artifact_root / "profile.yaml").read_text(encoding="utf-8"))
                record["model_artifacts"] = config["intrinsics"]["artifacts"]
                record["model"] = config["intrinsics"]["variant"]
                service = open_service(config, artifact_root=artifact_root)
                horizon = service["policy"].config.n_action_steps
                record["chunk_horizon"] = horizon
                while record["steps"] < args.steps:
                    count = min(horizon, args.steps - record["steps"])
                    actions = predict_chunk(service, obs)[:count]
                    worker.stdin.write(json.dumps(actions, allow_nan=False) + "\n")
                    worker.stdin.flush()
                    result = receive(worker)
                    executed = result["executed"]
                    if not 1 <= executed <= len(actions):
                        raise RuntimeError(f"invalid executed frame count: {executed}")
                    trace.write(json.dumps({"start_step": record["steps"], "actions": actions[:executed], **result}) + "\n")
                    record["steps"] += executed
                    record["vla_calls"] += 1
                    record["success"] = result["success"]
                    record["final_sim_time"] = result["sim_time"]
                    print(f"steps={record['steps']} simulator_success={record['success']}", flush=True)
                    if result["success"] or result["done"] or record["steps"] == args.steps:
                        read_observation(instruction, "final")
                        record["terminal_reason"] = (
                            "simulator_success" if result["success"]
                            else "simulator_done" if result["done"] else "step_budget"
                        )
                        break
                    obs = read_observation(instruction)
            finally:
                try:
                    worker.stdin.close()
                finally:
                    worker.wait()
                    worker.stdout.close()
            if worker.returncode:
                raise RuntimeError(f"LIBERO worker exited with status {worker.returncode}; see simulator.log")
    except Exception as exc:
        record["terminal_reason"] = "runtime_error"
        record["error"] = str(exc)
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "episode.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roboenv", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--sim-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--socket", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.steps <= 520:
        parser.error("--steps must be between 1 and 520")
    if args.sim_worker:
        simulate(args.output, args.roboenv, args.seed, args.init_index, args.socket, args.steps)
        return
    if args.workspace is None or args.sim_python is None or args.socket is not None:
        parser.error("--workspace and --sim-python are required for a direct episode")
    print(json.dumps(episode(args), indent=2))


if __name__ == "__main__":
    main()
