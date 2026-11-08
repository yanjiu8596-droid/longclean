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
EXTRA_VIEWS = ("birdview", "frontview", "sideview", "galleryview", "paperview", "robot0_robotview")


class ReleaseSampling:
    """Sample the first 12 opening actions after a sustained close command."""

    def __init__(self):
        self.closed = 0
        self.opening = 0

    def observe(self, command):
        if command >= 0.5:
            self.closed += 1
            self.opening = 0
        elif command <= -0.5 and (self.closed >= 3 or self.opening):
            self.closed = 0
            self.opening += 1
            return self.opening <= 12
        else:
            self.closed = self.opening = 0
        return False


def read_rgbd_frame(path):
    import numpy as np
    from PIL import Image

    with np.load(path, allow_pickle=False) as raw:
        images = [Image.fromarray(raw[name][::-1, ::-1].copy()) for name in CAMERAS]
        geometry = {
            "depths": [raw[f"{name}_depth"][::-1, ::-1].copy() for name in CAMERAS],
            "intrinsics": [raw[f"{name}_intrinsic"] for name in CAMERAS],
            "extrinsics": [raw[f"{name}_extrinsic"] for name in CAMERAS],
            "eef": raw["eef_pos"],
        }
        return images, raw["gripper_qpos"], geometry


def validate_actions(actions: object) -> None:
    if not isinstance(actions, list) or not 1 <= len(actions) <= 50 or any(
        not isinstance(action, list) or len(action) != 7
        or not all(type(value) in (float, int) and math.isfinite(value) for value in action)
        for action in actions
    ):
        raise ValueError("expected 1 to 50 finite 7D actions")


def simulate(
    output: Path, roboenv: Path, seed: int, init_index: int,
    socket_path: Path | None, steps: int, scene: int, video_path: Path | None = None,
    extra_views: tuple[str, ...] = (), extra_view_frame: int | None = None, rgbd: bool = False,
    release_tail: bool = False, dense_release: bool = False, sampled_reset: bool = False,
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
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map

    suite = get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(scene - 1)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = suite.get_task_init_states(scene - 1)
    if not sampled_reset and not 0 <= init_index < len(states):
        raise ValueError(f"init index must be between 0 and {len(states) - 1}")
    encoder = None
    frames = 0
    sampling = ReleaseSampling() if dense_release else None

    def encode_frame(obs):
        if encoder is not None:
            frame = np.concatenate([obs[key][::-1, ::-1] for key in CAMERAS.values()], axis=1)
            encoder.stdin.write(frame.tobytes())

    def capture(obs, *, destination=None, **metadata):
        arrays = dict(
            **{name: obs[key] for name, key in CAMERAS.items()},
            eef_pos=obs["robot0_eef_pos"],
            eef_quat=obs["robot0_eef_quat"],
            gripper_qpos=obs["robot0_gripper_qpos"],
        )
        extra = {}
        if extra_views and (extra_view_frame is None or extra_view_frame == frames):
            extra = {
                name: env.env.sim.render(camera_name=name, width=512, height=512)[::-1].copy()
                for name in extra_views
            }
        sensors = {}
        if rgbd:
            sim = env.env.sim
            for name, key in CAMERAS.items():
                camera = key.removesuffix("_image")
                sensors[f"{name}_depth"] = get_real_depth_map(sim, obs[f"{camera}_depth"]).squeeze(-1)
                sensors[f"{name}_intrinsic"] = get_camera_intrinsic_matrix(sim, camera, 256, 256)
                sensors[f"{name}_extrinsic"] = get_camera_extrinsic_matrix(sim, camera)
        np.savez(
            destination or output / "observation.npz", **arrays, **sensors,
            **{f"visual_{name}": view for name, view in extra.items()},
        )
        return {
            **metadata,
            "sim_time": float(env.env.sim.data.time),
            "sim_state_sha256": hashlib.sha256(env.get_sim_state().tobytes()).hexdigest(),
            "observation_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in arrays.items()},
            **({"geometry_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in sensors.items()}} if sensors else {}),
            **({"extra_view_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in extra.items()}} if extra else {}),
            "audit_goals": [
                {"predicate": goal, "satisfied": bool(env.env._eval_predicate(goal))}
                for goal in env.env.parsed_problem["goal_state"]
            ],
        }

    def advance(actions):
        nonlocal frames
        observations = []
        for executed, action in enumerate(actions, start=1):
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
            frames += 1
            encode_frame(obs)
            success = bool(env.check_success())
            dense = False
            if sampling is not None:
                dense = sampling.observe(action[-1]) or (
                    sampling.closed >= 3 and frames % 10 == 5
                    and obs["robot0_eef_pos"][1] < 0.195 and obs["robot0_eef_pos"][2] > 0.50
                )
            if success or done:
                break
            if dense and executed < len(actions):
                observations.append(capture(obs, destination=output / f"release-{frames}.npz", step=frames))
        result = capture(obs, executed=executed, success=success, done=bool(done))
        if observations:
            result["release_observations"] = observations
        return result

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, control_freq=20,
        camera_depths=rgbd,
    )
    try:
        env.reset()
        env.seed(seed)
        env.reset()
        if not sampled_reset:
            env.set_init_state(states[init_index])
        initial_state_hash = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
        if sampled_reset and initial_state_hash in {hashlib.sha256(np.asarray(state).tobytes()).hexdigest() for state in states}:
            raise RuntimeError("sampled reset collides with a fixed benchmark initial state")
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
            init_state_index=None if sampled_reset else init_index,
            init_state_count=len(states),
            success=bool(env.check_success()),
            init_state_sha256=initial_state_hash,
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            **({"initialization": "libero_seeded_reset"} if sampled_reset else {}),
        )
        if sampled_reset and any(goal["satisfied"] for goal in initial["audit_goals"]):
            raise RuntimeError("sampled reset already satisfies a placement goal after settling")
        if video_path is not None:
            encoder = subprocess.Popen([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", "512x256", "-framerate", "20",
                "-i", "pipe:0", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video_path),
            ], stdin=subprocess.PIPE)
            encode_frame(obs)
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
                if release_tail and result["success"] and actions[result["executed"] - 1][-1] < -0.5:
                    tail = output / "release-tail"
                    tail.mkdir()
                    diagnostics = []
                    for offset in range(1, 21):
                        obs, _, done, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                        if offset % 5 == 0:
                            diagnostics.append(capture(
                                obs, destination=tail / f"frame-{offset}.npz", offset=offset,
                                success=bool(env.check_success()), done=bool(done),
                            ))
                    (tail / "observations.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
                break
    finally:
        try:
            if encoder is not None:
                try:
                    encoder.stdin.close()
                finally:
                    if encoder.wait() != 0:
                        raise RuntimeError("FFmpeg failed to encode the replay")
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
        "task_id": f"libero/libero-10-{args.scene:02d}",
        "method": args.method,
        "seed": args.seed,
        "init_state_index": args.init_index,
        "step_budget": args.steps,
        "steps": 0,
        "vla_calls": 0,
        "visual_calls": 0,
        "success": None,
        "terminal_reason": "runtime_error",
    }
    torch.manual_seed(args.seed)
    processor = LiberoProcessorStep()
    worker_command = [
        str(args.sim_python), str(Path(__file__).resolve()), "--sim-worker",
        "--roboenv", str(args.roboenv), "--output", str(output),
        "--seed", str(args.seed), "--init-index", str(args.init_index), "--scene", str(args.scene),
    ]
    if args.sampled_reset:
        worker_command.append("--sampled-reset")
    visual_provider = None
    visual_path = None
    supplemental_camera = None
    use_rgbd = False
    dense_release = False
    history = None
    confirmed_frames = {}
    invalidated_through = {}
    scheduled_methods = args.fair_methods or args.vision_runtime_probe
    if scheduled_methods and args.visual_validation is not None:
        sys.path.insert(0, str(PROJECT / "src"))
        from native_method_schedule import accepted_visual_provider
        from native_rgbd_placement import RGBD_RELEASE, RGBD_STRATEGIES

        visual_path, visual_provider = accepted_visual_provider(
            args.visual_validation, PROJECT, args.workspace, runtime_probe=args.vision_runtime_probe,
        )
        supplemental_camera = visual_provider["supplemental_camera"]
        if supplemental_camera:
            worker_command.extend(("--extra-view", supplemental_camera))
        use_rgbd = visual_provider["strategy"] in RGBD_STRATEGIES
        if use_rgbd:
            worker_command.append("--rgbd")
        dense_release = args.method == "closed_loop_vla" and visual_provider["strategy"] == RGBD_RELEASE
        if dense_release:
            worker_command.append("--dense-release")
    if scheduled_methods:
        record["supplemental_camera"] = supplemental_camera
        record["rgbd"] = use_rgbd

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
            supplemental = obs[f"visual_{supplemental_camera}"] if supplemental_camera else None
            geometry = {
                "depths": [obs[f"{name}_depth"][::-1, ::-1].copy() for name in CAMERAS],
                "intrinsics": [obs[f"{name}_intrinsic"] for name in CAMERAS],
                "extrinsics": [obs[f"{name}_extrinsic"] for name in CAMERAS],
                "eef": obs["eef_pos"],
            } if use_rgbd and args.method == "closed_loop_vla" else None
        batch = processor.observation(batch)
        if supplemental is not None:
            batch["visual_rgb"] = supplemental
        if geometry is not None:
            batch["visual_geometry"] = geometry
        if image_label:
            for name in CAMERAS:
                rgb = batch[f"observation.images.{name}"][0].permute(1, 2, 0)
                Image.fromarray(rgb.mul(255).round().byte().numpy()).save(output / f"{image_label}-{name}.png")
            if supplemental is not None:
                Image.fromarray(supplemental).save(output / f"{image_label}-{supplemental_camera}.png")
        return {**batch, "task": instruction}

    def receive(worker):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("LIBERO worker closed its response pipe; see simulator.log")
        return json.loads(line)

    def record_visual(observations, simulator, stage):
        # The predicates are written to evidence after inference and never enter the prompt.
        record["visual_calls"] += sum(observation.get("inferences", 1) for observation in observations)
        with (output / "visual.jsonl").open("a", encoding="utf-8") as log:
            for observation in observations:
                log.write(json.dumps({
                    **observation, "stage": stage, "frame": record["steps"],
                    "audit_goals": simulator["audit_goals"],
                    **({"geometry_sha256": simulator["geometry_sha256"]} if use_rgbd else {}),
                    **({"extra_view_sha256": simulator["extra_view_sha256"]} if supplemental_camera else {}),
                }) + "\n")

    def observe_history(observation, simulator, commands):
        start = record["steps"] - len(commands)
        last = 0
        for item in simulator.get("release_observations", []):
            step = item["step"]
            path = output / f"release-{step}.npz"
            images, qpos, geometry = read_rgbd_frame(path)
            history.observe(step, images, commands[last:step - start],
                            [item["observation_sha256"][name] for name in CAMERAS], qpos, geometry)
            path.unlink()
            last = step - start
        history.observe(
            record["steps"], planner.image_views(observation), commands[last:],
            [simulator["observation_sha256"][name] for name in CAMERAS],
            observation["observation.state"][0, -2:].tolist(),
            observation.get("visual_geometry"),
        )

    def inspect_target(observation, target_id):
        if history is None:
            return planner.inspect(observation, target_id)
        label = planner.controller.state.get(target_id).label
        decision = history.inspect(
            planner, target_id, label, planner.strategy,
            invalidated_through=invalidated_through.get(target_id, -1),
        )
        if decision["confirmed_frame"] is not None:
            confirmed_frames[target_id] = decision["confirmed_frame"]
        return decision

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
                planner = None
                schedule = None
                if scheduled_methods or args.method != "direct_vla":
                    import tomllib

                    sys.path.insert(0, str(PROJECT / "src"))
                    from native_visual import VisualPlanner

                    with (PROJECT / "configs" / "v1_eval.toml").open("rb") as settings:
                        scenes = tomllib.load(settings)["scenes"]
                    labels = next(scene["objects"] for scene in scenes if scene["source_task"] == record["task_id"])
                    if scheduled_methods:
                        from native_method_schedule import MethodSchedule
                        from tableclear.state import TaskState

                        state = TaskState.from_labels(record["task_id"], labels)
                        if args.method == "closed_loop_vla":
                            from native_temporal_vision import TEMPORAL_STRATEGIES, TemporalVision
                            from native_visual_references import load_references

                            references = visual_provider["references"]
                            planner = VisualPlanner(
                                visual_path, record["task_id"], labels, fixed=False, strategy=visual_provider["strategy"],
                                min_basket_margin=visual_provider["min_basket_margin"],
                                references=load_references(args.visual_validation.parent, references) if references else None,
                            )
                            if visual_provider.get("logo_templates"):
                                from native_planar_identity import load_logo

                                planner.held_logo = load_logo(args.visual_validation.parent, visual_provider["logo_templates"])
                            record["vision"] = visual_provider
                            if planner.strategy in (*TEMPORAL_STRATEGIES, *RGBD_STRATEGIES):
                                history = TemporalVision()
                                observe_history(obs, initial, [])
                            record_visual(planner.inventory(obs, inspect=inspect_target), initial, "inventory")
                            state = planner.controller.state
                        schedule = MethodSchedule(
                            args.method, state, args.steps, args.chunk_steps,
                            compact_prompt=args.diagnostic_compact_prompt, attempt_chunks=args.diagnostic_attempt_chunks,
                        )
                        record["chunk_horizon"] = args.chunk_steps
                        record["method_protocol"] = (
                            "vision-runtime-probe-v1" if args.vision_runtime_probe else "shared-action-budget-v1"
                        )
                        if args.diagnostic_compact_prompt or args.diagnostic_attempt_chunks != 13:
                            record["method_protocol"] = "diagnostic-schedule-ablation-v1"
                            record["schedule_ablation"] = {
                                "compact_prompt": args.diagnostic_compact_prompt, "attempt_chunks": args.diagnostic_attempt_chunks,
                            }
                    else:
                        base = next(a for a in record["model_artifacts"] if a["name"] == "smolvlm2_500m_video_instruct")
                        planner = VisualPlanner(
                            (artifact_root / base["path"]).resolve(), record["task_id"], labels,
                            fixed=args.method == "fixed_checklist_vla",
                        )
                        record_visual(planner.inventory(obs), initial, "inventory")
                service = None
                horizon = args.chunk_steps or (10 if planner else None)
                record["final_sim_time"] = initial["sim_time"]
                record["terminal_reason"] = "simulator_success" if initial["success"] else "step_budget"
                while not record["success"] and record["steps"] < args.steps:
                    target = None
                    if schedule:
                        planned = schedule.action(record["steps"], initial["instruction"])
                        if planned is None:
                            record["terminal_reason"] = "visual_clear"
                            break
                        target, instruction, count = planned
                    elif planner:
                        subtask = planner.next_subtask()
                        if subtask is None:
                            record["terminal_reason"] = "checklist_end" if planner.fixed else (
                                "visual_clear" if planner.controller.is_cleared() else "attempt_limit"
                            )
                            break
                        target = subtask.target_id
                        instruction = subtask.instruction
                    if service is None:
                        service = open_service(config, artifact_root=artifact_root)
                        horizon = horizon or service["policy"].config.n_action_steps
                        record["chunk_horizon"] = horizon
                    if schedule is None:
                        count = min(horizon, args.steps - record["steps"])
                    policy_obs = {key: value for key, value in obs.items() if key not in ("visual_rgb", "visual_geometry")}
                    actions = predict_chunk(service, {**policy_obs, "task": instruction})[:count]
                    worker.stdin.write(json.dumps(actions, allow_nan=False) + "\n")
                    worker.stdin.flush()
                    result = receive(worker)
                    executed = result["executed"]
                    if not 1 <= executed <= len(actions):
                        raise RuntimeError(f"invalid executed frame count: {executed}")
                    trace.write(json.dumps({
                        "start_step": record["steps"], "actions": actions[:executed],
                        "instruction": instruction, "target_id": target, **result,
                    }) + "\n")
                    record["steps"] += executed
                    record["vla_calls"] += 1
                    record["success"] = result["success"]
                    record["final_sim_time"] = result["sim_time"]
                    record["audit_goals"] = result["audit_goals"]
                    obs = read_observation(instruction)
                    release_ready = False
                    if history is not None:
                        chunk_start = record["steps"] - executed
                        observe_history(obs, result, [action[-1] for action in actions[:executed]])
                        release_ready = any(after.step > chunk_start for _, _, after in history.release_windows(rgbd=use_rgbd))
                        for completed_target, confirmed in list(confirmed_frames.items()):
                            if not history.confirmation_valid(confirmed, rgbd=use_rgbd):
                                del confirmed_frames[completed_target]
                                invalidated_through[completed_target] = record["steps"]
                                schedule.reopen(completed_target)
                                record_visual([{
                                    "target_id": completed_target, "target": schedule.state.get(completed_target).label,
                                    "status": "unknown", "raw": "basket_changed_since_confirmation",
                                    "parsed": True, "inferences": 0, "previous_confirmation": confirmed,
                                }], result, "recheck")
                    if schedule and planner and (release_ready or schedule.needs_visual_update(
                        record["steps"], result["success"] or result["done"],
                    )):
                        decision = inspect_target(obs, target)
                        record_visual([decision], result, "progress")
                        schedule.apply_visual(decision["status"], record["steps"])
                    elif schedule is None and planner and not planner.fixed:
                        record_visual([planner.update(obs, target)], result, "progress")
                    print(f"steps={record['steps']} simulator_success={record['success']}", flush=True)
                    if result["success"] or result["done"] or record["steps"] == args.steps:
                        record["terminal_reason"] = (
                            "simulator_success" if result["success"]
                            else "simulator_done" if result["done"] else "step_budget"
                        )
                        break
                read_observation(instruction, "final")
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
        if torch.cuda.is_initialized():
            record["cuda_peak_reserved_mib"] = math.ceil(torch.cuda.max_memory_reserved() / (1024 * 1024))
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
    parser.add_argument("--scene", type=int, choices=(1, 2, 8), default=1)
    parser.add_argument("--chunk-steps", type=int, choices=range(1, 51), metavar="1..50")
    parser.add_argument("--method", choices=("direct_vla", "fixed_checklist_vla", "closed_loop_vla"), default="direct_vla")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--fair-methods", action="store_true")
    protocol.add_argument("--vision-runtime-probe", action="store_true")
    parser.add_argument("--visual-validation", type=Path)
    parser.add_argument("--sim-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--socket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--video", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--extra-view", action="append", choices=EXTRA_VIEWS, default=[], help=argparse.SUPPRESS)
    parser.add_argument("--extra-view-frame", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rgbd", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--release-tail", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dense-release", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sampled-reset", action="store_true", help="use LIBERO's seeded reset for visual validation collection")
    parser.add_argument("--diagnostic-compact-prompt", action="store_true")
    parser.add_argument("--diagnostic-attempt-chunks", type=int, choices=(13, 26), default=13)
    args = parser.parse_args()
    if not 1 <= args.steps <= 520:
        parser.error("--steps must be between 1 and 520")
    if args.sampled_reset and (args.fair_methods or args.vision_runtime_probe or args.socket is not None
                               or args.method != "direct_vla" or args.init_index != 0):
        parser.error("sampled resets are only for separate visual validation collection")
    if args.sim_worker:
        if args.video is not None and args.socket is not None:
            parser.error("video recording requires a standalone replay worker")
        if args.release_tail and (not args.rgbd or args.socket is not None or args.video is not None):
            parser.error("diagnostic release observations require a standalone RGB-D replay")
        if args.dense_release and (not args.rgbd or args.socket is not None or args.video is not None):
            parser.error("dense release observations require the standalone RGB-D worker")
        simulate(
            args.output, args.roboenv, args.seed, args.init_index, args.socket, args.steps, args.scene,
            args.video, tuple(dict.fromkeys(args.extra_view)), args.extra_view_frame, args.rgbd, args.release_tail,
            args.dense_release, args.sampled_reset,
        )
        return
    if args.workspace is None or args.sim_python is None or args.socket is not None or args.video is not None:
        parser.error("--workspace and --sim-python are required for a direct episode")
    if args.fair_methods and (args.chunk_steps != 10 or args.steps != 520):
        parser.error("the paired method protocol uses 10-step chunks and 520 available actions")
    if args.diagnostic_compact_prompt or args.diagnostic_attempt_chunks != 13:
        if (not args.fair_methods or args.init_index < 15 or args.method == "direct_vla"
                or (args.diagnostic_attempt_chunks != 13 and args.method != "closed_loop_vla")):
            parser.error("schedule ablations require a diagnostic index and a matching checklist method")
    if args.fair_methods and args.method == "closed_loop_vla" and args.visual_validation is None:
        parser.error("--visual-validation is required for the paired closed-loop method")
    if args.vision_runtime_probe and (
        args.method != "closed_loop_vla" or args.visual_validation is None
        or args.chunk_steps != 10 or args.steps > 20 or args.init_index < 15
    ):
        parser.error("the vision runtime probe requires a diagnostic initial state, a visual report, and at most 20 steps")
    print(json.dumps(episode(args), indent=2))


if __name__ == "__main__":
    main()
