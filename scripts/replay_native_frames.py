"""Sample native OSC observations while checking every original action boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tomllib

from replay_native_video import OBSERVATION_FIELDS
from run_native_osc import CAMERAS
from validate_native_vision import PROJECT, panel


DEVELOPMENT = range(19, 40)
HOLDOUT = range(40, 50)


def replay(source: Path, output: Path, sim_python: Path, roboenv: Path, *, rgbd: bool = False, release_tail: bool = False) -> None:
    import numpy as np

    episode = json.loads((source / "episode.json").read_bytes())
    trace = [json.loads(line) for line in (source / "actions.jsonl").read_bytes().splitlines()]
    if not trace or episode["steps"] < 1 or type(episode["success"]) is not bool:
        raise ValueError(f"incomplete source trajectory: {source}")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(sim_python), str(Path(__file__).with_name("run_native_osc.py")), "--sim-worker",
        "--output", str(output), "--roboenv", str(roboenv), "--scene", episode["task_id"].rsplit("-", 1)[1],
        "--seed", str(episode["seed"]), "--init-index", str(episode["init_state_index"]),
    ]
    if rgbd:
        command.append("--rgbd")
    if release_tail:
        command.append("--release-tail")
    frame_path = output / "observation.npz"
    frames, images, gripper_qpos, gripper_commands, image_hashes = [], [], [], [], []
    depths, intrinsics, extrinsics, eef_positions = [], [], [], []

    def verify(actual, expected, fields):
        if any(actual[field] != expected[field] for field in fields):
            raise ValueError(f"replay differs at step {step}: {source}")

    def receive(worker):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"simulator did not respond; see {output / 'simulator.log'}")
        return json.loads(line)

    def sample(metadata, path=frame_path):
        with np.load(path, allow_pickle=False) as raw:
            frames.append(step)
            images.append([raw[camera][::-1, ::-1].copy() for camera in CAMERAS])
            gripper_qpos.append(raw["gripper_qpos"])
            image_hashes.append([metadata["observation_sha256"][camera] for camera in CAMERAS])
            if rgbd:
                depths.append([raw[f"{camera}_depth"][::-1, ::-1].copy() for camera in CAMERAS])
                intrinsics.append([raw[f"{camera}_intrinsic"] for camera in CAMERAS])
                extrinsics.append([raw[f"{camera}_extrinsic"] for camera in CAMERAS])
                eef_positions.append(raw["eef_pos"])

    step = 0
    with (output / "simulator.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True) as worker:
            try:
                initial = receive(worker)
                verify(initial, episode, (*OBSERVATION_FIELDS, "init_state_sha256", "bddl_sha256"))
                sample(initial)
                for row in trace:
                    actions = row["actions"]
                    if row["start_step"] != step or row["executed"] != len(actions):
                        raise ValueError(f"source action index is invalid: {source}")
                    offset = 0
                    while offset < len(actions):
                        count = min(10 - step % 10, len(actions) - offset)
                        segment = actions[offset:offset + count]
                        worker.stdin.write(json.dumps(segment, allow_nan=False) + "\n")
                        worker.stdin.flush()
                        current = receive(worker)
                        if current["executed"] != count:
                            raise ValueError(f"unexpected simulator termination at step {step}: {source}")
                        gripper_commands.extend(action[-1] for action in segment)
                        offset += count
                        step += count
                        if step % 10 == 0 or step == episode["steps"]:
                            sample(current)
                    verify(current, row, (*OBSERVATION_FIELDS, "success", "done", "audit_goals"))
            finally:
                worker.stdin.close()
        if worker.returncode:
            raise RuntimeError(f"replay failed; see {output / 'simulator.log'}")
    if step != episode["steps"] or current["success"] != episode["success"]:
        raise ValueError(f"replay terminal result disagrees with {source}")
    official_step = step
    tail = output / "release-tail"
    tail_records = []
    if tail.is_dir():
        tail_records = json.loads((tail / "observations.json").read_bytes())
        for record in tail_records:
            offset = record["offset"]
            step = official_step + offset
            sample(record, tail / f"frame-{offset}.npz")
        gripper_commands.extend([-1.0] * (step - official_step))
    np.savez_compressed(
        output / "frames.npz", steps=frames, images=images, gripper_qpos=gripper_qpos,
        gripper_commands=gripper_commands, image_hashes=image_hashes,
        **(dict(depths=depths, intrinsics=intrinsics, extrinsics=extrinsics, eef_positions=eef_positions) if rgbd else {}),
    )
    frame_path.unlink()
    (output / "verified.json").write_text(json.dumps({
        "source": source.name, "steps": official_step, "sampled_frames": len(frames),
        "init_state_sha256": episode["init_state_sha256"], "bddl_sha256": episode["bddl_sha256"],
        "terminal_state_sha256": current["sim_state_sha256"], "official_success": episode["success"],
        "rgbd": rgbd,
        **({"diagnostic_tail": tail_records} if tail_records else {}),
        "actions_sha256": sha256((source / "actions.jsonl").read_bytes()).hexdigest(),
        "frames_sha256": sha256((output / "frames.npz").read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replayed": source.name, "sampled_frames": len(frames)}), flush=True)


def replay_panel(cases, output: Path, sim_python: Path, roboenv: Path, workers: int, *, rgbd: bool = False) -> None:
    sources = {folder.name: folder for samples in cases.values() for _, folder in samples.values()}
    output.mkdir(parents=True, exist_ok=True)

    def run(folder):
        destination = output / folder.name
        if (destination / "verified.json").is_file():
            previous = json.loads((destination / "verified.json").read_bytes())
            if bool(previous.get("rgbd")) != rgbd:
                raise ValueError(f"replay observation mode differs: {destination}")
            return
        staging = destination.with_name(destination.name + ".partial")
        while staging.exists():
            staging = staging.with_suffix(staging.suffix + "-retry")
        replay(folder, staging, sim_python, roboenv, rgbd=rgbd)
        staging.rename(destination)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(run, sources.values()):
            pass


def scene_labels() -> dict:
    with (PROJECT / "configs" / "v1_eval.toml").open("rb") as handle:
        return {int(s["source_task"].rsplit("-", 1)[1]): s["objects"] for s in tomllib.load(handle)["scenes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "sim-python", "roboenv"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--episode-dir", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=8)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--release-tail", action="store_true")
    args = parser.parse_args()
    if args.release_tail and (args.episode_dir is None or not args.rgbd):
        parser.error("--release-tail requires --episode-dir and --rgbd")
    if args.episode_dir is not None:
        replay(args.episode_dir, args.output, args.sim_python, args.roboenv, rgbd=args.rgbd, release_tail=args.release_tail)
    else:
        calibration = panel(args.run, scene_labels(), DEVELOPMENT, 2)
        replay_panel(calibration, args.output, args.sim_python, args.roboenv, args.workers, rgbd=args.rgbd)
