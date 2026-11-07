"""Render and verify a saved native OSC episode as a two-camera MP4."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


OBSERVATION_FIELDS = ("sim_time", "sim_state_sha256", "observation_sha256")


def replay(source: Path, output: Path, sim_python: Path, roboenv: Path) -> dict:
    episode_path = source / "episode.json"
    actions_path = source / "actions.jsonl"
    episode_bytes = episode_path.read_bytes()
    action_bytes = actions_path.read_bytes()
    episode = json.loads(episode_bytes)
    trace = [json.loads(line) for line in action_bytes.splitlines()]
    if not trace or not isinstance(episode["success"], bool) or episode["terminal_reason"] == "runtime_error":
        raise ValueError("a completed episode with native action evidence is required")

    scene = int(episode["task_id"].rsplit("-", 1)[1])
    output.mkdir(parents=True, exist_ok=False)
    pending = (output / "replay.pending.mp4").resolve()
    command = [
        str(sim_python), str(Path(__file__).with_name("run_native_osc.py")), "--sim-worker",
        "--output", str(output), "--roboenv", str(roboenv), "--scene", str(scene),
        "--seed", str(episode["seed"]), "--init-index", str(episode["init_state_index"]),
        "--video", str(pending),
    ]

    def compare(worker, expected, fields, frame):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("replay worker closed its response pipe; see simulator.log")
        actual = json.loads(line)
        for name in fields:
            if actual[name] != expected[name]:
                raise RuntimeError(f"replay differs in {name} at frame {frame}")

    frames = 0
    with (output / "simulator.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True) as worker:
            try:
                compare(worker, episode, (*OBSERVATION_FIELDS, "init_state_sha256", "bddl_sha256", "control_hz"), 0)
                for row in trace:
                    if row["start_step"] != frames or len(row["actions"]) != row["executed"]:
                        raise ValueError(f"invalid action trace at frame {frames}")
                    worker.stdin.write(json.dumps(row["actions"], allow_nan=False) + "\n")
                    worker.stdin.flush()
                    frames += row["executed"]
                    compare(worker, row, (*OBSERVATION_FIELDS, "executed", "success", "done"), frames)
            finally:
                worker.stdin.close()
        if worker.returncode:
            raise RuntimeError("replay worker failed; see simulator.log")

    if frames != episode["steps"] or trace[-1]["success"] != episode["success"]:
        raise ValueError("source trace and terminal summary disagree")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,r_frame_rate,width,height", "-of", "json", str(pending),
    ], text=True))["streams"][0]
    if (int(probe["nb_frames"]), probe["r_frame_rate"], probe["width"], probe["height"]) != (frames + 1, "20/1", 512, 256):
        raise RuntimeError(f"encoded replay does not match the action trace: {probe}")

    video = output / "replay.mp4"
    pending.replace(video)
    manifest = {
        "source_episode": source.name, "project_commit": episode["project_commit"],
        "replay_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "task_id": episode["task_id"], "seed": episode["seed"], "init_state_index": episode["init_state_index"],
        "init_state_sha256": episode["init_state_sha256"], "success": episode["success"],
        "steps": frames, "video_frames": frames + 1, "fps": 20,
        "left_view": "agentview", "right_view": "robot0_eye_in_hand",
        "state_and_image_hashes_match_at_chunk_boundaries": True,
        "episode_sha256": sha256(episode_bytes).hexdigest(), "actions_sha256": sha256(action_bytes).hexdigest(),
        "video_sha256": sha256(video.read_bytes()).hexdigest(),
    }
    (output / "replay.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sim-python", required=True, type=Path)
    parser.add_argument("--roboenv", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.episode_dir, args.output, args.sim_python, args.roboenv)))
