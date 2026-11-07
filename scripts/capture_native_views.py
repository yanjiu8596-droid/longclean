"""Render existing LIBERO cameras from a hash-verified native action replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from replay_native_video import OBSERVATION_FIELDS
from run_native_osc import EXTRA_VIEWS


def capture(source: Path, output: Path, sim_python: Path, roboenv: Path, views: list[str], *, initial=False):
    import numpy as np
    from PIL import Image

    episode = json.loads((source / "episode.json").read_bytes())
    trace = [json.loads(line) for line in (source / "actions.jsonl").read_bytes().splitlines()]
    if not trace or not isinstance(episode["success"], bool):
        raise ValueError("the source must contain a completed action trace")
    output.mkdir(parents=True, exist_ok=False)
    frame = 0 if initial else episode["steps"]
    command = [
        str(sim_python), str(Path(__file__).with_name("run_native_osc.py")), "--sim-worker",
        "--output", str(output), "--roboenv", str(roboenv), "--scene", episode["task_id"].rsplit("-", 1)[1],
        "--seed", str(episode["seed"]), "--init-index", str(episode["init_state_index"]),
        "--extra-view-frame", str(frame),
    ]
    for view in views:
        command.extend(("--extra-view", view))

    def receive(expected, fields):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"simulator exited before its response, see {output / 'simulator.log'}")
        actual = json.loads(line)
        if any(actual[name] != expected[name] for name in fields):
            raise RuntimeError("replayed control or observation hashes differ")
        return actual

    with (output / "simulator.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True) as worker:
            try:
                latest = receive(episode, (*OBSERVATION_FIELDS, "init_state_sha256", "bddl_sha256"))
                if not initial:
                    for row in trace:
                        worker.stdin.write(json.dumps(row["actions"], allow_nan=False) + "\n")
                        worker.stdin.flush()
                        latest = receive(row, (*OBSERVATION_FIELDS, "executed", "success", "done"))
            finally:
                worker.stdin.close()
        if worker.returncode:
            raise RuntimeError(f"simulator replay failed, see {output / 'simulator.log'}")
    raw_path = output / "observation.npz"
    with np.load(raw_path, allow_pickle=False) as batch:
        for view in views:
            Image.fromarray(batch[f"visual_{view}"]).save(output / f"{view}.png")
    raw_path.unlink()
    result = {
        "source": source.name, "frame": frame, "sim_state_sha256": latest["sim_state_sha256"],
        "extra_view_sha256": latest["extra_view_sha256"], "init_state_sha256": episode["init_state_sha256"],
    }
    (output / "views.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "sim-python", "roboenv"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--view", action="append", choices=EXTRA_VIEWS, required=True)
    parser.add_argument("--initial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(capture(args.source, args.output, args.sim_python, args.roboenv, args.view, initial=args.initial)))
