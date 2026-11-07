"""Replay Kinex's submitted actions through the standalone native worker."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def verify(args):
    import numpy as np
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep
    from PIL import Image

    run = args.run
    kinex = run / "kinex-osc"
    initial = json.loads((kinex / "initial.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (kinex / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError("Kinex did not step the simulator")
    direct = run / "direct-replay"
    direct.mkdir()
    processor = LiberoProcessorStep()
    command = [
        str(args.sim_python), str(Path(__file__).with_name("run_native_osc.py")), "--sim-worker",
        "--output", str(direct), "--roboenv", str(args.roboenv),
        "--seed", str(initial["seed"]), "--init-index", str(initial["init_state_index"]),
    ]

    def compare(worker, expected, fields):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("direct replay worker exited; see direct-replay/simulator.log")
        actual = json.loads(line)
        for field in fields:
            if actual[field] != expected[field]:
                raise RuntimeError(f"frame {expected['frame']}: replay differs in {field}")
        if not math.isclose(actual["sim_time"], initial["sim_time"] + expected["frame"] / 20, abs_tol=1e-8):
            raise RuntimeError("simulator clock disagrees with the action count")
        with np.load(direct / "observation.npz", allow_pickle=False) as obs:
            state = processor.observation({"observation.robot_state": {
                "eef": {key: torch.from_numpy(obs[f"eef_{key}"]).unsqueeze(0) for key in ("pos", "quat")},
                "gripper": {"qpos": torch.from_numpy(obs["gripper_qpos"]).unsqueeze(0)},
            }})["observation.state"][0].numpy()
            for i, name in enumerate(("camera1", "camera2")):
                with Image.open(expected["raw_images"][i]) as raw, Image.open(expected["images"][i]) as display:
                    np.testing.assert_array_equal(np.asarray(raw), obs[name])
                    np.testing.assert_array_equal(np.asarray(display), obs[name][::-1, ::-1])
        np.testing.assert_allclose(state, expected["state"], atol=1e-6, rtol=0)
        return float(np.max(np.abs(state - expected["state"])))

    fields = ("sim_state_sha256", "observation_sha256", "sim_time")
    with (direct / "simulator.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True) as worker:
            try:
                state_error = compare(worker, initial, (*fields, "init_state_sha256", "bddl_sha256"))
                for row in rows:
                    worker.stdin.write(json.dumps(row["actions"], allow_nan=False) + "\n")
                    worker.stdin.flush()
                    state_error = max(state_error, compare(worker, row, (*fields, "success", "done", "executed")))
            finally:
                worker.stdin.close()
        if worker.returncode:
            raise RuntimeError("direct replay worker failed; see direct-replay/simulator.log")

    result = {
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip(),
        "passed": True,
        "frames": rows[-1]["frame"],
        "chunks": len(rows),
        "elapsed_sim_s": rows[-1]["sim_time"] - initial["sim_time"],
        "max_model_state_error": state_error,
        "native_success": rows[-1]["success"],
    }
    (run / "parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--roboenv", type=Path, required=True)
    verify(parser.parse_args())
