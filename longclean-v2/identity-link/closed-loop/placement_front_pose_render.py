"""Resample front RGB-D at 512 px while exactly replaying the three known trajectories."""

import argparse
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from actual_physics_simulator import load_native
from near_table_placement_pair import compare_native
from placement_orientation_feedback import DEV
from soup_placement_feasibility_v2 import pinned, read_lines
from stage4_fixture import require, validate_study
from common import CAMERAS, array_hash, read, sha, write


RESOLUTION = 512


def resample(scene, original, state, folder):
    capture = scene.capture()
    require(capture["state_sha256"] == state["state_sha256"] == original["state_sha256"],
            "Archived frame and physical state did not match")
    require(capture["observation_sha256"] == original["observation_sha256"],
            f"Original-resolution frame did not reproduce at step {scene.frame}")
    sim = scene.env.env.sim
    camera = CAMERAS["camera1"].removesuffix("_image")
    utils = scene.camera_utils
    with np.load(scene.output / "observation.npz", allow_pickle=False) as obs:
        at_size = {name: obs[name] for name in ("eef_pos", "eef_quat", "gripper_qpos", "camera2_extrinsic")}
        raw_rgb, raw_depth = sim.render(width=256, height=256, camera_name=camera, depth=True)
        require(np.array_equal(raw_rgb, obs["camera1"])
                and np.array_equal(utils.get_real_depth_map(sim, raw_depth), obs["camera1_depth"]),
                "Direct rendering differs from the archived camera convention")
    rgb, depth = sim.render(width=RESOLUTION, height=RESOLUTION, camera_name=camera, depth=True)
    at_size.update(camera1=rgb, camera1_depth=utils.get_real_depth_map(sim, depth),
                   camera1_intrinsic=utils.get_camera_intrinsic_matrix(sim, camera, RESOLUTION, RESOLUTION),
                   camera1_extrinsic=utils.get_camera_extrinsic_matrix(sim, camera))
    require(rgb.shape == (RESOLUTION, RESOLUTION, 3) and depth.shape == (RESOLUTION, RESOLUTION)
            and array_hash(scene.env.get_sim_state()) == state["state_sha256"], "Hi-res rendering changed the state")
    output = folder / "frames" / f"{scene.frame}.npz"
    np.savez_compressed(output, **at_size)
    return {"step": scene.frame, "state_sha256": state["state_sha256"], "file": output.relative_to(folder).as_posix(),
            "sha256": sha(output), "original_frame_sha256": original["sha256"],
            "original_observation_sha256": capture["observation_sha256"]}


def replay(native, bddl, seed, old, sources, folder):
    result_path = pinned(old / "result.json", sources["fixed"])
    result = read(result_path)
    inputs = {name: pinned(old / name, digest) for name, digest in result["file_sha256"].items()}
    trace = read_lines(inputs["physics.jsonl"])
    start, end = trace[0]["step"], next(row["end_step"] for row in result["phases"] if row["phase"] == "lower")
    prefix = read_lines(inputs["replayed-actions.jsonl"])
    require(prefix[-1]["step"] == start and [row["step"] for row in trace] == list(range(start, trace[-1]["step"] + 1)),
            "Source native action interval is incomplete")
    originals = {frame["step"]: frame for frame in result["frames"] if start <= frame["step"] <= end}
    require(start in originals and end in originals, "Both RGB-D endpoints are needed")
    for frame in originals.values():
        pinned(old / frame["file"], frame["sha256"])
    (folder / "frames").mkdir(parents=True)
    snapshots = []
    with TemporaryDirectory(prefix="sim-observation-", dir=folder) as scratch:
        scene = native.Simulator(bddl, Path(scratch), seed, False)
        try:
            for expected in prefix:
                require(scene.frame == expected["start_step"], "Changed action boundary")
                actual = scene.advance(expected["actions"])
                compare_native({**expected, **actual, "actions": expected["actions"][:actual["executed"]]}, expected, [], False)
            snapshots.append(resample(scene, originals[start], trace[0], folder))
            for state in trace[1:end - start + 1]:
                action = np.asarray(state["action"], dtype=np.float32)
                scene.obs, _, done, _ = scene.env.step(action)
                scene.frame += 1
                require(scene.frame == state["step"] and bool(done) == state["environment_done"]
                        and array_hash(scene.env.get_sim_state()) == state["state_sha256"],
                        f"Changed native dynamics at step {scene.frame}")
                if scene.frame in originals:
                    snapshots.append(resample(scene, originals[scene.frame], state, folder))
        finally:
            scene.close()
    require([frame["step"] for frame in snapshots] == sorted(originals), "Hi-res archive missed an RGB-D checkpoint")
    for name, path in inputs.items():
        copyfile(path, folder / name)
    output = folder / "result.json"
    write(output, {key: result[key] for key in ("target_m", "phases", "opening", "file_sha256")}
          | {"frames": snapshots, "resolution": RESOLUTION, "original_result_sha256": sha(result_path)})
    print("RENDER_REPLAY_OK", seed, len(prefix), end - start, len(snapshots), sha(output), flush=True)
    return {"fixed": sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
            and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin a free render device before starting")
    bddl, _ = validate_study(args.study.resolve())
    pair_file = args.source / "orientation-pairs.json"
    source = read(pair_file)
    require(set(source) == set(map(str, DEV)), "The resampling request must use the frozen development group")
    args.output.mkdir(parents=True, exist_ok=False)
    native = load_native()
    native.ensure_libero_config()
    index = {}
    for seed in sorted(DEV):
        path = f"pairs/{seed}/fixed"
        index[str(seed)] = replay(native, bddl, seed, args.source / path, source[str(seed)], args.output / path)
    write(args.output / "orientation-pairs.json", index)
    write(args.output / "render.json", {"source_index_sha256": sha(pair_file), "source_sha256": sha(Path(__file__)),
                                         "seeds": sorted(DEV), "width": RESOLUTION,
                                         "pose_source": "identical native OSC state replay"})


if __name__ == "__main__":
    main()
