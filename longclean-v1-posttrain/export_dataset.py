"""Write lossless 20 Hz LeRobot datasets from verified V1 demonstrations."""

from argparse import ArgumentParser
from pathlib import Path
import tempfile

import numpy as np

from common import CAMERA_KEYS, FPS, STATE_ENCODING, array_sha, file_sha, read_json, write_json


FEATURES = {
    **{name: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
       for name in CAMERA_KEYS},
    "observation.state": {
        "dtype": "float32", "shape": (8,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "finger1", "finger2"],
    },
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}


def repo_id(split):
    return f"tableclear/v1-scene08-{split}"


def pixel_array(image):
    pixels = image.permute(1, 2, 0).contiguous().numpy()
    if pixels.dtype == np.uint8:
        return pixels
    # The pinned HF image-column reader normalizes PNGs to float32. The
    # return_uint8 option applies to video frames only.
    if pixels.dtype != np.float32 or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Expected uint8 RGB or normalized float32 RGB")
    quantized = np.rint(pixels * 255).astype(np.uint8)
    if not np.array_equal(pixels, quantized.astype(np.float32) / 255):
        raise ValueError("Image column contains values outside the expected 8-bit RGB grid")
    return quantized


def build_split(root, replay, split, cases):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = root / split
    writer = LeRobotDataset.create(
        repo_id=repo_id(split), root=path, fps=FPS, robot_type="franka_panda",
        features=FEATURES, use_videos=False, image_writer_threads=2,
    )
    copies = []
    sampled = []
    cursor = 0
    try:
        for episode_index, case in enumerate(cases):
            folder = replay / case["id"]
            if (file_sha(folder / "dense.npz") != case["dense_sha256"]
                    or file_sha(folder / "actions.npy") != case["native_actions_sha256"]):
                raise ValueError(f"Replay checksum mismatch for {case['id']}")
            actions = np.load(folder / "actions.npy", allow_pickle=False)
            with np.load(folder / "dense.npz", allow_pickle=False) as snapshot:
                front, wrist, states = (snapshot[key] for key in ("front", "wrist", "state"))
            for frame_index, action in enumerate(actions):
                frame = {
                    CAMERA_KEYS[0]: front[frame_index],
                    CAMERA_KEYS[1]: wrist[frame_index],
                    "observation.state": states[frame_index],
                    "action": action,
                    "task": case["task"],
                }
                if frame_index in (0, len(actions) - 1):
                    sample = {key: array_sha(value) for key, value in frame.items() if key != "task"}
                    sampled.append((cursor + frame_index, episode_index, case["task"], sample))
                writer.add_frame(frame)
            writer.save_episode()
            copies.append({**case, "episode_index": episode_index, "first_index": cursor})
            cursor += len(actions)
            print("EXPORT_OK", split, case["id"], cursor, flush=True)
    finally:
        writer.finalize()
    del writer

    reader = LeRobotDataset(repo_id(split), root=path, download_videos=False, return_uint8=True)
    if (len(reader), reader.num_episodes, reader.fps) != (cursor, len(copies), FPS):
        raise ValueError("LeRobot dataset length, episode count or fps differs")
    for index, episode_index, task, hashes in sampled:
        frame = reader[index]
        if frame["episode_index"].item() != episode_index or frame["task"] != task:
            raise ValueError(f"Data identity mismatch at local row {index}")
        for key, expected in hashes.items():
            value = pixel_array(frame[key]) if key in CAMERA_KEYS else frame[key].numpy()
            if array_sha(value) != expected:
                raise ValueError(f"Lossless data roundtrip failed at row {index}: {key}")
    return {"episodes": copies, "rows": cursor, "roundtrip_checks": len(sampled)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "replay-manifest.json"
    expected = read_json(source)
    if (len(expected["episodes"]) != 120 or expected["rejected"]
            or any(case.get("state_encoding") != STATE_ENCODING for case in expected["episodes"])):
        raise ValueError("Complete audited historical replay is required")
    final = args.run / "dataset"
    if final.is_dir():
        if read_json(final / "build-manifest.json")["source_sha256"] != file_sha(source):
            raise ValueError("Existing dataset has a different replay manifest")
        return
    attempts = args.run / "export-attempts"
    attempts.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="v3-", dir=attempts))
    manifest = {"source_sha256": file_sha(source), "fps": FPS, "splits": {}}
    for split in ("train", "validation"):
        cases = [case for case in expected["episodes"] if case["split"] == split]
        manifest["splits"][split] = build_split(stage, args.run / "replay", split, cases)
    manifest["sha256"] = {
        path.relative_to(stage).as_posix(): file_sha(path) for path in sorted(stage.rglob("*")) if path.is_file()
    }
    write_json(stage / "build-manifest.json", manifest)
    stage.rename(final)


if __name__ == "__main__":
    main()
