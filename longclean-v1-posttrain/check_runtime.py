"""Verify the pinned V1 training imports and image dataset roundtrip on CPU."""

from importlib.metadata import version
import json
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory

import lerobot
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from peft import LoraConfig, get_peft_model

from common import DEFAULT_LAB, LEROBOT_REVISION, Lab, file_sha
from export_dataset import build_split, repo_id


def verify_image_dataset(root):
    episode = root / "probe"
    episode.mkdir()
    front = np.random.default_rng(18).integers(0, 256, (5, 256, 256, 3), dtype=np.uint8)
    actions = np.arange(28, dtype=np.float32).reshape(4, 7) / 28
    np.save(episode / "actions.npy", actions, allow_pickle=False)
    np.savez(episode / "dense.npz", front=front, wrist=front[..., ::-1], state=np.zeros((5, 8), np.float32))
    results = build_split(root, root, "train", [{
        "id": "probe",
        "task": "put the alphabet soup in the basket",
        "dense_sha256": file_sha(episode / "dense.npz"),
        "native_actions_sha256": file_sha(episode / "actions.npy"),
    }])
    ds = LeRobotDataset(repo_id("train"), root=root / "train", download_videos=False, return_uint8=True)
    tabular = np.stack(ds.hf_dataset.select_columns(["action"])[:]["action"]).astype(np.float32)
    np.testing.assert_array_equal(tabular, actions)
    return {"rows": results["rows"], "replayed_boundaries": results["roundtrip_checks"]}


def main():
    checkout = Path(lerobot.__file__).resolve().parents[2]
    commit = check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != LEROBOT_REVISION:
        raise ValueError(f"Expected LeRobot {LEROBOT_REVISION}, got {commit}")
    versions = {
        package: version(package)
        for package in ("torch", "transformers", "lerobot", "peft", "datasets", "pyarrow", "pandas")
    }
    model = Lab(DEFAULT_LAB).model
    files = [model / "config.json", model / "tokenizer/tokenizer.json",
             *sorted(model.glob("policy_preprocessor*")), *sorted(model.glob("policy_postprocessor*"))]
    processors = {path.relative_to(model).as_posix(): file_sha(path) for path in files}
    with TemporaryDirectory(prefix="tableclear-dataset-check-") as tmp:
        io = verify_image_dataset(Path(tmp))
    print(json.dumps({
        "lerobot_revision": commit, "packages": versions, "model_processor_sha256": processors, "dataset_smoke": io,
    }, indent=2))


if __name__ == "__main__":
    main()
