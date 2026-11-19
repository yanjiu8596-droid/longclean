"""Shared pins, paths and observation contract for the Scene 08 study."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_LAB = Path(os.environ.get("TABLECLEAR_LAB", "."))
PROJECT_REVISION = "42a8fa2d01a60e82d3da8707b421f6b6d722aad1"
LEROBOT_REVISION = "772bc21ef1ab17b72e17ebb1593ff78725a21d55"
ARCHIVE_SHA256 = "26c7d50a7e87de2248a0882c9ac971d4c0fe17a7ec8797a48e422befff51577a"
RULE_SHA256 = "18d5f85103681fd40c918d33ff6f7920d28ac80bf86e7b89b9a6911b02c5b911"
SCENE = 8
TASK_ID = "libero/libero-10-08"
SOUP = "alphabet_soup_1"
FPS = 20
BUDGET = 520
EXECUTE = 10
HORIZON = 50
VAL_LAYOUTS = frozenset((24, 37, 40))
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
RAW_KEYS = ("camera1", "camera2", "eef_pos", "eef_quat", "gripper_qpos")
ARMS = ("base", "rule", "natural", "corrected")
METHODS = ARMS[2:]
TRAIN_SEED = 20261119
REPLAY_FIELDS = ("sim_time", "sim_state_sha256", "observation_sha256")


@dataclass(frozen=True)
class Lab:
    root: Path

    @property
    def project(self):
        return self.root / "longclean"

    @property
    def roboenv(self):
        return self.root / "roboenv"

    @property
    def sim_python(self):
        return Path(os.environ.get("TABLECLEAR_SIM_PYTHON", self.roboenv / ".venv-sim/libero/bin/python"))

    @property
    def workspace(self):
        return self.root / "workspace"

    @property
    def model(self):
        return self.workspace / "models/pi05_libero_finetuned_v044"

    @property
    def review(self):
        return self.workspace / "results/lab-backbone-v1/review"

    @property
    def archive(self):
        return Path(os.environ.get("TABLECLEAR_V1_ARCHIVE", self.review / "scene08-night-release-120/night-evidence.zip"))

    def add_imports(self):
        sys.path[:0] = [
            str(self.project / "scripts"),
            str(self.project / "src"),
            str(self.project / "kinex_devices/smolvla"),
        ]


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def file_sha(path):
    digest = sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def array_sha(array):
    return sha256(array.tobytes()).hexdigest()


def state_vector(eef_pos, eef_quat, fingers):
    """Match V1's float32 LIBERO state encoding."""
    import numpy as np

    quat = eef_quat.astype(np.float32)
    one = np.float32(1)
    w = np.clip(quat[3], -one, one)
    denominator = np.sqrt(one - w * w)
    axis_angle = np.zeros(3, dtype=np.float32)
    if denominator > 1e-10:
        axis_angle = (quat[:3] / denominator) * (np.float32(2) * np.arccos(w))
    return np.concatenate([eef_pos, axis_angle, fingers]).astype(np.float32)


def read_physics(folder, steps, terminal):
    rows = [json.loads(line) for line in (folder / "audit.jsonl").read_bytes().splitlines()]
    if [row["step"] for row in rows] != list(range(steps + 1)) or rows[-1]["goal"] != terminal:
        raise ValueError("Per-control-step physical audit differs from native terminal")
    first_held, first_release = None, None
    for row in rows:
        if first_held is None and row["soup_held"]:
            first_held = row["step"]
        if first_held is not None and first_release is None and not row["soup_held"] and row["soup_in"]:
            first_release = row["step"]
    return {
        "first_held": first_held, "first_release_in_basket": first_release,
        "release": first_release is not None
        and all(row["soup_in"] and not row["soup_held"] for row in rows[first_release:]),
        "non_goal_contact_steps": sum(bool(row["non_goal_contacts"]) for row in rows[1:]),
    }
