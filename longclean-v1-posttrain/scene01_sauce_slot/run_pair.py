"""Replay exact LIBERO actions and compare two continuations at the pre-open state."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "protocol.json").read_text())
SOUP, SAUCE = "alphabet_soup_1", "tomato_sauce_1"
REVIEW = ROOT.parent
SYNC = REVIEW / "scene01-warm75-20261124-sync"
SOURCE = REVIEW / PLAN["reference"]
LAB = Path(os.environ["TABLECLEAR_LAB"])
sys.path[:0] = [str(SYNC), str(REVIEW / "scene01-visible-low-place-20261124"),
                str(LAB / "longclean/scripts"), str(LAB / "roboenv/src"), str(LAB / "roboenv/runtime/src")]


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def put(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf8")


def digest(data):
    return sha256(data).hexdigest()


def image_hashes(obs, cameras):
    values = {k: obs[v] for k, v in cameras.items()}
    values.update({k: obs["robot0_" + k] for k in ("eef_pos", "eef_quat", "gripper_qpos")})
    return {k: digest(v.tobytes()) for k, v in values.items()}


def measured(env, obs):
    sim = env.env.sim
    eef = env.env.robots[0].controller.eef_name
    return {"site_position": sim.data.get_site_xpos(eef).copy(), "site_rotation": sim.data.get_site_xmat(eef).copy(),
            "gripper_qpos": obs["robot0_gripper_qpos"].copy()}


def sensor_frame(env, obs, own, step, cameras):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    sim = env.env.sim
    frame = dict(step=np.array(step), **measured(env, obs), **own)
    for label, key in cameras.items():
        name = key.removesuffix("_image")
        frame[label] = obs[key].copy()
        frame[label + "_depth"] = get_real_depth_map(sim, obs[name + "_depth"]).squeeze(-1)
        frame[label + "_intrinsic"] = get_camera_intrinsic_matrix(sim, name, 256, 256)
        frame[label + "_extrinsic"] = get_camera_extrinsic_matrix(sim, name)
    return frame


class Scorer:
    """Simulation labels and contacts are consumed only in audit records."""
    def __init__(self, wrapped):
        self.env = wrapped.env
        self.ids = {n: {self.env.sim.model.geom_name2id(g) for g in self.env.objects_dict[n].contact_geoms}
                    for n in (SOUP, SAUCE)}

    def record(self, step):
        sim = self.env.sim
        site = "basket_1_contain_region"
        site_xyz = sim.data.get_site_xpos(site)
        site_rot = sim.data.get_site_xmat(site).reshape(3, 3)
        half = abs(site_rot @ np.asarray(self.env.object_sites_dict[site].size))
        lower = site_xyz - half
        lower[2] -= .010
        upper = site_xyz + half
        row = {"step": step, "lower": lower.tolist(), "upper": upper.tolist()}
        for name, ids in self.ids.items():
            contacts = set()
            for c in sim.data.contact[:sim.data.ncon]:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if a in ids and b not in ids:
                        other = sim.model.geom_id2name(b)
                        if other:
                            contacts.add(other)
            xyz = sim.data.body_xpos[self.env.obj_body_id[name]]
            row[name] = {"xyz": xyz.tolist(), "contacts": sorted(contacts),
                         "in": bool(self.env._eval_predicate(["in", name, site]))}
            assert row[name]["in"] == bool(np.all(xyz > lower) and np.all(xyz < upper))
        row["dual"] = bool(self.env._check_success())
        assert row["dual"] == all(row[n]["in"] for n in self.ids)
        return row


def run(arm, out):
    from roboenv_runtime.adapters.libero.config import ensure_libero_config
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from run_native_osc import CAMERAS
    from same_frame import capture_step

    tag = PLAN["case"]
    prior = SOURCE / "runs" / (tag + "-retry") / "integrated"
    reference = json.loads((prior / "episode.json").read_bytes())
    diag = SOURCE / "diagnostic"
    chunks = read_rows(prior / "actions.jsonl") + read_rows(diag / "extension.jsonl")
    state_rows = read_rows(diag / "state.jsonl")
    native = {r["step"]: r for r in read_rows(diag / "placement-contacts.jsonl")}
    actions = [a for r in chunks for a in r["actions"][:r["executed"]]]
    ends = {r["start_step"] + r["executed"]: r for r in chunks}
    ends[0] = reference
    fork, end = PLAN["fork"], PLAN["end"]
    assert len(actions) == len(state_rows) - 1 and actions[fork - 1][-1] > 0 and actions[fork][-1] < 0
    assert digest((SOURCE / "budget-evidence.tar.gz").read_bytes()) == PLAN["reference_sha256"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    suite = get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    assert digest(bddl.read_bytes()) == reference["bddl_sha256"]
    with np.load(SOURCE / "layouts.npz", allow_pickle=False) as arrays:
        matches = [a.copy() for name in arrays.files if digest((a := arrays[name]).tobytes()) == reference["init_state_sha256"]]
    assert len(matches) == 1
    wrapped = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 control_freq=20, camera_depths=True)
    try:
        wrapped.reset()
        wrapped.seed(reference["seed"])
        wrapped.reset()
        wrapped.set_init_state(matches[0])
        assert digest(wrapped.get_sim_state().tobytes()) == reference["init_state_sha256"]
        for _ in range(10):
            obs = wrapped.step([0, 0, 0, 0, 0, 0, -1])[0]
        controller = wrapped.env.robots[0].controller
        controller.use_delta = True
        assert controller.name == "OSC_POSE" and wrapped.env.control_freq == 20
        score = Scorer(wrapped)
        policy = None
        frames = []
        reference_rgb_matches = {}
        actual = []
        with (out / "states.jsonl").open("w") as states, (out / "actions.jsonl").open("w") as applied:
            for j in range(end + 1):
                if j:
                    sample = j in PLAN["view_steps"]
                    if sample:
                        response, own = capture_step(wrapped, command, type(wrapped).step)
                    else:
                        response = wrapped.step(command)
                    obs = response[0]
                    assert not wrapped.env.done, "Unexpected native time limit"
                    if sample:
                        frame = sensor_frame(wrapped, obs, own, j, CAMERAS)
                        np.savez_compressed(out / f"view-{j}.npz", **frame)
                        frames.append(frame)
                state_sha = digest(wrapped.get_sim_state().tobytes())
                scoring = score.record(j)
                assert scoring["dual"] == bool(wrapped.check_success())
                if j <= fork or arm == "A":
                    for name in (SOUP, SAUCE):
                        assert scoring[name]["xyz"] == state_rows[j][name]["xyz"], (j, name, "pose")
                        assert scoring[name]["in"] == state_rows[j][name]["in"]
                        if j in native:
                            assert scoring[name]["contacts"] == native[j][name], (j, name, "contacts")
                    if j in ends:
                        assert state_sha == ends[j]["sim_state_sha256"], (j, "physics")
                        pixels = image_hashes(obs, CAMERAS)
                        matches = {k: v == ends[j]["observation_sha256"][k] for k, v in pixels.items()}
                        assert all(matches[k] for k in ("eef_pos", "eef_quat", "gripper_qpos"))
                        if os.environ["MUJOCO_GL"] == "egl":
                            assert all(matches.values()), (j, "original observation")
                        reference_rgb_matches[str(j)] = matches
                scoring["sim_state_sha256"] = state_sha
                states.write(json.dumps(scoring, allow_nan=False) + "\n")
                if j == fork:
                    hashes = {"physics": state_sha, "observation": image_hashes(obs, CAMERAS),
                              "frames": [{k: digest(v.tobytes()) for k, v in f.items()} for f in frames]}
                    if arm == "B":
                        assert hashes == json.loads((ROOT / "A" / "fork.json").read_bytes())
                        from slot import Placement
                        policy = Placement(frames, PLAN["geometry"])
                        put(out / "selection.json", policy.decision)
                    put(out / "fork.json", hashes)
                if j == end:
                    break
                value, note = actions[j], {"phase": "original"}
                if policy is not None:
                    value, note = policy.action(j, measured(wrapped, obs), value)
                command = np.asarray(value, dtype=np.float32)
                assert command.shape == (7,) and np.isfinite(command).all()
                if j >= fork:
                    actual.append(command.tolist())
                    applied.write(json.dumps({"start": j, "action": command.tolist(), **note}, allow_nan=False) + "\n")
        put(out / "finished.json", {"arm": arm, "fork": fork, "end": end, "backend": os.environ["MUJOCO_GL"],
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "init_sha256": reference["init_state_sha256"],
            "reference_observations": reference_rgb_matches, "suffix_actions": len(actual),
            "source_sha256": PLAN["reference_sha256"], "suffix_float32_sha256": digest(np.asarray(actual,dtype="<f4").tobytes())})
        print("VERIFIED", arm, fork, end, flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("A", "B"))
    args = parser.parse_args()
    run(args.arm, ROOT / args.arm)
