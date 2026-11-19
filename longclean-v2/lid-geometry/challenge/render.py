"""Generate blind suite: settled release plus counterfactual RGB-D snapshots."""
import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import CAMERAS, SAUCE, SOUP, V1, array_hash, read, sha, write
from simulator import Simulator, ensure_libero_config

CASES = {
    "inside": (True, False),
    "outside": (False, False),
    "rim": (False, False),
    "other_can": (False, True),
    "soup_offscreen": (False, False),
}
VIEWS = ((265, .670), (270, .675), (275, .680))


def drive(scene, target, rotation, grip, steps):
    robot = scene.env.env.robots[0]
    data = scene.env.env.sim.data
    site = robot.controller.eef_name
    for _ in range(steps):
        dp = 1.4 * (target - data.get_site_xpos(site)) / .05
        dr = Rotation.from_matrix(rotation @ data.get_site_xmat(site).reshape(3, 3).T).as_rotvec() / .5
        action = np.r_[np.clip(np.r_[dp, dr], -.85, .85), grip].astype(np.float32)
        scene.obs, *_ = scene.env.step(action)
        scene.frame += 1


def held(scene):
    env = scene.env.env
    return bool(env._check_grasp(env.robots[0].gripper, env.objects_dict[SOUP]))


def positive(scene, seed, asset):
    """Reuse the vertical-withdrawal control sequence from active-look."""
    xyz = scene.xyz()
    floor = {name: float(xyz[name][2]) for name in (SOUP, SAUCE)}
    safe = scene.env.env.sim.data.get_site_xpos(scene.env.env.robots[0].controller.eef_name).copy()
    safe[2] = max(.74, safe[2])
    basket = np.asarray(xyz["basket_1"])
    grab = np.asarray(xyz[SOUP]) + asset["grasp_xyz_offset_m"]
    high = grab + [0, 0, .16]
    over = np.array([basket[0] + (-.022 if seed % 2 else .022), basket[1] - .025, .67])
    lower = over.copy()
    lower[2] = .555
    rotation = np.asarray(asset["base_rotation_matrix"])
    stages = [(safe, 16, -1), (high, 33, -1), (grab, 44, -1), (grab, 24, 1),
              (high, 29, 1), (over, 44, 1), (lower, 28, 1), (lower, 20, -1), (over, 22, -1)]
    for stage, (point, steps, grip) in enumerate(stages):
        if stage in (4, 7) and not held(scene):
            raise ValueError(f"grasp not established at stage {stage}")
        drive(scene, point, rotation, grip, steps)
    assert scene.frame == 260
    if held(scene) or not scene.goals()[SOUP]:
        raise ValueError("common release did not settle inside before camera samples")
    return floor, over, rotation


def place_center(scene, name, target):
    """Translate the free joint to a requested world body center without stepping."""
    env = scene.env.env
    sim = env.sim
    joint, = env.objects_dict[name].joints
    pose = sim.data.get_joint_qpos(joint).copy()
    assert pose.shape == (7,)
    pose[:3] += np.asarray(target) - sim.data.body_xpos[env.obj_body_id[name]]
    sim.data.set_joint_qpos(joint, pose)
    sim.forward()


def stage_control(scene, case, floor, basket, soup, seed):
    """Compose snapshot-only counterexamples from identical settled control states."""
    if case == "inside":
        return
    if case == "rim":
        # A nearby elevated can models a geometric rim distractor. Gravity is
        # intentionally not stepped: this fixture is not a claim of support.
        sign = -1 if seed % 2 else 1
        place_center(scene, SOUP, [basket[0] + sign * .062, soup[1], soup[2] + .08])
    else:
        outside_y = basket[1] - .135 if case == "outside" else -.16
        place_center(scene, SOUP, [soup[0], outside_y, floor[SOUP]])
        if case == "other_can":
            # Can root heights differ; use the soup's observed lift from table
            # to basket floor, not the soup center z as a shared root height.
            place_center(scene, SAUCE, [soup[0], soup[1], floor[SAUCE] + soup[2] - floor[SOUP]])
    scene.obs = scene.env.regenerate_obs_from_state(scene.env.get_sim_state())


def registered_masks(scene):
    env = scene.env.env
    sim = env.sim
    camera = CAMERAS["camera2"].removesuffix("_image")
    raw = scene.obs[CAMERAS["camera2"]]
    h, w = raw.shape[:2]
    rendered = sim.render(camera_name=camera, width=w, height=h)
    transforms = [(sy, sx) for sy in (1, -1) for sx in (1, -1)
                  if np.array_equal(raw, rendered[::sy, ::sx])]
    if len(transforms) != 1:
        raise RuntimeError(f"wrist RGB does not uniquely register to segmentation at {scene.frame}")
    sy, sx = transforms[0]
    seg = sim.render(camera_name=camera, width=w, height=h, segmentation=True)[::sy, ::sx]
    assert seg.shape == (h, w, 2)
    is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    masks = {}
    for name in (SOUP, SAUCE):
        root = env.obj_body_id[name]
        geom_ids = []
        for geom, body in enumerate(sim.model.geom_bodyid):
            while body not in (0, root):
                body = sim.model.body_parentid[body]
            if body == root:
                geom_ids.append(geom)
        assert geom_ids
        mask = is_geom & np.isin(seg[..., 1], geom_ids)
        masks[name] = mask[::-1, ::-1].copy()
    assert not (masks[SOUP] & masks[SAUCE]).any()
    return masks


def capture(scene, out, step):
    obs = scene.capture()
    masks = registered_masks(scene)
    assert array_hash(scene.env.get_sim_state()) == obs["state_sha256"]
    (scene.output / "observation.npz").replace(out / f"{step}.npz")
    np.save(out / f"{step}-mask.npy", masks[SOUP])
    np.save(out / f"{step}-sauce-mask.npy", masks[SAUCE])
    xyz = scene.xyz()
    return {"step": step, "scene_step": scene.frame, "goals": obs["audit"]["goals"],
            "soup_in": obs["audit"]["goals"][SOUP], "sauce_in": obs["audit"]["goals"][SAUCE],
            "held_soup": obs["audit"]["held_soup"], "state_sha256": obs["state_sha256"],
            "observation_sha256": obs["observation_sha256"], "soup_xyz": xyz[SOUP],
            "sauce_xyz": xyz[SAUCE], "basket_xyz": xyz["basket_1"],
            "soup_pixels": int(masks[SOUP].sum()), "sauce_pixels": int(masks[SAUCE].sum())}


def run_layout(seed, bddl, asset, out):
    out.mkdir()
    scene = Simulator(bddl, out, seed, False)
    try:
        initial = array_hash(scene.env.get_sim_state())
        try:
            floor, over, rotation = positive(scene, seed, asset)
        except ValueError as exc:
            return {"seed": seed, "initial_sha256": initial, "valid": False, "reason": str(exc), "cases": {}}
        camera_states = []
        for step, z in VIEWS:
            before = scene.frame
            drive(scene, np.r_[over[:2], z], rotation, -1, step - before)
            assert scene.frame == step
            camera_states.append((step, scene.env.get_sim_state().copy()))
        results = {}
        for name, expected in CASES.items():
            dest = out / name
            dest.mkdir()
            truth = {"seed": seed, "case": name, "condition": name, "snapshot_only": name != "inside",
                     "expected_soup_in_basket": expected[0], "expected_sauce_in_basket": expected[1], "frames": []}
            for step, state in camera_states:
                scene.frame = step
                scene.obs = scene.env.regenerate_obs_from_state(state)
                xyz = scene.xyz()
                stage_control(scene, name, floor, xyz["basket_1"], xyz[SOUP], seed)
                truth["frames"].append(capture(scene, dest, step))
            predicate_match = all(not r["held_soup"] and (r["soup_in"], r["sauce_in"]) == expected
                                  for r in truth["frames"])
            if name == "soup_offscreen":
                visibility_match = all(r["soup_pixels"] == 0 for r in truth["frames"])
            else:
                visible = "sauce_pixels" if name == "other_can" else "soup_pixels"
                visibility_match = sum(r[visible] >= 50 for r in truth["frames"]) >= 2
            truth["eligible"] = truth["valid"] = predicate_match and visibility_match
            truth["predicate_match"] = predicate_match
            truth["visibility_match"] = visibility_match
            truth["soup_in"] = all(r["soup_in"] for r in truth["frames"])
            write(dest / "truth.json", truth)
            results[name] = {"eligible": truth["eligible"], "valid": truth["valid"], "frames": len(truth["frames"]),
                             "path": str(dest.relative_to(out.parent))}
        return {"seed": seed, "initial_sha256": initial, "valid": all(x["valid"] for x in results.values()),
                "cases": results}
    finally:
        scene.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--seed-base", type=int, default=928100)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    if args.count < 1 or not 928000 <= args.seed_base <= args.seed_base + args.count - 1 <= 928999:
        p.error("use the independent 928000–928999 seed range")
    out = args.output.resolve()
    out.mkdir(parents=True)
    plan = read(ROOT / "study4" / "plan.json")
    freeze = read(ROOT / "study4" / "frozen.json")
    assert all(sha(ROOT / name) == expected for name, expected in freeze["code"].items())
    bddl = ROOT / "study4" / plan["bddl"]
    assert sha(bddl) == plan["bddl_sha256"]
    asset_path = V1 / "grasp-reference.json"
    asset = read(asset_path)
    ensure_libero_config()
    results = []
    for seed in range(args.seed_base, args.seed_base + args.count):
        result = run_layout(seed, bddl, asset, out / str(seed))
        results.append(result)
        print(seed, result["valid"], result.get("reason", result["cases"]), flush=True)
    assert len({r["initial_sha256"] for r in results}) == len(results)
    write(out / "manifest.json", {"source_sha256": sha(Path(__file__)), "bddl_sha256": sha(bddl),
                                  "grasp_reference_sha256": sha(asset_path), "layouts": results})


if __name__ == "__main__":
    main()
