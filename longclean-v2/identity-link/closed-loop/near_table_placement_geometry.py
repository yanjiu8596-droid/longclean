"""Screen held-soup release points against native collision geometry on CPU."""

import argparse
from itertools import product
from math import hypot
import os
from pathlib import Path

import mujoco
import numpy as np

from stage4_fixture import require, validate_study
from actual_physics_simulator import load_native
from common import SOUP, read, sha, write
from near_table_placement_pair import COMPLETED, FORK, LAYOUT, SOURCE_SHA
from soup_placement_feasibility_v2 import pinned, read_lines


OBSTACLES = (*COMPLETED, "basket_1")
GRID_MM = tuple(range(-40, 41, 10))
HEIGHTS_FROM_BASKET_TOP_MM = tuple(range(35, -6, -5))
MIN_CLEARANCE_M, DISTANCE_CAP_M = .003, .02
POSE_GRID_MM = tuple(range(-45, 46, 5))
TILT_DEGREES = (15, 30, 45)
TILT_AXIS_AZIMUTHS = tuple(range(0, 360, 45))
DENSE_HEIGHTS_MM = tuple(range(35, -6, -1))
FIXED_SCREEN_SHA = "e40084117f8ba2d414e58a51bafdfd3aada90b878c8d29f1de1f0e2a394af799"
TILT_SCREEN_SHA = "0ae103c47de2fb6624aab375ea7658fd52c284398347a5709064d8b6bf10c843"


def load_fork(pair):
    summary = read(pair / "summary.json")
    require(summary["layout"] == LAYOUT and summary["fork_step"] == FORK
            and summary["source_sha256"] == SOURCE_SHA and not summary["different_renderer_allowed"],
            "Screening requires the strictly reproduced placement pair")
    arm = pair / "placement"
    result = read(pinned(arm / "result.json", summary["arms"]["placement"]["result_sha256"]))
    path = pinned(arm / "physics.jsonl", result["file_sha256"]["physics.jsonl"])
    fork = read_lines(path)[0]
    handoff = result["handoff"]
    require(fork["step"] == FORK and fork["held"][SOUP] and all(fork["goals"][obj] for obj in COMPLETED)
            and fork["state_sha256"] == handoff["state_sha256"] == summary["common_fork"]["state_sha256"],
            "The archived held-soup fork disagrees with the pair")
    return fork, handoff["region"], result["target_m"], sha(path)


def set_pose(sim, obj, position, rotation):
    joint, = obj.joints
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=np.float64).reshape(-1))
    sim.data.set_joint_qpos(joint, np.r_[position, quaternion])


def restore_fork(env, fork, region):
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    objects = env.objects_dict
    props = {**fork["body_poses"], "basket_1": {"position_m": region["basket_body_position_m"],
                                            "rotation": region["basket_body_rotation"]}}
    for obj, pose in props.items():
        set_pose(sim, objects[obj], pose["position_m"], pose["rotation"])
    mujoco.mj_kinematics(model, data)
    for obj, pose in props.items():
        body = env.obj_body_id[obj]
        require(np.allclose(data.xpos[body], pose["position_m"], atol=1e-9, rtol=0)
                and np.allclose(data.xmat[body].reshape(3, 3), pose["rotation"], atol=1e-9, rtol=0),
                f"Restored pose disagrees for {obj}")

    geoms = {obj: [sim.model.geom_name2id(name) for name in objects[obj].contact_geoms] for obj in props}
    require(all(geoms.values()) and all(geom >= 0 for ids in geoms.values() for geom in ids),
            "Missing collision geometry for the supplied objects")
    site = sim.model.site_name2id(region["name"])
    require(np.allclose(data.site_xpos[site], region["position_m"], atol=1e-9, rtol=0), "Basket region moved")
    return {obj: tuple(product(geoms[SOUP], geoms[obj])) for obj in OBSTACLES}


def vertical_clearance(env, geometry_pairs, xy, top, rotation, heights):
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    clearance = dict.fromkeys(OBSTACLES, DISTANCE_CAP_M)
    for dz in heights:
        set_pose(sim, env.objects_dict[SOUP], np.r_[xy, top + dz / 1000], rotation)
        mujoco.mj_kinematics(model, data)
        for obj, pairs in geometry_pairs.items():
            closest = min(mujoco.mj_geomDistance(model, data, first, second, DISTANCE_CAP_M, None)
                          for first, second in pairs)
            clearance[obj] = min(clearance[obj], closest)
    return clearance


def sample_paths(env, geometry_pairs, region, prior_target, rotation, grid):
    lower, upper = (np.asarray(region["objects"][SOUP][key]) for key in ("lower", "upper"))
    center = np.asarray(region["position_m"])
    release_above_top_m = prior_target[2] - upper[2]
    # Restore only the object poses needed for collision queries. This never steps the simulation.
    cases = []
    for dx, dy in product(grid, repeat=2):
        xy = center[:2] + np.array([dx, dy]) / 1000
        if not np.all((lower[:2] < xy) & (xy < upper[:2])):
            continue
        clearance = vertical_clearance(env, geometry_pairs, xy, upper[2], rotation, HEIGHTS_FROM_BASKET_TOP_MM)
        cases.append({"offset_mm": [dx, dy], "release_target_m": [*xy, upper[2] + release_above_top_m],
                      "clearance_m": clearance, "min_clearance_m": min(clearance.values())})
    return cases


def screen(env, fork, region, prior_target):
    geometry_pairs = restore_fork(env, fork, region)
    rotation = fork["body_poses"][SOUP]["rotation"]
    cases = sample_paths(env, geometry_pairs, region, prior_target, rotation, GRID_MM)
    previous, = [case for case in cases if np.allclose(case["release_target_m"], prior_target, atol=1e-9, rtol=0)]
    eligible = [case for case in cases if case["min_clearance_m"] >= MIN_CLEARANCE_M]
    chosen = max(eligible, key=lambda case: (case["min_clearance_m"], -hypot(*case["offset_mm"])), default=None)
    return cases, previous, chosen


def dense_path(env, geometry_pairs, region, case, rotation):
    top = region["objects"][SOUP]["upper"][2]
    xy = np.asarray(case["release_target_m"][:2])
    rows = []
    for dx, dy in product((-1, 0, 1), repeat=2):
        sample_xy = xy + np.array([dx, dy]) / 1000
        clear = vertical_clearance(env, geometry_pairs, sample_xy, top, rotation, DENSE_HEIGHTS_MM)
        rows.append({"jitter_mm": [dx, dy], "clearance_m": clear, "min_clearance_m": min(clear.values())})
    return {"samples": rows, "min_clearance_m": min(row["min_clearance_m"] for row in rows)}


def screen_poses(env, fork, region, prior_target, reference):
    from scipy.spatial.transform import Rotation

    geometry_pairs = restore_fork(env, fork, region)
    original_rotation = np.asarray(fork["body_poses"][SOUP]["rotation"])
    fixed = sample_paths(env, geometry_pairs, region, prior_target, original_rotation, POSE_GRID_MM)
    by_offset = {tuple(case["offset_mm"]): case for case in fixed}
    require(all(by_offset[tuple(case["offset_mm"])] == case for case in reference["cases"]),
            "Fixed-pose results changed at the original 81 points")

    poses = []
    for tilt, azimuth in product(TILT_DEGREES, TILT_AXIS_AZIMUTHS):
        az = np.deg2rad(azimuth)
        axis = np.array([np.cos(az), np.sin(az), 0.])
        rotation = Rotation.from_rotvec(np.deg2rad(tilt) * axis).as_matrix() @ original_rotation
        paths = sample_paths(env, geometry_pairs, region, prior_target, rotation, POSE_GRID_MM)
        poses.append({"tilt_deg": tilt, "axis_azimuth_deg": azimuth, "soup_rotation": rotation.tolist(),
                      "cases": paths})
        print(f"tilt={tilt} axis={azimuth}: {sum(row['min_clearance_m'] >= MIN_CLEARANCE_M for row in paths)} eligible",
              flush=True)

    ranked = [(pose, case) for pose in poses for case in pose["cases"] if case["min_clearance_m"] >= MIN_CLEARANCE_M]
    ranked.sort(key=lambda item: (-item[1]["min_clearance_m"], item[0]["tilt_deg"], hypot(*item[1]["offset_mm"])))
    refined = []
    for pose, case in ranked:
        rotation = pose["soup_rotation"]
        check = dense_path(env, geometry_pairs, region, case, rotation)
        choice = {**{key: pose[key] for key in ("tilt_deg", "axis_azimuth_deg", "soup_rotation")},
                  **case, "dense": check, "fixed_at_same_point": by_offset[tuple(case["offset_mm"])]}
        refined.append(choice)
        if check["min_clearance_m"] >= MIN_CLEARANCE_M:
            choice["fixed_dense_at_same_point"] = dense_path(env, geometry_pairs, region, case, original_rotation)
            return fixed, poses, refined, choice
    return fixed, poses, refined, None


def refine_fixed_pose(env, fork, region, prior_target, reference):
    """Check every 1 mm point near a coarse cell that could reach the clearance floor."""
    geometry_pairs = restore_fork(env, fork, region)
    rotation = fork["body_poses"][SOUP]["rotation"]
    top = region["objects"][SOUP]["upper"][2]
    center = np.asarray(region["position_m"][:2])
    grid_limit = max(POSE_GRID_MM)
    subgrid = tuple(product(range(-2, 3), repeat=2))
    cell_search_floor = MIN_CLEARANCE_M - hypot(2, 2) / 1000
    offsets = set()
    for row in reference["fixed_cases"]:
        if row["min_clearance_m"] >= cell_search_floor:
            x, y = row["offset_mm"]
            offsets.update((x + dx, y + dy) for dx, dy in subgrid
                           if max(abs(x + dx), abs(y + dy)) <= grid_limit)
    cases = []
    for dx, dy in sorted(offsets):
        xy = center + np.array([dx, dy]) / 1000
        clearance = vertical_clearance(env, geometry_pairs, xy, top, rotation, DENSE_HEIGHTS_MM)
        cases.append({"offset_mm": [dx, dy], "release_target_m": [*xy, prior_target[2]], "clearance_m": clearance,
                      "min_clearance_m": min(clearance.values())})
    ranked = sorted(cases, key=lambda case: (-case["min_clearance_m"], hypot(*case["offset_mm"])))
    refined = []
    for case in ranked:
        if case["min_clearance_m"] < MIN_CLEARANCE_M:
            break
        check = dense_path(env, geometry_pairs, region, case, rotation)
        choice = {**case, "soup_rotation": rotation, "dense": check}
        refined.append(choice)
        if check["min_clearance_m"] >= MIN_CLEARANCE_M:
            return cases, refined, choice
    return cases, refined, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pose-search", type=Path, help="Use the archived fixed-pose screen as a regression reference")
    mode.add_argument("--refine-fixed", type=Path, help="Refine promising fixed-pose cells from the tilt screen")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("MUJOCO_GL") == "disable",
            "Disable CUDA and rendering before screening")
    require(not args.output.exists(), "Screening output already exists")
    bddl, _ = validate_study(args.study.resolve())
    fork, region, prior_target, trace_sha = load_fork(args.pair)
    report = {"layout": LAYOUT, "step": FORK, "fork_state_sha256": fork["state_sha256"],
              "trace_sha256": trace_sha, "script_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__, "min_clearance_m": MIN_CLEARANCE_M,
              "sample_heights_from_region_top_mm": list(HEIGHTS_FROM_BASKET_TOP_MM)}
    load_native().ensure_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv

    scene = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False,
                       has_offscreen_renderer=False, control_freq=20)
    try:
        if args.pose_search is not None:
            reference = read(pinned(args.pose_search, FIXED_SCREEN_SHA))
            require(reference["fork_state_sha256"] == fork["state_sha256"] and reference["trace_sha256"] == trace_sha,
                    "Fixed screen came from a different fork")
            fixed, poses, refined, chosen = screen_poses(scene.env, fork, region, prior_target, reference)
            report.update(schema="v2-near-table-tilt-screen/1", fixed_reference_sha256=FIXED_SCREEN_SHA,
                          dense_heights_from_region_top_mm=list(DENSE_HEIGHTS_MM), fixed_cases=fixed,
                          poses=poses, refined_candidates=refined, selected_target=chosen)
            cases = fixed
        elif args.refine_fixed is not None:
            reference = read(pinned(args.refine_fixed, TILT_SCREEN_SHA))
            require(reference["fork_state_sha256"] == fork["state_sha256"] and reference["trace_sha256"] == trace_sha,
                    "Tilt screen came from a different fork")
            cases, refined, chosen = refine_fixed_pose(scene.env, fork, region, prior_target, reference)
            report.update(schema="v2-near-table-fixed-refinement/1", tilt_screen_sha256=TILT_SCREEN_SHA,
                          dense_heights_from_region_top_mm=list(DENSE_HEIGHTS_MM),
                          cases=cases, refined_candidates=refined, selected_target=chosen)
        else:
            cases, previous, chosen = screen(scene.env, fork, region, prior_target)
            report.update(cases=cases, original_target=previous, selected_target=chosen)
    finally:
        scene.close()
    write(args.output, report)
    print(f"Screened {len(cases)} basket points on CPU. Selected: {None if chosen is None else chosen['offset_mm']}")


if __name__ == "__main__":
    main()
