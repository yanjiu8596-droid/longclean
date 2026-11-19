"""Compare archived soup motions with headless static contact queries."""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from stage4_fixture import require, validate_study
from actual_physics_simulator import load_native
from common import SOUP, read, sha, write
from near_table_placement_geometry import DISTANCE_CAP_M, restore_fork, set_pose
from placement_holdout import read_fork
from soup_placement_feasibility_v2 import pinned, read_lines


CASES = (935630, 935631)
REFERENCE = "9bf02ed69564b4b82ec5da268547d426933a84c4354676f9bd4b66f41745c635"
ENVELOPE_MM = (60, 45, 35, 30, 25, 20, 15, 10, 5, 0, -5)
NATIVE_CCD = int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)


def native_contacts(data, pairs):
    lookup = {frozenset(pair): group for group, group_pairs in pairs.items() for pair in group_pairs}
    result = {group: [] for group in pairs}
    for contact in data.contact[:data.ncon]:
        group = lookup.get(frozenset((int(contact.geom1), int(contact.geom2))))
        if group is not None:
            result[group].append({"geoms": [int(contact.geom1), int(contact.geom2)], "distance_m": float(contact.dist)})
    return result


def inspect_distances(env, pairs, point, orientation):
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    flags = model.opt.enableflags
    set_pose(sim, env.objects_dict[SOUP], point, orientation)
    time = data.time
    try:
        results = {}
        for label, mask in (("loaded", flags), ("alternate_ccd", flags ^ NATIVE_CCD)):
            model.opt.enableflags = mask
            mujoco.mj_forward(model, data)
            contacts = native_contacts(data, pairs)
            groups = {}
            for obj, group in pairs.items():
                values = []
                for first, second in group:
                    dist = mujoco.mj_geomDistance(model, data, first, second, DISTANCE_CAP_M, None)
                    flipped = mujoco.mj_geomDistance(model, data, second, first, DISTANCE_CAP_M, None)
                    with_cap = {str(mm): mujoco.mj_geomDistance(model, data, first, second, mm / 1000, None)
                                for mm in (3, 5, 10, 50)}
                    center_dist = np.linalg.norm(data.geom_xpos[first] - data.geom_xpos[second])
                    radius_sum = model.geom_rbound[first] + model.geom_rbound[second]
                    values.append({"geoms": [model.geom(first).name, model.geom(second).name],
                                   "distance_m": dist, "reversed_m": flipped, "other_cutoffs_m": with_cap,
                                   "sphere_overlap_bound_m": max(0., float(radius_sum - center_dist)),
                                   "radius_sum_m": float(radius_sum)})
                groups[obj] = {"closest": min(values, key=lambda row: row["distance_m"]),
                               "cutoff_minima_m": {str(mm): min(row["other_cutoffs_m"][str(mm)] for row in values)
                                                   for mm in (3, 5, 10, 50)}, "contacts": contacts[obj]}
            results[label] = {"native_ccd": bool(mask & NATIVE_CCD), "objects": groups}
        require(data.time == time, "Static contact query advanced simulation time")
        return results
    finally:
        model.opt.enableflags = flags


def review_case(env, root, row, index, pair):
    seed = row["seed"]
    fork, frame = read_fork(root, seed, index)
    require(frame["state_sha256"] == row["state_sha256"], "The probe must use the evaluated fork")
    region = frame["region"]
    top = region["objects"][SOUP]["upper"][2]
    pairs = restore_fork(env, fork, region)
    folder = root / "pairs" / str(seed) / "lateral"
    arm = read(pinned(folder / "result.json", pair[str(seed)]["lateral"]))
    trace = read_lines(pinned(folder / "physics.jsonl", arm["file_sha256"]["physics.jsonl"]))
    require(trace[0] == fork, "Archived trajectory begins at the wrong fork")
    target = arm["target_m"]
    rotation = fork["body_poses"][SOUP]["rotation"]
    sweep = [{"height_above_top_mm": z, **inspect_distances(env, pairs, [*target[:2], top+z/1000], rotation)}
             for z in ENVELOPE_MM]
    release = arm["outcome"]["first_unheld_step"]
    opening = trace[-1]["step"] if arm["opening"] is None else arm["opening"]["state_step"]
    highlights = {opening, trace[-1]["step"]}
    if release is not None:
        highlights.update(range(release-1, release+7))
    physical = []
    for sample in trace:
        if sample["step"] not in highlights:
            continue
        for obj, pose in sample["body_poses"].items():
            set_pose(env.sim, env.objects_dict[obj], pose["position_m"], pose["rotation"])
        pose = sample["body_poses"][SOUP]
        physical.append({"step": sample["step"], "held": sample["held"][SOUP], "forces_n": sample["soup_normal_force_N"],
                         "native_goals": sample["goals"],
                         **inspect_distances(env, pairs, pose["position_m"], pose["rotation"])})
    return {"sha256": frame["state_sha256"], "sweep": sweep, "actual_poses": physical, "target_m": target,
            "lateral_result_sha256": pair[str(seed)]["lateral"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("MUJOCO_GL") == "disable",
            "Run contact probes on the CPU with rendering disabled")
    require(not args.output.exists(), "Refuse to overwrite contact review")
    bddl, _ = validate_study(args.study.resolve())
    source = args.source.resolve()
    report = read(pinned(source / "summary.json", REFERENCE))
    index, pair = read(source / "inspect.json"), read(source / "pair.json")
    load_native().ensure_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False,
                     has_offscreen_renderer=False, control_freq=20)
    try:
        cases = {str(row["seed"]): review_case(env.env, source, row, index, pair)
                 for row in report["rows"] if row["seed"] in CASES}
        require(set(cases) == {str(seed) for seed in CASES}, "Lost a development probe")
    finally:
        env.close()
    write(args.output, {"mujoco": mujoco.__version__, "source_sha256": REFERENCE, "script_sha256": sha(Path(__file__)),
                        "cases": cases})


if __name__ == "__main__":
    main()
