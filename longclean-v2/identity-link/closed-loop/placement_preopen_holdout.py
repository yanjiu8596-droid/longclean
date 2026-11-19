"""Test a held-path contact screen in crowded layouts left out of placement development."""

import argparse
from itertools import product
from math import dist
import os
from pathlib import Path

from stage4_fixture import require, validate_study
from actual_physics_simulator import instrument, load_native
from common import CHEESE, SAUCE, SOUP, read, sha, write
from near_table_placement_pair import RELEASE_ABOVE_M, lateral_target
from placement_holdout import audit_arm, inspect_forks, read_fork, read_index, run_arm
from placement_holdout_supplement import collect, load_reference_sources, make_plan as reference_plan
from soup_placement_feasibility_v2 import pinned


PROTOCOL = Path(__file__).with_name("placement_preopen_holdout_protocol.md")
OLD_CROWDED = "9bf02ed69564b4b82ec5da268547d426933a84c4354676f9bd4b66f41745c635"
FIRST_SEED, END_SEED, QUOTA = 935632, 935640, 4
OFFSETS_MM = ((30, 0), (-30, 0), (0, 30), (0, -30), (0, 0))
HELD_HEIGHTS, FULL_HEIGHTS = (35, 30, 25, 20, 15), (35, 30, 25, 20, 15, 10, 5, 0, -5)
CLEARANCE = .003


def make_plan(primary, development):
    old = read(pinned(development / "summary.json", OLD_CROWDED))
    seen = {r["seed"] for r in old["rows"]}
    reference = reference_plan(primary)
    both, = [g for g in reference["groups"] if g["name"] == "both"]
    candidates = [r for r in both["candidates"] if FIRST_SEED <= r["seed"] < END_SEED]
    require([r["seed"] for r in candidates] == list(range(FIRST_SEED, END_SEED))
            and not any(r["seed"] in seen for r in candidates), "Prospective crowded seeds differ")
    return {**reference, "schema": "v2-preopen-crowded/1", "protocol_sha256": sha(PROTOCOL),
            "script_sha256": sha(Path(__file__)), "development_summary_sha256": OLD_CROWDED,
            "groups": [{**both, "quota": QUOTA, "candidates": candidates}]}


def selected(rows):
    eligible = [row for row in rows if row["min_clearance_m"] >= CLEARANCE and row["dense"]["min_clearance_m"] > 0]
    return max(eligible, key=lambda row: row["object_separation_m"], default=None)


def check_points(env, fork, region):
    import mujoco
    from near_table_placement_geometry import restore_fork, set_pose

    pairs = restore_fork(env, fork, region)
    group_pairs = [pair for group in pairs.values() for pair in group]
    model, data = env.sim.model._model, env.sim.data._data
    rotation = fork["body_poses"][SOUP]["rotation"]
    center = region["position_m"]
    site = region["objects"][SOUP]
    top = site["upper"][2]
    targets = [obj for obj in (CHEESE, SAUCE) if fork["goals"][obj]]

    def scan(xy, heights, cutoffs):
        values = {cap: [] for cap in cutoffs}
        for height in heights:
            set_pose(env.sim, env.objects_dict[SOUP], [*xy, top + height/1000], rotation)
            mujoco.mj_kinematics(model, data)
            for cap in cutoffs:
                values[cap].append(min(mujoco.mj_geomDistance(model, data, a, b, cap, None) for a, b in group_pairs))
        return values

    rows = []
    for dx, dy in OFFSETS_MM:
        xy = [center[0]+dx/1000, center[1]+dy/1000]
        if any(not low < val < high for low, val, high in zip(site["lower"][:2], xy, site["upper"][:2], strict=True)):
            continue
        paths = scan(xy, FULL_HEIGHTS, (.003, .020))
        values = {f"{mm}mm_{band}": min(paths[cap][part])
                  for cap, mm in ((.003, 3), (.020, 20))
                  for band, part in (("full", slice(None)), ("held", slice(len(HELD_HEIGHTS))))}
        nominal = values["3mm_held"]
        jitter = []
        if nominal >= CLEARANCE:
            for jx, jy in product((-1, 0, 1), repeat=2):
                probe = [xy[0]+jx/1000, xy[1]+jy/1000]
                actual = min(scan(probe, HELD_HEIGHTS, (.003,))[.003])
                jitter.append({"jitter_mm": [jx, jy], "gap_m": actual})
        rows.append({"offset_mm": [dx, dy], "release_target_m": [*xy, top+RELEASE_ABOVE_M],
                     "soup_rotation": rotation, "min_clearance_m": nominal, "geometry_ablation_m": values,
                     "dense": {"min_clearance_m": min((row["gap_m"] for row in jitter), default=0.), "samples": jitter},
                     "object_separation_m": min((dist(xy, fork["body_poses"][obj]["position_m"][:2]) for obj in targets),
                                                default=0.)})
    return {"candidates": rows, "selected": selected(rows)}


def screen(root, development, bddl, cases):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("MUJOCO_GL") == "disable",
            "Screen on the CPU")
    index = read_index(root, "inspect.json", cases)
    load_native().ensure_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv

    scene = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False,
                       has_offscreen_renderer=False, control_freq=20)
    try:
        previous = read(development / "inspect.json")
        development_results = {}
        for seed in (935630, 935631):
            fork, frame = read_fork(development, seed, previous)
            development_results[str(seed)] = check_points(scene.env, fork, frame["region"])
        known_right, = [r for r in development_results["935630"]["candidates"] if r["offset_mm"] == [30, 0]]
        require(known_right["geometry_ablation_m"]["20mm_full"] < CLEARANCE
                and known_right["geometry_ablation_m"]["3mm_held"] >= CLEARANCE,
                "The cutoff and held-window development contrast failed to replay")
        write(root / "development.json", development_results)
        outputs = {}
        for seed in cases:
            fork, frame = read_fork(root, seed, index)
            result = check_points(scene.env, fork, frame["region"])
            path = root / "forks" / str(seed) / "preopen-screen.json"
            write(path, {"fork_sha256": index[str(seed)], **result})
            outputs[str(seed)] = sha(path)
            print("PREOPEN_SCREEN", seed, None if result["selected"] is None else result["selected"]["offset_mm"], flush=True)
        write(root / "preopen-screen.json", outputs)
    finally:
        scene.close()


def read_choice(root, seed, index, forks):
    decision = read(pinned(root / "forks" / str(seed) / "preopen-screen.json", index[str(seed)]))
    require(decision["fork_sha256"] == forks[str(seed)] and decision["selected"] == selected(decision["candidates"]),
            "Screen no longer matches the frozen selection")
    return decision


def pair(root, bddl, cases):
    forks, index = (read_index(root, key, cases) for key in ("inspect.json", "preopen-screen.json"))
    native = instrument(load_native())
    native.ensure_libero_config()
    reports = {}
    for seed, source in cases.items():
        expected = read_fork(root, seed, forks)
        choice = read_choice(root, seed, index, forks)["selected"]
        folder = root / "pairs" / str(seed)
        arms = {"lateral": run_arm(native, bddl, seed, source, folder / "lateral", expected, None)}
        if choice is not None:
            arms["preopen"] = run_arm(native, bddl, seed, source, folder / "preopen", expected, choice)
        reports[str(seed)] = arms
        print("PREOPEN_PAIR", seed, {key: read(folder/key/"result.json")["outcome"]["stable_placement"]
                                      for key in arms}, flush=True)
    write(root / "pair.json", reports)


def audit(root, collection, cases):
    forks, index, pairs = (read_index(root, key, cases) for key in ("inspect.json", "preopen-screen.json", "pair.json"))
    rows = []
    for seed, source in cases.items():
        expected = read_fork(root, seed, forks)
        fork, frame = expected
        choice = read_choice(root, seed, index, forks)
        target, _ = lateral_target(fork, frame["region"])
        targets = {"lateral": target.tolist()}
        if choice["selected"] is not None:
            targets["preopen"] = choice["selected"]["release_target_m"]
        actual = pairs[str(seed)]
        require(set(actual) == set(targets), "Unexpected pair arms")
        folder = root / "pairs" / str(seed)
        arms = {arm: audit_arm(folder/arm, digest, source, expected, targets[arm]) for arm, digest in actual.items()}
        if "preopen" in arms and targets["lateral"] == targets["preopen"]:
            require(sha(folder/"lateral/physics.jsonl") == sha(folder/"preopen/physics.jsonl"),
                    "Identical same-state placement commands have different native effects")
        modes = {key: sum(p["geometry_ablation_m"][key] >= CLEARANCE for p in choice["candidates"])
                 for key in ("3mm_held", "3mm_full", "20mm_held", "20mm_full")}
        rows.append({"seed": seed, "cases_passing_nominal": modes,
                     "selected": choice["selected"], "fork_step": fork["step"], "arms": arms})
    matched = [(row["arms"]["lateral"]["stable_placement"], row["arms"]["preopen"]["stable_placement"])
               for row in rows if "preopen" in row["arms"]]
    total = {"eligible": len(rows), "attempts": len(collection["attempts"]), "matched": len(matched),
             "lateral_stable": sum(row["arms"]["lateral"]["stable_placement"] for row in rows),
             "preopen_stable": sum(b for _, b in matched),
             "wins": sum(b and not a for a, b in matched), "losses": sum(a and not b for a, b in matched),
             "ties": sum(a == b for a, b in matched),
             "any_object_regression_arms": sum(bool(arm["reversals"] or arm["completed_regrasp_steps"])
                                               for row in rows for arm in row["arms"].values()),
             "same_target": sum(row["selected"] is not None
                                and row["arms"]["lateral"]["target_m"] == row["arms"]["preopen"]["target_m"] for row in rows)}
    write(root / "summary.json", {"schema": "v2-preopen-pairs/1", "total": total, "rows": rows})
    print("PREOPEN_AUDIT_OK", total, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "inspect", "screen", "pair", "audit"))
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = make_plan(args.primary, args.development)
    if args.stage in ("collect", "inspect", "pair"):
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin a free render device")
    if args.stage == "collect":
        bddl, _ = validate_study(args.study.resolve())
        collect(args.output, args.primary, bddl, plan)
        return
    collection, cases = load_reference_sources(args.output, plan)
    if args.stage == "audit":
        audit(args.output, collection, cases)
        return
    bddl, _ = validate_study(args.study.resolve())
    if args.stage == "screen":
        screen(args.output, args.development, bddl, cases)
    else:
        {"inspect": inspect_forks, "pair": pair}[args.stage](args.output, bddl, cases)


if __name__ == "__main__":
    main()
