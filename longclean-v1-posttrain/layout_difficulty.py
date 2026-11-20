"""Calibrate Scene 08 difficulty on fresh native layouts near a known release stall."""

from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

from common import Lab, SCENE, array_sha, file_sha, read_json, write_json
from score import audit_episode, per_arm


ANCHOR_INIT = 44
POLICY_SEED = 8
DEV_SEEDS = range(862100, 862196)
LAYOUTS_PER_GROUP = 8
REGIONS = {"basket_1": 0.02, "alphabet_soup_1": 0.05, "cream_cheese_1": 0.05}


def object_xy(env):
    task = env.env
    return {name: task.sim.data.body_xpos[task.obj_body_id[name]][:2].tolist() for name in REGIONS}


def distance_squared(anchor, candidate):
    return sum(((actual - reference) / width) ** 2
               for name, width in REGIONS.items()
               for actual, reference in zip(candidate[name], anchor[name], strict=True))


def profile(lab, run):
    import numpy as np

    sys.path[:0] = [str(lab.roboenv / "src"), str(lab.roboenv / "runtime/src")]
    from roboenv_runtime.adapters.libero.config import ensure_libero_config

    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from sim_worker import verify_egl

    suite = get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(SCENE - 1)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    native = suite.get_task_init_states(SCENE - 1)
    fixed_hashes = sorted({array_sha(np.asarray(state)) for state in native})
    random.seed(POLICY_SEED)
    np.random.seed(POLICY_SEED)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             control_freq=20, camera_depths=True)
    candidates = []
    try:
        env.reset()
        env.seed(POLICY_SEED)
        env.reset()
        env.set_init_state(native[ANCHOR_INIT])
        verify_egl()
        anchor = object_xy(env)
        for number, seed in enumerate(DEV_SEEDS, 1):
            random.seed(seed)
            np.random.seed(seed)
            env.reset()
            env.seed(seed)
            env.reset()
            xy = object_xy(env)
            candidates.append({"layout_seed": seed, "xy_m": xy,
                               "raw_sha256": array_sha(env.get_sim_state()),
                               "distance_squared": distance_squared(anchor, xy)})
            if number == 1 or number % 12 == 0:
                print("PROFILE", number, len(DEV_SEEDS), flush=True)
    finally:
        env.close()
    write_json(run / "geometry.json", {
        "bddl_sha256": file_sha(bddl), "anchor_fixed_index": ANCHOR_INIT, "anchor_xy_m": anchor,
        "region_width_m": REGIONS, "fixed_native_sha256": fixed_hashes, "candidates": candidates,
    })


def freeze(lab, primary, run, gpu):
    from evaluate import attempt_regular, historical

    plan_path = run / "test-plan.json"
    profile_path = run / "geometry.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["gpu_uuid"] != gpu or plan["geometry_sha256"] != file_sha(profile_path):
            raise ValueError("The saved development plan uses different inputs")
        return
    lab.add_imports()
    geometry = read_json(profile_path)
    raw, settled = historical(lab, primary)
    raw.update(geometry["fixed_native_sha256"])
    prior_plan = primary / "test-plan.json"
    for case in read_json(prior_plan)["cases"]:
        raw.add(case["initial"]["raw_sha256"])
        settled.add(case["initial"]["settled_sha256"])

    candidates = geometry["candidates"]
    near = sorted(candidates, key=lambda c: (c["distance_squared"], c["layout_seed"]))
    reference = sorted(candidates, key=lambda c: c["layout_seed"])
    random.Random(20261120).shuffle(reference)
    cases, rejections = [], []
    for cohort, queue in (("near_anchor", near), ("random_reference", reference)):
        accepted = 0
        for candidate in queue:
            if candidate["raw_sha256"] in raw:
                continue
            seed = candidate["layout_seed"]
            folder = run / "resets" / cohort / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            try:
                result = attempt_regular(lab, folder, seed)
            except RuntimeError:
                log = (folder / "simulator.log").read_text(encoding="utf-8", errors="replace")
                if "already satisfies a placement goal" not in log and "sampled reset collides" not in log:
                    raise
                rejections.append({"layout_seed": seed, "reason": "native_reset_rejected"})
                raw.add(candidate["raw_sha256"])
                continue
            state = result["initial"]
            if state["raw_sha256"] != candidate["raw_sha256"]:
                raise ValueError(f"Batched and standalone native reset hashes differ for {seed}")
            if state["bddl_sha256"] != geometry["bddl_sha256"]:
                raise ValueError("The development reset changed the original Scene 08 task")
            if result["goal_at_reset"] or state["settled_sha256"] in settled:
                rejections.append({"layout_seed": seed, "reason": "goal_at_reset_or_settled_duplicate"})
                raw.add(state["raw_sha256"])
                continue
            raw.add(state["raw_sha256"])
            settled.add(state["settled_sha256"])
            cases.append({"cohort": cohort, "layout_seed": seed, "initial": state,
                          "distance_squared": candidate["distance_squared"], "inference_seeds": [POLICY_SEED],
                          "representative_video": accepted < 1})
            accepted += 1
            print("FROZEN", cohort, accepted, LAYOUTS_PER_GROUP, seed, flush=True)
            if accepted == LAYOUTS_PER_GROUP:
                break
        if accepted != LAYOUTS_PER_GROUP:
            raise RuntimeError(f"Incomplete development group: {cohort}")
    write_json(plan_path, {
        "schema": "v1-scene08-layout-development/1", "cases": cases, "gpu_uuid": gpu,
        "geometry_sha256": file_sha(profile_path), "excluded_prior_test_sha256": file_sha(prior_plan),
        "excluded_replay_sha256": file_sha(primary / "replay-manifest.json"), "rejections": rejections,
    })


def summarize(run):
    plan_path = run / "test-plan.json"
    plan = read_json(plan_path)
    plan_sha = file_sha(plan_path)
    if plan["geometry_sha256"] != file_sha(run / "geometry.json"):
        raise ValueError("Development geometry has changed")
    cases = plan["cases"]
    for key in ("raw_sha256", "settled_sha256"):
        if len({case["initial"][key] for case in cases}) != len(cases):
            raise ValueError(f"Duplicate development layout ({key})")
    groups = {}
    for cohort in ("near_anchor", "random_reference"):
        rows = [audit_episode(run, case, POLICY_SEED, "base", plan_sha, plan["gpu_uuid"], {})
                for case in cases if case["cohort"] == cohort]
        if len(rows) != LAYOUTS_PER_GROUP:
            raise ValueError(f"Unexpected size for {cohort}")
        groups[cohort] = {**per_arm(rows),
                          "failed_layouts": [row["layout_seed"] for row in rows if not row["success"]],
                          "nonpersistent_release_layouts": [row["layout_seed"] for row in rows if not row["release"]],
                          "rows": rows}
    near = groups["near_anchor"]["failed_layouts"]
    reference = groups["random_reference"]["failed_layouts"]
    return {"test_plan_sha256": plan_sha, "episodes": len(cases), "groups": groups,
            "nearby_selection_has_failure_headroom": len(near) >= 2 and len(near) > len(reference)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("profile", "freeze", "score"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--lab", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.stage == "score":
        result = summarize(args.run)
        write_json(args.run / "summary.json", result)
        print(json.dumps({**result, "groups": {
            name: {key: value for key, value in group.items() if key != "rows"}
            for name, group in result["groups"].items()
        }}, indent=2), flush=True)
        return
    if args.lab is None or not args.run.resolve().is_relative_to(args.lab.resolve()):
        parser.error("Use a personal lab and an output directory on its data disk")
    lab = Lab(args.lab.resolve())
    if args.stage == "profile":
        profile(lab, args.run)
    else:
        if not args.primary or not args.gpu_uuid:
            parser.error("Supply the primary run and the authorized GPU")
        freeze(lab, args.primary, args.run, args.gpu_uuid)


if __name__ == "__main__":
    main()
