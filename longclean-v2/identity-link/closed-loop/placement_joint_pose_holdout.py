"""Freeze a paired relative pose fit after regression and score new soup holds."""

import argparse
import os
from pathlib import Path

from stage4_fixture import require, validate_study
from common import read, sha, write
from placement_holdout import inspect_forks
from placement_holdout_supplement import collect, load_reference_sources, make_plan as reference_plan
from placement_orientation_feedback import SELECTOR_SHA, audit_pairs, choices_for, execute_pairs
import placement_preopen_holdout as preopen
import placement_rgbd_pose_holdout as previous
from placement_joint_pose_sensor import solve_joint
from placement_rgbd_pose_sensor import SoupMesh, score_estimate
from soup_placement_feasibility_v2 import pinned, read_lines


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "placement_joint_pose_protocol.md"
SENSOR = HERE / "placement_joint_pose_sensor.py"
PREVIOUS = "stage4-rgbd-pose-935640"
CALIBRATION_SHA = "45b6e462764a52c1a4aa3b2c8f68a500b22eceebd9454c89eab9e3f0c9fc08ef"
FRESH = (935614, 935615, 935616, 935617)
DEVELOPMENT = ((935613, "fixed"), (935612, "fixed"), (935612, "feedback"))


def freeze():
    return {"source_sha256": sha(Path(__file__)), "sensor_sha256": sha(SENSOR), "protocol_sha256": sha(PROTOCOL),
            "calibration_sha256": CALIBRATION_SHA}


def plan_for(regression_path):
    previous_attempts, previous_initials = set(), set()
    historical = {}
    for old in (*previous.PRIOR_ROOTS, PREVIOUS):
        root = Path(old)
        path = root / "collection.json"
        historical[old] = sha(path)
        for r in read(path)["attempts"]:
            if r["eligible"]:
                seed = r["layout_seed"]
                source = read(pinned(root / "sources" / str(seed) / "result.json", r["result_sha256"]))
                previous_attempts.add(seed)
                previous_initials.add(source["initial"]["state_sha256"])
    plan = reference_plan(Path(previous.PRIMARY))
    group, = [g for g in plan["groups"] if g["name"] == "cheese"]
    candidates = [row for row in group["candidates"] if row["seed"] in FRESH]
    require(tuple(row["seed"] for row in candidates) == FRESH and not previous_attempts.intersection(FRESH),
            "A fresh held-state candidate was previously evaluated")
    pinned(HERE / "placement_preopen_holdout.py", SELECTOR_SHA)
    return {**plan, **freeze(), "schema": "v2-joint-pose-holdout/1", "groups": [
        {**group, "quota": len(FRESH), "candidates": candidates}],
        "historical_initial_sha256": sorted(previous_initials),
        "historical_collection_sha256": historical, "regression_sha256": sha(regression_path)}


def episode_frame(root, index, seed, phase):
    arm = root / "pairs" / str(seed) / phase
    result = read(pinned(arm / "result.json", index[str(seed)][phase]))
    end = next((p["end_step"] for p in result["phases"] if p["phase"] == "lower"), None)
    endpoint = next((f for f in result["frames"] if f["step"] == end), None)
    return arm, result, endpoint


def score_case(seed, name, arm, result, endpoint, calibration, model, depth_control):
    reference_frame = result["frames"][0]
    reference = previous.read_frame(arm, reference_frame)
    measured = previous.read_frame(arm, endpoint)
    fits = solve_joint((reference, measured), calibration, model, include_depth_control=depth_control)
    # Score only after the sensor estimate is computed.
    trace = read_lines(pinned(arm / "physics.jsonl", result["file_sha256"]["physics.jsonl"]))
    fork, last = trace[0], trace[endpoint["step"] - trace[0]["step"]]
    require(fork["state_sha256"] == reference_frame["state_sha256"] and last["state_sha256"] == endpoint["state_sha256"],
            "State and sensor frame provenance differ")
    rows = {}
    for method, predicted in fits.items():
        evaluated = score_estimate(predicted, fork, last, result["target_m"])
        rows[method] = {"seed": seed, "arm": name, "fork_step": fork["step"], "step": last["step"],
            "fork_sha256": fork["state_sha256"], "state_sha256": last["state_sha256"],
            "reference_frame_sha256": reference_frame["sha256"], "endpoint_frame_sha256": endpoint["sha256"], **evaluated}
        print("JOINT_POSE", method, seed, name, round(evaluated["truth_angle_deg"], 4), predicted["angle_deg"],
              predicted["decision"], evaluated.get("direction_cosine"), flush=True)
    return rows


def evaluate(root, output, calibration, *, regression):
    model = SoupMesh(previous.native_assets_ready())
    require(model.mesh_sha256 == calibration["mesh_sha256"], "The visual mesh hash changed")
    index_path = root / "orientation-pairs.json"
    index = read(index_path)
    cases = list(DEVELOPMENT) if regression else [
        (int(seed), arm) for seed in sorted(index) for arm in previous.ARM_NAMES if index[seed]]
    coverage = [{"seed": int(seed), "reason": "no_clear_release_point"} for seed in sorted(index)
                if not regression and not index[seed]]
    results = {"joint_texture": []}
    if regression:
        results["joint_depth"] = []
    seen = set()
    for seed, name in cases:
        arm, result, end = episode_frame(root, index, seed, name)
        if end is None:
            coverage.append({"seed": seed, "arm": name, "reason": "no_sampled_lower_endpoint"})
            continue
        key = seed, end["state_sha256"]
        if key in seen:
            coverage.append({"seed": seed, "arm": name, "reason": "duplicate_endpoint"})
            continue
        seen.add(key)
        for method, row in score_case(seed, name, arm, result, end, calibration, model, regression).items():
            results[method].append(row)
    summaries = {method: previous.physical_summary(rows) for method, rows in results.items()}
    report = {**freeze(), "source_index_sha256": sha(index_path), "split": "regression" if regression else "holdout",
              "mesh_sha256": model.mesh_sha256, "summary": summaries, "coverage": coverage, "rows": results}
    if regression:
        rows = {(r["seed"], r["arm"]): r for r in results["joint_texture"]}
        known, challenge = rows[935613, "fixed"], rows[935612, "fixed"]
        report["regression_passed"] = (known["sensor"]["decision"] == "allow"
                                       and challenge["sensor"]["raw_decision"] != "allow"
                                       and (challenge.get("direction_cosine") or 0.) > 0.)
    write(output, report)
    print("JOINT_POSE_SUMMARY", report["split"], summaries, "regression_gate", report.get("regression_passed"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("regression", "collect", "inspect", "screen", "pair", "audit", "score"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--regression-sha")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = previous.calibration_for(args.calibration, CALIBRATION_SHA)
    phase, dest = args.stage, args.output
    if phase == "regression":
        evaluate(Path(PREVIOUS), args.regression_report, calibration, regression=True)
        return
    old = read(pinned(args.regression_report, args.regression_sha))
    require(all(old[key] == value for key, value in freeze().items()) and old["regression_passed"],
            "Do not evaluate new data before the frozen regression passes")
    plan = plan_for(args.regression_report)
    if phase in ("collect", "inspect", "pair"):
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl",
                "Pin an idle CUDA and EGL card")
    if phase in ("collect", "inspect", "screen", "pair"):
        bddl, _ = validate_study(args.study.resolve())
    if phase == "collect":
        collect(dest, Path(previous.PRIMARY), bddl, plan)
        return
    _, sources = load_reference_sources(dest, plan)
    previous.prior_disjoint(plan, sources)
    require(sources, "No new held fork met the physical inclusion rule")
    if phase == "inspect":
        inspect_forks(dest, bddl, sources)
    elif phase == "screen":
        preopen.screen(dest, Path(previous.SUPPLEMENT), bddl, sources)
    elif phase == "pair":
        execute_pairs(dest, bddl, sources, choices_for(dest, sources))
    elif phase == "audit":
        audit_pairs(dest, sources, choices_for(dest, sources))
    else:
        evaluate(dest, dest / "joint-pose-score.json", calibration, regression=False)


if __name__ == "__main__":
    main()
