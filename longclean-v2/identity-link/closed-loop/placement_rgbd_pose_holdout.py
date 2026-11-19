"""Freeze a two-camera pose estimator and score new native held-state forks."""

import argparse
import os
from pathlib import Path

import numpy as np

from stage4_fixture import require, validate_study
from common import SOUP, read, sha, write
from placement_holdout import inspect_forks
from placement_holdout_supplement import collect, load_reference_sources, make_plan as reference_plan
from placement_orientation_feedback import DEV, SELECTOR_SHA, audit_pairs, choices_for, execute_pairs
import placement_preopen_holdout as preopen
from placement_rgbd_pose_sensor import SoupMesh, asset_path, estimate, fit, score_estimate
from soup_placement_feasibility_v2 import pinned, read_lines
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "placement_rgbd_pose_protocol.md"
DEV_SOURCE = "stage4-placement-orientation-935636-dev"
PRIMARY = "stage4-placement-holdout-935600"
SUPPLEMENT = "stage4-placement-holdout-supplement-935600"
PRIOR_ROOTS = (PRIMARY, SUPPLEMENT, "stage4-placement-preopen-935632", "stage4-placement-orientation-935636")
MODEL_SOURCE = HERE / "placement_rgbd_pose_sensor.py"
ARM_NAMES = ("fixed", "feedback")


def native_assets_ready():
    from actual_physics_simulator import load_native

    load_native().ensure_libero_config()
    return asset_path()


def read_frame(arm, frame):
    path = pinned(arm / frame["file"], frame["sha256"])
    observation(path, frame)
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def make_calibration(dev, output):
    from scipy.spatial.transform import Rotation

    path = dev / "orientation-pairs.json"
    pairs = read(path)
    require(set(pairs) == set(map(str, DEV)), "Development source changed")
    centers, rotations, source = [], [], []
    for seed in sorted(DEV):
        arm = dev / "pairs" / str(seed) / "fixed"
        result = read(pinned(arm / "result.json", pairs[str(seed)]["fixed"]))
        trace = read_lines(pinned(arm / "physics.jsonl", result["file_sha256"]["physics.jsonl"]))
        fork, frame = trace[0], result["frames"][0]
        measured = read_frame(arm, frame)
        require(frame["state_sha256"] == fork["state_sha256"] and fork["held"][SOUP], "Changed calibration fork")
        wrist = measured["camera2_extrinsic"][:3, :3]
        pose = fork["body_poses"][SOUP]
        centers.append(wrist.T @ (np.asarray(pose["position_m"]) - measured["eef_pos"]))
        rotations.append(wrist.T @ np.asarray(pose["rotation"]))
        source.append({"seed": seed, "fork_sha256": fork["state_sha256"], "frame_sha256": frame["sha256"]})
    mean_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
    data = {"grip_center_m": np.median(centers, axis=0).tolist(), "grip_rotation": mean_rotation.tolist(),
            "development": source, "source_index_sha256": sha(path), "mesh_sha256": sha(native_assets_ready()),
            "model_source_sha256": sha(MODEL_SOURCE), "protocol_sha256": sha(PROTOCOL)}
    write(output, data)
    print("RGBD_CALIBRATION", sorted(DEV), sha(output), flush=True)


def calibration_for(path, expected):
    data = read(pinned(path, expected))
    require(data["model_source_sha256"] == sha(MODEL_SOURCE) and data["protocol_sha256"] == sha(PROTOCOL),
            "The estimator or protocol changed after calibration")
    require([row["seed"] for row in data["development"]] == sorted(DEV), "Unexpected calibration data")
    return data


def plan_for(calibration_sha):
    seen, initial = set(), set()
    prior_hashes = {}
    for name in PRIOR_ROOTS:
        parent = Path(name)
        path = parent / "collection.json"
        collection = read(path)
        prior_hashes[name] = sha(path)
        for row in collection["attempts"]:
            if not row["eligible"]:
                continue
            seed = row["layout_seed"]
            record = read(pinned(parent / "sources" / str(seed) / "result.json", row["result_sha256"]))
            seen.add(seed)
            initial.add(record["initial"]["state_sha256"])
    reference = reference_plan(Path(PRIMARY))
    groups = []
    for group in reference["groups"]:
        candidates = [row for row in group["candidates"] if row["seed"] not in seen
                      and (group["name"] == "cheese" or row["seed"] in (935640, 935641))]
        if candidates:
            groups.append({**group, "candidates": candidates, "quota": min(2, len(candidates))})
    both = [g for g in groups if g["name"] == "both"]
    require(len(both) == 1 and [row["seed"] for row in both[0]["candidates"]] == [935640, 935641],
            "The untouched crowded prefix pool changed")
    pinned(HERE / "placement_preopen_holdout.py", SELECTOR_SHA)
    return {**reference, "schema": "v2-two-camera-pose-validation/1", "groups": groups,
            "protocol_sha256": sha(PROTOCOL), "fixture_source_sha256": sha(Path(__file__)),
            "model_source_sha256": sha(MODEL_SOURCE), "calibration_sha256": calibration_sha,
            "historical_initial_sha256": sorted(initial), "historical_collection_sha256": prior_hashes}


def prior_disjoint(plan, sources):
    initial = [case["result"]["initial"]["state_sha256"] for case in sources.values()]
    require(len(initial) == len(set(initial)) and not set(initial).intersection(plan["historical_initial_sha256"]),
            "A new held-state fixture repeated an evaluated layout")


def physical_summary(rows):
    eligible = [row for row in rows if row["held"] and row["xyz_qualified"]]
    result = {"layouts": len({r["seed"] for r in rows}), "unique_endpoints": len(rows), "eligible_endpoints": len(eligible),
              "estimates": sum(r["sensor"]["angle_deg"] is not None for r in eligible),
              "truth_allow": sum(r["truth"] == "allow" for r in eligible),
              "truth_block": sum(r["truth"] == "block" for r in eligible)}
    for decision_key in ("decision", "raw_decision"):
        pairs = [(r["truth"], r["sensor"][decision_key]) for r in eligible]
        result[decision_key] = {"correct": sum(x == y for x, y in pairs),
                               "false_allow": pairs.count(("block", "allow")),
                               "false_block": pairs.count(("allow", "block")),
                               "abstain": sum(choice == "abstain" for _, choice in pairs)}
    important = [row for row in eligible if row["truth"] == "block"]
    directional = [row for row in important if row.get("direction_cosine") is not None]
    result["overlimit_direction"] = {"total": len(important), "estimates": len(directional),
        "positive_projection": sum(row["direction_cosine"] > 0 for row in directional),
        "bounded_reduces_angle": sum(row["bounded_reduces_angle"] for row in directional),
        "bounded_recovers_under_limit": sum(row["bounded_recovers_under_limit"] for row in directional)}
    return result


def run_score(root, calibration, output, mode):
    model = SoupMesh(native_assets_ready())
    require(model.mesh_sha256 == calibration["mesh_sha256"], "Tracker CAD asset changed")
    pairs_path = root / "orientation-pairs.json"
    pairs = read(pairs_path)
    rows, coverage = [], []
    for seed, arms in sorted(pairs.items()):
        seen = set()
        reference_fit, reference_meta = None, None
        if not arms:
            coverage.append({"seed": int(seed), "status": "no_clear_release_point"})
            continue
        for name in ARM_NAMES:
            arm = root / "pairs" / seed / name
            result = read(pinned(arm / "result.json", arms[name]))
            end = next((s["end_step"] for s in result["phases"] if s["phase"] == "lower"), None)
            if end is None:
                coverage.append({"seed": int(seed), "arm": name, "status": "lower_phase_not_reached"})
                continue
            frames = {m["step"]: m for m in result["frames"]}
            reference_frame, endpoint = result["frames"][0], frames.get(end)
            if endpoint is None:
                coverage.append({"seed": int(seed), "arm": name, "status": "unsampled_lower_endpoint", "step": end})
                continue
            if endpoint["state_sha256"] in seen:
                coverage.append({"seed": int(seed), "arm": name, "status": "duplicate_endpoint", "step": end})
                continue
            seen.add(endpoint["state_sha256"])
            # Inference has a sensor-only interface. Physical traces below are
            # accessed separately for the official eligibility and scores.
            reference = read_frame(arm, reference_frame)
            if reference_fit is None:
                reference_fit = fit(reference, calibration, model)
                reference_meta = reference_frame["observation_sha256"]
            require(reference_meta == reference_frame["observation_sha256"], "Reference views differ between arms")
            measured = read_frame(arm, endpoint)
            predicted = estimate(reference, measured, calibration, model, reference_fit=reference_fit)
            trace = read_lines(pinned(arm / "physics.jsonl", result["file_sha256"]["physics.jsonl"]))
            fork, last = trace[0], trace[end - trace[0]["step"]]
            require(fork["state_sha256"] == reference_frame["state_sha256"] and last["state_sha256"] == endpoint["state_sha256"],
                    "Scoring truth belongs to a different RGB-D frame")
            evaluation = score_estimate(predicted, fork, last, result["target_m"])
            row = {"seed": int(seed), "arm": name, "fork_step": fork["step"], "step": end,
                   "fork_sha256": fork["state_sha256"], "state_sha256": endpoint["state_sha256"],
                   "reference_frame_sha256": reference_frame["sha256"], "endpoint_frame_sha256": endpoint["sha256"],
                   **evaluation}
            rows.append(row)
            print("RGBD_POSE", seed, name, round(row["truth_angle_deg"], 4), predicted["angle_deg"],
                  predicted["decision"], row.get("direction_cosine"), row.get("bounded_residual_deg"), flush=True)
    summary = physical_summary(rows)
    write(output, {"source_index_sha256": sha(pairs_path), "split": mode, "mesh_sha256": model.mesh_sha256,
                   "model_source_sha256": sha(MODEL_SOURCE), "summary": summary, "coverage": coverage, "rows": rows})
    print("RGBD_POSE_SUMMARY", mode, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "dev-score", "collect", "inspect", "screen", "pair", "audit", "score"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-sha")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "calibrate":
        make_calibration(Path(DEV_SOURCE), args.calibration)
        return
    expected = args.calibration_sha
    require(expected is not None, "Pin calibration SHA-256")
    calibration = calibration_for(args.calibration, expected)
    if args.stage == "dev-score":
        run_score(Path(DEV_SOURCE), calibration, args.output, "development")
        return
    plan = plan_for(expected)
    root = args.output
    phase = args.stage
    if phase in ("collect", "inspect", "pair"):
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin a free native render device")
    if phase in ("collect", "inspect", "screen", "pair"):
        bddl, _ = validate_study(args.study.resolve())
    if phase == "collect":
        collect(root, Path(PRIMARY), bddl, plan)
        return
    _, sources = load_reference_sources(root, plan)
    prior_disjoint(plan, sources)
    require(sources, "No newly qualified held fork")
    if phase == "inspect":
        inspect_forks(root, bddl, sources)
    elif phase == "screen":
        preopen.screen(root, Path(SUPPLEMENT), bddl, sources)
    elif phase == "pair":
        execute_pairs(root, bddl, sources, choices_for(root, sources))
    elif phase == "audit":
        audit_pairs(root, sources, choices_for(root, sources))
    else:
        run_score(root, calibration, root / "rgbd-pose-score.json", "new_held_forks")


if __name__ == "__main__":
    main()
