"""Measure a three-frame wrist fallback on archived sauce release trajectories."""

import argparse
from itertools import combinations
from math import dist
from pathlib import Path

import numpy as np

from stage4_fixture import require
from common import SAUCE, array_hash, read, sha, write
from dense_online_diagnostic import checked, jsonl
from dense_release_observer import DenseReleaseObserver, OPEN_GAP
from release_filter_analysis import load_frame


CONTROL_SHA = "8866d8b3627b7afe04fde0914237e39d3e8c64fc7b5b6684144fe3ecbce8ea82"
HOLDOUT_SHA = "1315345e2915197e0e301cc2aca926c567c8bfd911771920b8b3d0bda91a8213"
AGES = (10, 20, 30)
MIN_WRIST_PIXELS = 16
MAX_CENTER_SPREAD_M = .012
MIN_CAMERA_SPREAD_M = .010


def wrist_sample(module, frame, origin, step, pending=None):
    points = module.cloud(frame, 1)[module.colored_pocket(frame, 1, SAUCE, origin)]
    if pending is not None and len(points):
        offset = points - frame.geometry["eef"]
        away = (offset[:, 2] < -.073) | (np.linalg.norm(offset[:, :2], axis=1) > .065)
        points = points[away & (pending.old_points.query(points)[0] > .012)]
    require(np.isfinite(points).all(), "Nonfinite wrist evidence")
    return {"step": step, "open": frame.gap >= OPEN_GAP, "pixels": len(points),
            "center_m": np.median(points, axis=0).tolist() if len(points) else None,
            "camera_m": frame.geometry["extrinsics"][1][:3, 3].tolist()}


def measure(samples):
    supported = len(samples) == len(AGES) and all(s["open"] and s["pixels"] >= MIN_WRIST_PIXELS for s in samples)
    center_spread = (max(dist(left["center_m"], right["center_m"]) for left, right in combinations(samples, 2))
                     if supported else None)
    camera_spread = max((dist(left["camera_m"], right["camera_m"]) for left, right in combinations(samples, 2)), default=0.)
    return {"samples": samples, "center_spread_m": center_spread, "camera_spread_m": camera_spread,
            "would_confirm": bool(supported and center_spread <= MAX_CENTER_SPREAD_M and camera_spread >= MIN_CAMERA_SPREAD_M)}


def holdout(root):
    summary = read(checked(root, "summary.json", HOLDOUT_SHA))
    trace = jsonl(checked(root, "actions.jsonl", summary["actions_sha256"]))
    origin_row, = (r for r in trace if r["step"] == 240)
    pose = np.asarray(origin_row["audit"]["site_xyz"])
    require(array_hash(pose) == origin_row["observation_sha256"]["eef_pos"], "Release origin is not the recorded sensor pose")
    observer = DenseReleaseObserver()
    selected, evaluation = [], None
    for sample in summary["capture"]:
        if sample["step"] - 250 not in AGES:
            continue
        wrist = sample["channels"][1]
        require(wrist["colored_pocket_pixels"] == wrist["final_pixels"], "Cannot recover exact admitted set from this archive")
        record = sample["holdout_observation"]
        frame = load_frame(root, {**record, **record["fingerprint"]})
        result = wrist_sample(observer._frozen, frame, pose[:2], sample["step"])
        require(result["pixels"] == wrist["final_pixels"] and result["open"] is sample["open_gate"], "Changed holdout sensor evidence")
        selected.append(result)
        evaluation = sample["evaluation_only"]
    require([s["step"] - 250 for s in selected] == list(AGES), "Missing release sample")
    return {"layout_seed": summary["layout"], "frozen_abstained": True, **measure(selected), "evaluation_only": evaluation}


def control(root, entry):
    layout = entry["layout_seed"]
    folder = root / str(layout)
    report = read(checked(folder, "report.json", entry["report_sha256"]))
    actions = jsonl(checked(folder, report["actions"], report["actions_sha256"]))
    frames = report["frames"]
    require([r["step"] for r in frames] == [0, *[r["step"] for r in actions]], "Frame alignment differs")
    observer = DenseReleaseObserver()
    observer.start(load_frame(folder, frames[0]), 0)
    selected, abstentions, commits = {}, [], []
    for action, record in zip(actions, frames[1:], strict=True):
        step = record["step"]
        frame = load_frame(folder, record)
        pending = observer._coarse.pending
        if pending is not None and pending.target == SAUCE and step - pending.step in AGES:
            result = wrist_sample(observer._frozen, frame, pending.origin, step, pending)
            selected.setdefault(pending.step, []).append(result)
        for event in observer.observe(frame, action["actions"], step):
            if event["event"] == "in_basket":
                commits.append((event["object"], step))
            if event["event"] == "abstained" and event["object"] == SAUCE:
                require(pending is not None, "Release must be pending immediately before abstention")
                truth = record["grasp_audit"]
                abstentions.append({"release_step": pending.step, "step": step,
                                    **measure(selected.pop(pending.step, [])),
                                    "evaluation_only": {"official_goal": truth["goals"][SAUCE], "held": truth["held_by_target"][SAUCE]}})
    require(commits == [(e["object"], e["step"]) for e in report["arms"]["candidate"]["commits"]], "Archived commits did not replay")
    return {"layout_seed": layout, "treatment": entry["treatment"], "qualified": entry["qualified"],
            "frozen_commits": commits, "sauce_abstentions": abstentions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("controls", "holdout", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Result already exists")
    evidence = holdout(args.holdout)
    controls = read(checked(args.controls, "audit.json", CONTROL_SHA))
    require(sha(Path(__file__).with_name("dense_release_observer.py")) == controls["candidate_sha256"], "Frozen observer changed")
    cases = [control(args.controls, entry) for entry in controls["cases"]]
    require([c["layout_seed"] for c in cases] == list(range(929336, 929352)), "Incomplete control bank")
    write(args.out, {"schema": "v2-sauce-temporal-release-diagnostic/1", "producer_sha256": sha(Path(__file__)),
                     "inputs": {"controls": CONTROL_SHA, "holdout": HOLDOUT_SHA},
                     "criteria": {"ages": AGES, "min_wrist_pixels": MIN_WRIST_PIXELS,
                                  "max_center_spread_m": MAX_CENTER_SPREAD_M, "min_camera_spread_m": MIN_CAMERA_SPREAD_M},
                     "development_case": evidence, "controls": cases})
    print("SAUCE_TEMPORAL", evidence["would_confirm"], len(cases), flush=True)


if __name__ == "__main__":
    main()
