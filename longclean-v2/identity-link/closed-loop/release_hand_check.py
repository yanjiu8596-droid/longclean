"""CPU diagnostic of novelty before hand exclusion; never a revised observer."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import release_filter_analysis as base

ANALYSIS_SHA = "916e6202854d4221ac69143a44680bcacf64a0a56eae41a94fd9f0026cd825b8"


def reconstruct(directory, report, expected, module):
    frames = {r["step"]: base.load_frame(directory, r) for r in report["frames"]}
    path = base.checked(directory, report["actions"], report["actions_sha256"])
    rows = {r["step"]: r for r in map(json.loads, path.read_text().splitlines())}
    fork, end = report["fork"]["step"], report["window"]["end_step"]
    observer = module.PlacementObserver()
    observer.clear_view = frames[fork]
    actions, emitted, release = [], [], None
    for step in range(fork + 5, end + 1, 5):
        row = rows[step]
        base.require(row["start_step"] == step - 5 and len(row["actions"]) == 5, "Action cadence")
        actions.extend(row["actions"])
        if (step - fork) % 10:
            continue
        events = [{"step": step, **e} for e in observer.observe(frames[step], actions, step)]
        emitted.extend(events)
        if release is None and any(e["object"] == base.SAUCE and e["event"] == "pending_release" for e in events):
            base.require(step == expected["pending_step"] and observer.pending is not None, "Pending differs")
            release = observer.pending
        actions = []
    base.require(release is not None and base.project(emitted) == base.project(expected["original_events"]), "History differs")
    return frames, release


def channel(frame, release, camera, module):
    xyz = base.cloud(frame, camera)
    pocket = module.colored_pocket(frame, camera, release.target, release.origin)
    points = xyz[pocket]
    offset = points - frame.geometry["eef"]
    xy = np.linalg.norm(offset[:, :2], axis=-1)
    z_ok, xy_ok = offset[:, 2] < -.073, xy > .065
    if camera == 0:
        evidence = (release.baseline.geometry["depths"][0] - frame.geometry["depths"][0])[pocket]
        threshold, name = .009, "head_depth_change_m"
    else:
        evidence = release.old_points.query(points)[0] if len(points) else np.empty(0)
        threshold, name = .012, "wrist_old_color_distance_m"
    novel, hand = evidence > threshold, z_ok | xy_ok
    return {"camera": camera, "pocket_pixels": len(points), "prehand_novel_pixels": int(novel.sum()),
            "existing_final_pixels": int((novel & hand).sum()), "novel_rejected_by_hand": int((novel & ~hand).sum()),
            "pocket_xyz": base.point_stats(points), name: base.quantiles(evidence),
            "novel_eef_xy_distance_m": base.quantiles(xy[novel]),
            "novel_eef_z_offset_m": base.quantiles(offset[novel, 2]),
            "fixed_threshold_bins": {"z_only": int((novel & z_ok & ~xy_ok).sum()),
                "xy_only": int((novel & xy_ok & ~z_ok).sum()), "both": int((novel & z_ok & xy_ok).sum()),
                "neither": int((novel & ~hand).sum()), "novelty_gt_threshold": int(novel.sum()),
                "novelty_le_threshold": int((evidence <= threshold).sum()), "novelty_nan": int(np.isnan(evidence).sum())}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "analysis", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    base.require(not args.out.exists() and base.sha(args.analysis) == ANALYSIS_SHA, "Changed analysis or existing output")
    accepted = base.read(args.analysis)
    base.require(base.sha(Path(base.__file__)) == accepted["analysis_source_sha256"], "Changed helper")
    base.require(accepted["observer_sha256"] == base.OBSERVER_SHA, "Changed observer pin")
    base.require(base.sha(Path(inspect.getsourcefile(base.basket_pixels))) == accepted["basket_pixels_source_sha256"], "Changed geometry")
    module = base.load_exact("hand_check_frozen_observer", base.PLACEMENT / "observer.py", base.OBSERVER_SHA)
    cases = []
    for expected in accepted["cases"]:
        path = base.checked(args.run, expected["report"], expected["report_sha256"])
        report = base.read(path)
        frames, release = reconstruct(path.parent, report, expected, module)
        records = []
        for row in expected["frames"]:
            if row["context_only"]:
                continue
            frame = frames[row["step"]]
            channels = [channel(frame, release, c, module) for c in range(2)]
            final, _ = module.deposit(frame, release)
            base.require(final == row["pixels"] == [r["existing_final_pixels"] for r in channels], "Final counts differ")
            records.append({"step": row["step"], "age_steps": row["age_steps"], "gap_m": frame.gap,
                "within_original_age_gap": bool(0 < row["age_steps"] <= 30 and frame.gap >= .067),
                "posthoc_goal_true": row["posthoc_goal_true"], "posthoc_held": row["posthoc_held"], "channels": channels})
        cases.append({"layout_seed": expected["layout_seed"], "report_sha256": expected["report_sha256"], "frames": records})
    base.write(args.out, {"schema": "v2-release-hand-check/1", "analysis_sha256": ANALYSIS_SHA,
        "source_sha256": base.sha(Path(__file__)), "observer_sha256": base.OBSERVER_SHA, "cases": cases,
        "scope": "Seen-development ignored-finger diagnostic only; no revised observer or recovered episode evaluated.",
        "online_release": False, "independent_pilot_release": False})


if __name__ == "__main__":
    main()
