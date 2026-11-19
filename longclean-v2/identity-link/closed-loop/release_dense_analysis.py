"""Sensor-only dense release opportunities with frozen ten-step observer state."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

import release_filter_analysis as base
import release_hand_check as hand


HAND_SHA = "3fa46a9cbb6f045faefd8d2c4f152a2ea35107a531b7b78c83f03f9b03f31178"
DENSE_RUNNER_SHA = "b9dfebf929e6591c0466ba7843241de2c664a73ca961787dccdf7eb501a534f4"
SCOPE = ("Four seen development layouts; hypothetical sampling of the frozen deposit rule only. "
         "No revised observer, live control, new threshold, or independent success rate was tested.")


def jsonl(directory, report, key):
    path = base.checked(directory, report[key], report[key + "_sha256"])
    return list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def keyed(rows, description):
    result = {r["step"]: r for r in rows}
    base.require(len(result) == len(rows) and list(result) == sorted(result), description)
    return result


def reconstruct(frames, anchors, report, expected, module):
    fork, end = report["fork"]["step"], report["window"]["end_step"]
    observer = module.PlacementObserver()
    observer.clear_view = frames[fork]
    actions, emitted, release = [], [], None
    for step in range(fork + 5, end + 1, 5):
        row = anchors[step]
        base.require(row["start_step"] == step - 5 and len(row["actions"]) == 5, "Changed anchor cadence")
        actions.extend(row["actions"])
        if (step - fork) % 10:
            continue
        base.require(len(actions) == 10, "Changed frozen observer action chunk")
        events = [{"step": step, **e} for e in observer.observe(frames[step], actions, step)]
        emitted.extend(events)
        if release is None and any(e["object"] == base.SAUCE and e["event"] == "pending_release" for e in events):
            base.require(step == expected["pending_step"] and observer.pending is not None
                         and observer.pending.target == base.SAUCE, "First sauce pending did not reproduce")
            release = observer.pending
        actions = []
    prior_events = [e for e in expected["original_events"] if e["step"] <= end]
    base.require(release is not None and not actions and base.project(emitted) == base.project(prior_events)
                 == base.project(report["observer_events"]), "Frozen ten-step observer history changed")
    base.require(np.array_equal(release.origin, expected["release_origin_xy_m"]), "Release origin changed")
    return release, emitted


def surface(frame, release, camera, module):
    """Describe visible novel color points; medians are not a tracked object pose."""
    result = hand.channel(frame, release, camera, module)
    xyz = base.cloud(frame, camera)
    mask = module.colored_pocket(frame, camera, release.target, release.origin)
    points = xyz[mask]
    if camera == 0:
        novel = (release.baseline.geometry["depths"][0] - frame.geometry["depths"][0])[mask] > .009
    else:
        novel = release.old_points.query(points)[0] > .012 if len(points) else np.zeros(0, dtype=bool)
    points = points[novel]
    base.require(len(points) == result["prehand_novel_pixels"], "Prehand surface count differs")
    center = np.median(points, axis=0) if len(points) else None
    result.update(prehand_novel_xyz=base.point_stats(points),
                  prehand_novel_center_m=None if center is None else center.tolist(),
                  prehand_novel_eef_offset_world_axes_m=None if center is None else (center - frame.geometry["eef"]).tolist())
    return result


def rates(record, previous):
    """Use only adjacent control steps; changing visibility can move these medians."""
    for camera, channel in enumerate(record["channels"]):
        channel["apparent_center_shift_m_per_control_step"] = None
        channel["apparent_relative_shift_m_per_control_step"] = None
        if previous is None or record["step"] - previous["step"] != 1:
            continue
        old = previous["channels"][camera]
        if channel["prehand_novel_center_m"] is not None and old["prehand_novel_center_m"] is not None:
            for source, target in (("prehand_novel_center_m", "apparent_center_shift_m_per_control_step"),
                                   ("prehand_novel_eef_offset_world_axes_m", "apparent_relative_shift_m_per_control_step")):
                channel[target] = (np.asarray(channel[source]) - old[source]).tolist()
    record["eef_shift_m_per_control_step"] = None if previous is None else (
        np.asarray(record["eef_m"]) - previous["eef_m"]).tolist()


def decision(row):
    return {k: row[k] for k in ("step", "age_steps", "gap_m", "pixels", "posthoc_goal_true",
                                "posthoc_held", "posthoc_correct_released")}


def sampling(records, fork, stride):
    eligible = [r for r in records if r["candidate_if_sampled"] and (r["step"] - fork) % stride == 0]
    return {"stride": stride, "candidate_steps": [r["step"] for r in eligible],
            "first": None if not eligible else decision(eligible[0]),
            "safe_candidate_steps": [r["step"] for r in eligible if r["posthoc_correct_released"]],
            "unsafe_candidate_steps": [r["step"] for r in eligible if not r["posthoc_correct_released"]]}


def analyze(directory, report, expected, module):
    p = expected["pending_step"]
    fork = report["fork"]["step"]
    start, end = p - 10, p + 30
    base.require(report["window"] == {"pending_step": p, "start_step": start, "end_step": end, "stride": 1}
                 and report["observer_stride"] == 10 and fork <= start and (start - fork) % 5 == 0,
                 "Dense window changed")
    archived = keyed(report["frames"], "Repeated or unordered frame")
    base.require(list(archived) == list(range(fork, start, 5)) + list(range(start, end + 1)), "Missing dense frames")
    native = keyed(jsonl(directory, report, "actions"), "Repeated or unordered actions")
    physical = keyed(jsonl(directory, report, "grasp_audit"), "Repeated or unordered physical audit")
    anchors = keyed(jsonl(directory, report, "source_anchors"), "Repeated or unordered original anchor")
    frames = {step: base.load_frame(directory, row) for step, row in archived.items()}
    for step, record in archived.items():
        audit, row = physical[step], native[step]
        base.require(audit == record["grasp_audit"] and audit["state_sha256"] == record["state_sha256"] == row["state_sha256"]
                     and record["observation_sha256"] == row["observation_sha256"]
                     and audit["goals"] == row["audit"]["goals"] and set(audit["held_by_target"]) == set(base.GOALS)
                     and all(type(x) is bool for x in audit["held_by_target"].values())
                     and frames[step].gap == record["grip_gap"], "Dense frame physical join changed")
        if step in anchors:
            base.require(record["state_sha256"] == anchors[step]["state_sha256"]
                         and record["observation_sha256"] == anchors[step]["observation_sha256"], "Original state/observation anchor changed")
    release, emitted = reconstruct(frames, anchors, report, expected, module)
    prior_frames = {r["step"]: r for r in expected["frames"]}
    records, previous = [], None
    for step in range(start, end + 1):
        frame = frames[step]
        if step < p:
            records.append({"step": step, "age_steps": step - p, "context_only": True, "gap_m": frame.gap,
                "target_color_xyz": [base.point_stats(base.cloud(frame, c)[base.color_mask(frame.images[c], base.LABELS[base.SAUCE])])
                                     for c in range(2)]})
            continue
        channels = [surface(frame, release, c, module) for c in range(2)]
        pixels, center = module.deposit(frame, release)
        base.require(pixels == [c["existing_final_pixels"] for c in channels], "Frozen deposit final counts differ")
        qualifies = bool(0 < step - p <= 30 and frame.gap >= .067 and center is not None)
        # Simulator truth is joined only after the observation-only computation above.
        audit = physical[step]
        row = {"step": step, "age_steps": step - p, "context_only": False, "gap_m": frame.gap,
               "eef_m": frame.geometry["eef"].tolist(), "original_10_step_sample": (step - fork) % 10 == 0,
               "original_5_step_sample": (step - fork) % 5 == 0, "candidate_if_sampled": qualifies,
               "pixels": pixels, "center_m": None if center is None else center.tolist(), "channels": channels,
               "posthoc_goal_true": audit["goals"][base.SAUCE], "posthoc_held": audit["held_by_target"][base.SAUCE],
               "posthoc_correct_released": bool(audit["goals"][base.SAUCE] and not audit["held_by_target"][base.SAUCE])}
        if step in prior_frames:
            before = prior_frames[step]
            base.require(all(row[k] == before[k] for k in ("gap_m", "pixels", "center_m", "candidate_if_sampled",
                         "posthoc_goal_true", "posthoc_held", "posthoc_correct_released")), "Prior five-step analysis differs")
        rates(row, previous)
        records.append(row)
        previous = row
    post = [r for r in records if not r["context_only"]]
    grids = {str(stride): sampling(post, fork, stride) for stride in (1, 5, 10)}
    original = next((e for e in emitted if e["object"] == base.SAUCE and e["event"] == "in_basket"
                     and e.get("release_step") == p), None)
    base.require((None if original is None else original["step"]) ==
                 (None if grids["10"]["first"] is None else grids["10"]["first"]["step"]), "Original ten-step decision differs")
    prior_first = expected["first_hypothetical_5_step_commit"]
    base.require((None if prior_first is None else decision(prior_first)) == grids["5"]["first"], "Original five-step first differs")
    return {"layout_seed": expected["layout_seed"], "pending_step": p, "original_events": emitted,
            "sampling_comparison": grids, "frames": records,
            "motion_caveat": "Point-set median shifts in world axes per control step; changing visible pixels prevent treating them as tracked object velocity."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "analysis", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    root, out = args.run.resolve(), args.out.resolve()
    base.require(not out.exists() and base.sha(args.analysis) == hand.ANALYSIS_SHA, "Changed prior analysis or existing output")
    accepted = base.read(args.analysis)
    base.require(base.sha(Path(base.__file__)) == accepted["analysis_source_sha256"]
                 and base.sha(Path(hand.__file__)) == HAND_SHA and accepted["observer_sha256"] == base.OBSERVER_SHA,
                 "Changed frozen helper or observer")
    base.require(base.sha(Path(inspect.getsourcefile(base.basket_pixels))) == accepted["basket_pixels_source_sha256"], "Changed geometry")
    base.require(base.sha(Path(__file__).with_name("release_dense_replay.py")) == DENSE_RUNNER_SHA, "Changed dense runner")
    module = base.load_exact("dense_analysis_frozen_observer", base.PLACEMENT / "observer.py", base.OBSERVER_SHA)
    cases, sources = [], []
    for card, layouts in base.CASES.items():
        manifest_path = root / f"release-dense-{card}.json"
        manifest = base.read(manifest_path)
        plan_path = base.checked(root, f"release-dense-plan-{card}.json", manifest["plan_sha256"])
        plan = base.read(plan_path)
        base.require(manifest["schema"] == "v2-stage4-release-dense-manifest/1" and manifest["complete"] is True
                     and manifest["card"] == card and [r["layout_seed"] for r in manifest["records"]] == list(layouts)
                     and plan["observer_sha256"] == base.OBSERVER_SHA and plan["prior_audit_sha256"] == base.PRIOR_SHA
                     and plan["source_sha256"] == DENSE_RUNNER_SHA,
                     "Incomplete or changed dense assignment")
        sources.append({"card": card, "manifest_sha256": base.sha(manifest_path), "plan_sha256": base.sha(plan_path),
                        "runner_sha256": plan["source_sha256"]})
        for item in manifest["records"]:
            path = base.checked(root, item["report"], item["report_sha256"])
            report = base.read(path)
            expected, = [r for r in accepted["cases"] if r["layout_seed"] == item["layout_seed"]]
            base.require(report["schema"] == "v2-stage4-release-dense/1" and report["plan_sha256"] == base.sha(plan_path)
                         and report["layout_seed"] == item["layout_seed"] and report["observer_sha256"] == base.OBSERVER_SHA
                         and report["prior_audit_sha256"] == base.PRIOR_SHA
                         and report["prior_report_sha256"] == expected["prior_report_sha256"]
                         and report["source_sha256"] == DENSE_RUNNER_SHA, "Changed dense report provenance")
            result = analyze(path.parent, report, expected, module)
            result.update(report=item["report"], report_sha256=item["report_sha256"])
            cases.append(result)
            print("DENSE_ANALYSIS", result["layout_seed"], {k: v["first"] for k, v in result["sampling_comparison"].items()}, flush=True)
    base.write(out, {"schema": "v2-release-dense-analysis/1", "scope": SCOPE, "analysis_source_sha256": base.sha(Path(__file__)),
        "prior_analysis_sha256": hand.ANALYSIS_SHA, "observer_sha256": base.OBSERVER_SHA, "hand_helper_sha256": HAND_SHA,
        "sources": sources, "cases": cases, "online_release": False, "independent_pilot_release": False})


if __name__ == "__main__":
    main()
