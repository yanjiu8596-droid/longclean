"""CPU-only decomposition of frozen release evidence on four exact replays."""
import argparse
from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from actual_progress_replay import OBSERVER_SHA, PLACEMENT, load_exact
from common import GOALS, LABELS, SAUCE, SOUP, array_hash, read, sha, write
from stage4_fixture import require
from vision import Frame, basket_pixels, cloud, color_mask


HERE = Path(__file__).resolve().parent
PRIOR_SHA = "34970f0017ca2909a2dcbd3e3138b83f0b579f250c8072b7076ecf72e7b2bb7a"
CASES = {2: (931000, 931200), 3: (931005, 931201)}
META = {"assigned", "writer", "applied", "matches_audit", "held_soup", "held_by_target",
        "released_in_audit", "previous_sampled_held_step", "evidence_file", "evidence_sha256"}


def checked(root, name, checksum):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve())
            and path.is_file() and sha(path) == checksum, f"Changed or unsafe input: {name}")
    return path


def load_frame(directory, record):
    path = checked(directory, record["file"], record["sha256"])
    with np.load(path, allow_pickle=False) as data:
        hashes = record["observation_sha256"]
        require(set(data.files) == set(hashes) and all(array_hash(data[k]) == value for k, value in hashes.items()),
                f"Observation changed: {path}")
    return replace(Frame.read(path), source=None)


def quantiles(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"n": int(values.size), "finite": int(finite.size),
            "positive_infinite": int(np.isposinf(values).sum()), "negative_infinite": int(np.isneginf(values).sum()),
            "nan": int(np.isnan(values).sum()),
            "q0_q25_q50_q75_q100": None if not finite.size else np.quantile(finite, [0, .25, .5, .75, 1]).tolist()}


def point_stats(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    return {"n": int(len(points)), "finite": int(len(finite)),
            "bbox_min_m": None if not len(finite) else finite.min(axis=0).tolist(),
            "bbox_max_m": None if not len(finite) else finite.max(axis=0).tolist(),
            "median_xyz_m": None if not len(finite) else np.median(finite, axis=0).tolist()}


def decompose(frame, release, module):
    """Duplicate only masks, and require equality with both frozen functions."""
    require(release.target == SAUCE, "Expected the first sauce pending release")
    channels, final_counts, centers, drawing_masks = [], [], [], []
    for camera in range(2):
        xyz = cloud(frame, camera)
        x, y, z = np.moveaxis(xyz, -1, 0)
        label = color_mask(frame.images[camera], LABELS[release.target])
        x_ok = (x > -.055) & (x < .065)
        y_ok = (y > .195) & (y < .335)
        z_ok = (z > .468) & (z < .57)
        radial = np.hypot(x - release.origin[0], y - release.origin[1])
        radius_ok = radial < .075
        pocket = x_ok & y_ok & z_ok & radius_ok
        require(np.array_equal(pocket, basket_pixels(frame, camera, release.origin)),
                "Basket mask decomposition differs from frozen basket_pixels")
        require(np.array_equal(label & pocket, module.colored_pocket(frame, camera, SAUCE, release.origin)),
                "Target pocket differs from frozen observer")
        offset = xyz - frame.geometry["eef"]
        xy_clearance = np.linalg.norm(offset[..., :2], axis=-1)
        finger = (offset[..., 2] < -.073) | (xy_clearance > .065)
        masks = [("target_color", label), ("basket_x", label & x_ok),
                 ("basket_xy", label & x_ok & y_ok), ("basket_xyz", label & x_ok & y_ok & z_ok),
                 ("release_radius", label & pocket), ("finger_clear", label & pocket & finger)]
        depth_delta = release.baseline.geometry["depths"][camera] - frame.geometry["depths"][camera]
        if camera == 0:
            final = masks[-1][1] & (depth_delta > .009)
            novelty = None
            masks.append(("head_depth_change", final))
        else:
            final = masks[-1][1].copy()
            distances = release.old_points.query(xyz[final])[0] if np.any(final) else np.empty(0)
            final[final] = distances > .012
            novelty = quantiles(distances)
            masks.append(("wrist_novelty", final))
        count = int(final.sum())
        center = np.median(xyz[final], axis=0) if count else None
        final_counts.append(count)
        centers.append(center)
        previous = None
        waterfall = []
        for name, mask in masks:
            n = int(mask.sum())
            waterfall.append({"filter": name, "remaining_pixels": n,
                              "removed_pixels": None if previous is None else previous - n})
            previous = n
        channels.append({"camera": camera, "filters": waterfall,
                         "individual_rejection_counts_on_target_color": {
                             "x": int((label & ~x_ok).sum()), "y": int((label & ~y_ok).sum()),
                             "z": int((label & ~z_ok).sum()), "release_radius": int((label & ~radius_ok).sum())},
                         "target_color_xyz": point_stats(xyz[label]),
                         "color_in_xy_xyz": point_stats(xyz[label & x_ok & y_ok]),
                         "pocket_xyz": point_stats(xyz[label & pocket]), "final_xyz": point_stats(xyz[final]),
                         "target_color_release_radius_m": quantiles(radial[label]),
                         "pocket_eef_xy_distance_m": quantiles(xy_clearance[label & pocket]),
                         "pocket_eef_z_offset_m": quantiles(offset[..., 2][label & pocket]),
                         "finger_clear_depth_change_m": quantiles(depth_delta[label & pocket & finger]),
                         "wrist_old_color_nearest_distance_m": novelty})
        drawing_masks.append({name: mask for name, mask in masks})
    chosen = 0 if final_counts[0] >= 34 else 1 if final_counts[1] >= 45 else None
    center = None if chosen is None else centers[chosen]
    frozen_counts, frozen_center = module.deposit(frame, release)
    require(final_counts == frozen_counts and (center is None) == (frozen_center is None)
            and (center is None or np.array_equal(center, frozen_center)),
            "Decomposition differs from frozen deposit count, selection or center")
    return {"channels": channels, "pixels": final_counts, "qualifying_channel": chosen,
            "center_m": None if center is None else center.tolist()}, drawing_masks


def project(events):
    return [{k: v for k, v in event.items() if k not in META} for event in events]


def montage(path, frames, records, masks, selected):
    require(not path.exists(), "Montage output already exists")
    height, width = len(selected) * 290, 4 * 256
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, step in enumerate(selected):
        frame, record, camera_masks = frames[step], records[step], masks[step]
        top = row_index * 290
        draw.text((5, top + 3), f"step {step} age {record['age_steps']} counts {record['pixels']} gap {record['gap_m']:.4f}", fill="black")
        for camera in range(2):
            rgb = np.asarray(frame.images[camera]).copy()
            overlay = rgb.copy()
            pocket = camera_masks[camera]["release_radius"]
            final_name = "head_depth_change" if camera == 0 else "wrist_novelty"
            final = camera_masks[camera][final_name]
            overlay[pocket] = (.35 * overlay[pocket] + .65 * np.array([255, 180, 0])).astype(np.uint8)
            overlay[final] = (.25 * overlay[final] + .75 * np.array([0, 255, 0])).astype(np.uint8)
            left = camera * 512
            canvas.paste(Image.fromarray(rgb), (left, top + 28))
            canvas.paste(Image.fromarray(overlay), (left + 256, top + 28))
    canvas.save(path)


def analyze_case(directory, report, prior_report, module):
    fork = report["fork"]["step"]
    pending_step, start, end = (report["window"][k] for k in ("pending_step", "start_step", "end_step"))
    require(start == pending_step - 20 and end == pending_step + 40 and report["window"]["stride"] == 5,
            "Unexpected release window")
    require(report["fork"] == prior_report["fork"] and report["source"] == prior_report["source"], "Fork/source differs")
    row_file = checked(directory, report["actions"], report["actions_sha256"])
    import json
    native_rows = [json.loads(x) for x in row_file.read_text().splitlines()]
    rows = {r["step"]: r for r in native_rows if r["step"] >= fork}
    side = checked(directory, report["grasp_audit"], report["grasp_audit_sha256"])
    physical_rows = list(map(json.loads, side.read_text().splitlines()))
    physical = {r["step"]: r for r in physical_rows}
    archived = {r["step"]: r for r in report["frames"]}
    require(len(physical) == len(physical_rows) and len(archived) == len(report["frames"])
            and list(archived) == list(range(fork, end + 1, 5)), "Missing, duplicate or unordered release frames")
    frames = {step: load_frame(directory, row) for step, row in archived.items()}
    observer = module.PlacementObserver()
    observer.clear_view = frames[fork]
    emitted, actions, release = [], [], None
    for step in range(fork + 5, end + 1, 5):
        native = rows[step]
        require(native["start_step"] == step - 5 and len(native["actions"]) == 5,
                "Unexpected suffix action cadence")
        actions.extend(native["actions"])
        if (step - fork) % 10:
            continue
        events = [{"step": step, **e} for e in observer.observe(frames[step], actions, step)]
        emitted.extend(events)
        if any(e["object"] == SAUCE and e["event"] == "pending_release" for e in events) and release is None:
            require(step == pending_step and observer.pending is not None and observer.pending.target == SAUCE,
                    "The prescribed first pending release did not reproduce")
            release = observer.pending
        actions = []
    prior_events = [e for e in prior_report["arms"]["candidate"]["events"] if e["step"] <= end]
    require(release is not None and project(emitted) == project(prior_events)
            and project(report["observer_events"]) == project(prior_events), "Frozen observer history changed")

    analyses, drawing_masks = {}, {}
    for step in range(start, end + 1, 5):
        frame = frames[step]
        record = archived[step]
        audit = physical[step]
        native = rows[step]
        require(audit == record["grasp_audit"] and audit["state_sha256"] == native["state_sha256"]
                == record["state_sha256"] and audit["goals"] == native["audit"]["goals"]
                and set(audit["held_by_target"]) == set(GOALS)
                and all(type(x) is bool for x in audit["held_by_target"].values()), "Physical audit mismatch")
        # Pre-pending frames are visibility context; they cannot use future release evidence.
        age = step - pending_step
        if age < 0:
            analyses[step] = {"step": step, "age_steps": age, "context_only": True,
                              "gap_m": frame.gap, "target_color_xyz": [
                                  point_stats(cloud(frame, camera)[color_mask(frame.images[camera], LABELS[SAUCE])])
                                  for camera in range(2)]}
            continue
        inspected, masks = decompose(frame, release, module)
        qualifies = bool(0 < age <= 30 and frame.gap >= .067 and inspected["center_m"] is not None)
        truth = bool(audit["goals"][SAUCE] and not audit["held_by_target"][SAUCE])
        analyses[step] = {"step": step, "age_steps": age, "context_only": False, "gap_m": frame.gap,
                          "original_10_step_sample": (step - fork) % 10 == 0,
                          "candidate_if_sampled": qualifies, "past_original_expiry": age > 30,
                          "posthoc_goal_true": audit["goals"][SAUCE],
                          "posthoc_held": audit["held_by_target"][SAUCE],
                          "posthoc_correct_released": truth, **inspected}
        drawing_masks[step] = masks
    extra = [r for r in analyses.values() if not r["context_only"] and r["candidate_if_sampled"]
             and not r["original_10_step_sample"]]
    all_hits = [r for r in analyses.values() if not r["context_only"] and r["candidate_if_sampled"]]
    first = min(all_hits, key=lambda r: r["step"], default=None)
    safe_extra = [r for r in extra if r["posthoc_correct_released"]]
    unsafe_extra = [r for r in extra if not r["posthoc_correct_released"]]
    selected = sorted({pending_step, pending_step + 5, pending_step + 10,
                       first["step"] if first else pending_step + 20})
    image = directory / "release-evidence.png"
    montage(image, frames, analyses, drawing_masks, selected)
    return {"layout_seed": prior_report["layout_seed"], "pending_step": pending_step,
            "release_origin_xy_m": release.origin.tolist(), "observer_sha256": OBSERVER_SHA,
            "original_events": emitted, "frames": list(analyses.values()),
            "first_hypothetical_5_step_commit": first,
            "extra_5_step_candidate_steps": [r["step"] for r in extra],
            "extra_5_step_safe_steps": [r["step"] for r in safe_extra],
            "extra_5_step_unsafe_steps": [r["step"] for r in unsafe_extra],
            "montage": image.name, "montage_sha256": sha(image),
            "interpretation": "Post-pending sampling opportunities only; frozen identity/release state from 10-step observations. No revised observer, policy action, or online result was tested."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--prior", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    root, prior, out = args.run.resolve(), args.prior.resolve(), args.out.resolve()
    prior_files = [p for p in (prior / "audit.json", prior / "actual-progress-audit.json")
                   if p.is_file() and sha(p) == PRIOR_SHA]
    require(not out.exists() and prior_files, "Existing output or changed prior audit")
    accepted = read(prior_files[0])
    module = load_exact("release_analysis_frozen_placement", PLACEMENT / "observer.py", OBSERVER_SHA)
    analyzed, provenance = [], []
    for card, layouts in CASES.items():
        manifest_path = root / f"release-window-{card}.json"
        manifest = read(manifest_path)
        require(manifest["complete"] is True and [r["layout_seed"] for r in manifest["records"]] == list(layouts),
                "Incomplete or changed release assignment")
        provenance.append({"card": card, "manifest_sha256": sha(manifest_path)})
        for item in manifest["records"]:
            report_path = checked(root, item["report"], item["report_sha256"])
            report = read(report_path)
            layout = item["layout_seed"]
            original, = [r for r in accepted["cases"] if r["layout_seed"] == layout]
            require(report["prior_audit_sha256"] == PRIOR_SHA and report["layout_seed"] == layout
                    and report["prior_report_sha256"] == original["report_sha256"], "Changed prior evidence link")
            previous = read(checked(prior, report["prior_report"], report["prior_report_sha256"]))
            row = analyze_case(report_path.parent, report, previous, module)
            row.update(report=item["report"], report_sha256=item["report_sha256"],
                       prior_report_sha256=report["prior_report_sha256"])
            analyzed.append(row)
            print("RELEASE_FILTER", layout, "extra_safe", row["extra_5_step_safe_steps"],
                  "extra_unsafe", row["extra_5_step_unsafe_steps"], flush=True)
    write(out, {"schema": "v2-release-filter-analysis/1", "scope": "seen_development_sensor_diagnostic",
                "analysis_source_sha256": sha(Path(__file__)), "observer_sha256": OBSERVER_SHA,
                "basket_pixels_source_sha256": sha(Path(inspect.getsourcefile(basket_pixels))),
                "prior_audit_sha256": PRIOR_SHA, "manifest_sources": provenance, "cases": analyzed,
                "online_release": False, "independent_pilot_release": False})


if __name__ == "__main__":
    main()
