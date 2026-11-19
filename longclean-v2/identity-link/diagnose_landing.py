"""Post-hoc CPU diagnosis of the frozen natural trajectories; no policy calls."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval import trace_rows
from common import SAUCE, SOUP, array_hash, read, sha, write
import lid
import tracker
from vision import Frame


def mm(value):
    return round(1000 * float(value), 2)


def body_ray_proxy(frame, camera, center, radius):
    """Intersect depth rays with loose spheres about audited body positions.

    A ray with a measured surface in front of the sphere cannot see into
    the volume. A measured point inside the sphere need not belong to soup.
    Body orientation/shape and pixel-level instance labels were not archived.
    """
    intr = frame.geometry["intrinsics"][camera]
    ext = frame.geometry["extrinsics"][camera]
    depth = frame.geometry["depths"][camera]
    x, y, z = (center - ext[:3, 3]) @ ext[:3, :3]
    row, col = np.ogrid[:depth.shape[0], :depth.shape[1]]
    u = (depth.shape[1] - 1 - col - intr[0, 2]) / intr[0, 0]
    v = (row - intr[1, 2]) / intr[1, 1]
    a, b = u * u + v * v + 1, u * x + v * y + z
    disc = b * b - a * (x * x + y * y + z * z - radius * radius)
    half = np.sqrt(np.maximum(disc, 0))
    near, far = (b - half) / a, (b + half) / a
    in_view = (disc >= 0) & (far > .001)
    valid = in_view & np.isfinite(depth) & (depth > .001)
    nearer = valid & (depth < near - .003)
    beyond = valid & (depth > far + .003)
    through = valid & ~nearer & ~beyond
    return [int(mask.sum()) for mask in (in_view, valid, nearer, through, beyond)]


def body_proximity(center, positions):
    """Approximate location match for diagnostics; competing can stays explicit."""
    xy = [float(np.linalg.norm((center - pos)[:2])) for pos in positions]
    dz = [abs(float(center[2] - pos[2])) for pos in positions]
    nearby = [i for i in (0, 1) if xy[i] <= .055 and dz[i] <= .105]
    if len(nearby) == 2:
        nearby.sort(key=lambda i: xy[i])
        match = "ambiguous" if abs(xy[0] - xy[1]) < .020 else ("soup", "other_can")[nearby[0]]
    else:
        match = ("soup", "other_can")[nearby[0]] if nearby else "neither"
    return match, [mm(x) for x in xy], [mm(x) for x in dz]


def lid_filters(proposal, hand, origin, xyz, before):
    center = np.asarray(proposal["center_m"])
    x, y, z = center
    r = proposal["radius_m"]
    rim = min(x + .055, .065 - x, y - .195, .335 - y)
    clearance = hand[2] - z
    offset = np.linalg.norm(center[:2] - origin)
    failed = [name for name, condition in (
        ("height", not .470 < z < .575),
        ("basket_rim", rim < .55 * r),
        ("gripper_clearance", clearance < .065),
        ("release_offset", offset > .064),
    ) if condition]
    offsets = list(map(mm, (z, rim, clearance, offset)))
    if failed:
        return failed, offsets, None
    delta = xyz - center
    support = ((np.linalg.norm(delta[..., :2], axis=-1) <= .70 * r)
               & (np.abs(delta @ proposal["normal_xyz"]) < .0035))
    new, views = tracker.novelty(xyz[support], before)
    if new:
        novelty = "new_surface"
    elif int(support.sum()) < 22:
        novelty = "too_few_surface_rays"
    elif not views:
        novelty = "no_reference_ray_comparison"
    elif any(view["same_ratio"] >= .37 for view in views):
        novelty = "similar_depth_in_reference"
    else:
        novelty = "no_visible_depth_change"
    return failed, offsets, novelty


def fit_stage(points, mask):
    """Name the first failed test in the frozen circle fitter."""
    xyz = points[mask]
    if not 100 <= len(xyz) <= 6500:
        return "area"
    mean = xyz.mean(axis=0)
    _, sig, basis = np.linalg.svd(xyz - mean, full_matrices=False)
    if sig[0] == 0 or sig[1] / sig[0] < .61:
        return "elongated"
    if abs(basis[2, 2]) < .90:
        return "tilt"
    if np.quantile(np.abs((xyz - mean) @ basis[2]), .90) > .0022:
        return "nonplanar"
    boundary = mask & ~lid.ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    edge = points[boundary]
    if len(edge) < 22:
        return "edge_size"
    fit = lid._circle((edge - mean) @ basis[:2].T)
    if fit is None:
        return "circle_inliers"
    _, radius, residual, _, angular_coverage = fit
    violations = [name for name, condition in (
        ("radius", not .020 <= radius <= .039),
        ("radial_residual", np.quantile(residual, .90) > .0036),
        ("arc_coverage", angular_coverage < 4.8),
    ) if condition]
    return "+".join(violations) if violations else "fit"


def profile(frame, positions, pending, frozen_sample):
    components, proposals = [], []
    actual_fit = lid._proposal

    def record_component(points, mask, origin):
        result = actual_fit(points, mask, origin)
        if mask.any():
            center = np.median(points[mask], axis=0)
            match, _, _ = body_proximity(center, positions)
            stage = fit_stage(points, mask) if match == "soup" else None
            if stage is not None and (stage == "fit") != (result is not None):
                raise RuntimeError(f"Fitter diagnostics disagree with frozen result at {frame.source}")
            components.append((match, bool(result), int(mask.sum()), stage))
        return result

    def record_lids(f):
        inspected = lid.inspect_lids(f)
        proposals.extend(inspected)
        return inspected

    with patch.object(lid, "_proposal", record_component):
        if pending is None:
            record_lids(frame)
            chosen, failure = None, "not_sampled"
        else:
            with patch.object(tracker, "inspect_lids", record_lids):
                match, _, failure = tracker.evidence(frame, pending)
            chosen = match[0] if match is not None else None
            if (chosen, failure) != (frozen_sample["candidate"], frozen_sample["reason"]):
                raise RuntimeError(f"Candidate replay diverges at {frame.source}")

    fits = []
    hand = frame.geometry["eef"]
    xyz = tracker.cloud(frame, 1)
    for item in proposals:
        match, xy, dz = body_proximity(np.asarray(item["center_m"]), positions)
        row = {"match": match, "xy_mm": xy, "dz_mm": dz, "pixels": item["surface_pixels"]}
        if pending is not None:
            bad, dist, novelty = lid_filters(item, hand, pending.origin, xyz, pending.before)
            row.update(rejected=bad, dist_mm=dist, novelty=novelty)
        fits.append(row)
    soup_fits = [fit for fit in fits if fit["match"] == "soup"]
    wrist = [body_ray_proxy(frame, 1, positions[0], size) for size in (.045, .065)]
    fixed = [body_ray_proxy(frame, 0, positions[0], size) for size in (.045, .065)]
    if chosen:
        stage = "candidate"
    elif failure == "ambiguous_landing":
        stage = "ambiguous_landing"
    elif soup_fits:
        if pending is None:
            stage = "fit_while_not_sampled"
        else:
            stage = "novelty_rejected" if any(not fit["rejected"] for fit in soup_fits) else "association_rejected"
    elif any(fit["match"] == "ambiguous" for fit in fits):
        stage = "approximate_can_match_ambiguous"
    elif any(c[0] == "soup" for c in components):
        stage = "near_soup_component_no_fit"
    elif wrist[1][0] == 0:
        stage = "wrist_proxy_out_of_view"
    elif wrist[1][3] < 100:
        stage = "sparse_wrist_depth_support_proxy"
    else:
        stage = "no_near_soup_dark_flat_component"
    return {"stage": stage, "chosen": chosen, "wrist_ray_counts": wrist, "fixed_ray_counts": fixed,
            "components": [list(c) for c in components], "fits": fits}


def inspect_case(result, collection):
    seed, positive = result["seed"], result["positive_step"]
    if positive is None:
        return None
    score = result["scores"]["identity"]
    events = score["trace"]
    pending = {e["step"]: e for e in events if e["event"] == "identity_pending"}
    frozen = {e["step"]: e for e in events if e["event"] == "identity_landing_sample"
              and positive - 5 <= e["step"] <= positive + 40}
    raw_rows = {step: (path, recorded) for step, path, _, recorded in trace_rows(collection / str(seed))}
    cache = {}

    def verified_frame(step):
        if step not in cache:
            path, recorded = raw_rows[step]
            hashes = recorded["observation_sha256"]
            with np.load(path, allow_pickle=False) as raw:
                if not hashes or any(array_hash(raw[k]) != v for k, v in hashes.items()):
                    raise ValueError(f"Observation hash mismatch: {path}")
            cache[step] = Frame.read(path)
        return cache[step]

    rows = []
    for step in sorted(set(frozen) | {s for s in raw_rows if positive <= s <= positive + 40}):
        record = raw_rows[step][1]["audit"]
        positions = np.asarray([record["positions"][obj] for obj in (SOUP, SAUCE)])
        active = None
        if step in frozen:
            start = frozen[step]["release_step"]
            event = pending[start]
            active = tracker.Pending(start, np.asarray(event["origin_xy_m"]),
                                     verified_frame(start).geometry["eef"],
                                     tuple((t, verified_frame(t)) for t in event["baseline_steps"]))
        evaluated = profile(verified_frame(step), positions, active, frozen.get(step))
        rows.append({"step": step, "sampled": active is not None, "soup_in": record["goals"][SOUP],
                     "held": record["held_soup"], **evaluated})
    samples = [row for row in rows if row["sampled"]]
    selected = [row["step"] for row in samples if row["chosen"]]
    nearby = [e for e in events if e["event"] == "identity_release_rejected"
              and positive - 15 <= e["step"] <= positive + 15]
    return {"layout": seed, "positive_step": positive, "identity_hit": score["hit"],
            "samples": len(samples), "selected_steps": selected, "frames_verified": len(cache),
            "release_rejections": nearby, "sample_stages": dict(Counter(d["stage"] for d in samples)),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("independent", type=Path, help="Full frozen independent.json, including identity trace")
    parser.add_argument("out", type=Path)
    parser.add_argument("--expect-report-sha", help="SHA256 of the original independent.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    input_sha = sha(args.independent)
    if args.expect_report_sha and args.expect_report_sha != input_sha:
        raise ValueError("Independent report digest differs from frozen record")
    original = read(args.independent)
    freeze = read(Path(__file__).with_name("freeze.json"))
    if original["source"] != freeze["source"] or [
        r["seed"] for r in original["results"]
    ] != freeze["layouts"] or any(
        sha(Path(__file__).with_name(k)) != digest for k, digest in freeze["source"].items()
    ):
        raise ValueError("Frozen source/layout mismatch")
    all_cases = []
    for case in original["results"]:
        profile_case = inspect_case(case, args.collection)
        if profile_case:
            all_cases.append(profile_case)
    misses = [c for c in all_cases if c["identity_hit"] is None]
    selected = [r for c in misses for r in c["rows"] if r["sampled"]]
    summary = {
        "positives": len(all_cases),
        "missed": len(misses),
        "missed_with_no_evidence_samples": sum(not c["samples"] for c in misses),
        "missed_with_one_selected_sample": sum(len(c["selected_steps"]) == 1 for c in misses),
        "missed_sample_stages": dict(Counter(r["stage"] for r in selected)),
        "missed_case_stages": dict(Counter(s for c in misses for s in c["sample_stages"])),
        "missed_component_fit_rejections": dict(Counter(
            stage for r in selected for match, found, _, stage in r["components"]
            if match == "soup" and not found)),
        "missed_exact_lid_filter_counts": dict(Counter(k for r in selected for f in r["fits"]
                                                     if f["match"] == "soup" for k in f["rejected"])),
        "missed_near_soup_lid_novelty": dict(Counter(f["novelty"] for r in selected for f in r["fits"]
                                                  if f["match"] == "soup" and f["novelty"])),
        "missed_soup_fit_not_sampled_cases": sum(any(r["stage"] == "fit_while_not_sampled"
                                                    for r in c["rows"]) for c in misses),
    }
    write(args.out, {"source_sha256": sha(Path(__file__)), "input_sha256": input_sha,
                     "notes": {"ray_counts": "For each camera and radius 45/65mm: rays, valid depth, foreground, inside body sphere, background. Inside is geometric support proxy, not an instance mask.",
                               "fit_match": "Nearby audited soup or sauce body by XY and broad height bound; matched/ambiguous cannot establish identity.",
                               "fit_distances": "Height, rim margin, gripper clearance, release offset in mm.",
                               "stages": "Only a fitted proposal can fail a measured tracker spatial gate. Lack of a dark flat component despite depth support need not be an image fail: the sphere can include neighboring surfaces.",
                               "cadence": "Only archived frames; no interpolation claims about 1–4 unseen steps."},
                     "summary": summary, "results": all_cases})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
