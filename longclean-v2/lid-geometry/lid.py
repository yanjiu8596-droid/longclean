"""Sensor-only proposals for dark, near-horizontal circular lids in a wrist image.

A proposal describes a surface, not the identity of its object or whether that
object was placed. The caller supplies the released-object track and basket
geometry separately. Distances are expressed in meters unless marked otherwise.
"""
import numpy as np
from scipy import ndimage


def _circle(boundary):
    """Fit a circle to a 2D perimeter, trimming detached edge pixels."""
    initial = boundary.mean(axis=0)
    xy = boundary - initial
    active = np.ones(len(xy), dtype=bool)
    for _ in range(4):
        v = xy[active]
        fit, _, rank, _ = np.linalg.lstsq(
            np.column_stack((2 * v, np.ones(len(v)))),
            (v * v).sum(axis=1), rcond=None,
        )
        if rank != 3:
            return None
        center = fit[:2]
        r2 = fit[2] + center @ center
        if r2 <= 0:
            return None
        radius = np.sqrt(r2)
        radii = np.linalg.norm(xy - center, axis=1)
        error = np.abs(radii - radius)
        active = error <= max(.0016, min(.0035, 2.5 * np.median(error)))
        if active.sum() < .65 * len(xy):
            return None
    angles = np.sort(np.arctan2(*(xy[active] - center).T[::-1]))
    coverage = 2 * np.pi - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
    return center + initial, radius, error, active, coverage


def _proposal(point_image, mask, origin):
    points = point_image[mask]
    if not 100 <= len(points) <= 6500:
        return None
    mean = points.mean(axis=0)
    _, sigma, basis = np.linalg.svd(points - mean, full_matrices=False)
    if sigma[0] == 0 or sigma[1] / sigma[0] < .61 or abs(basis[2, 2]) < .90:
        return None
    plane_q90 = np.quantile(np.abs((points - mean) @ basis[2]), .90)
    if plane_q90 > .0022:
        return None

    border = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    edge_pts = point_image[border]
    if len(edge_pts) < 22:
        return None
    fit = _circle((edge_pts - mean) @ basis[:2].T)
    if fit is None:
        return None
    center_2d, radius, error, inliers, coverage = fit
    radial_q90 = np.quantile(error, .90)
    if not (.020 <= radius <= .039 and radial_q90 <= .0036 and coverage >= 4.8):
        return None
    center = mean + center_2d @ basis[:2]
    region_idxs = np.argwhere(mask)
    pix = region_idxs[np.argmin(np.linalg.norm(points - center, axis=1))]
    confidence = float(
        .35 * inliers.mean()
        + .20 * (coverage / (2 * np.pi))
        + .25 * (1 - radial_q90 / .0036)
        + .20 * (1 - plane_q90 / .0022)
    )
    normal = basis[2] * np.sign(basis[2, 2])
    return {
        "center_m": center.tolist(),
        "center_px": (pix + np.asarray(origin)).tolist(),  # [row, col] in Frame.images[1]
        "radius_m": float(radius),
        "normal_xyz": normal.tolist(),
        "confidence": confidence,
        "surface_pixels": int(len(points)),
        "boundary_pixels": int(len(edge_pts)),
        "boundary_inlier_fraction": float(inliers.mean()),
        "radial_q90_mm": float(1000 * radial_q90),
        "plane_q90_mm": float(1000 * plane_q90),
        "angular_coverage_deg": float(np.degrees(coverage)),
    }


def inspect_lids(frame):
    """Return ranked geometric proposals in the post-release tabletop workspace.

    Inputs are the calibrated wrist RGB/depth and sensed hand position only.
    The workspace extends in front of and beside the basket so an outside lid
    is observable without being considered a confirmed placement.
    """
    from vision import cloud

    points = np.asarray(cloud(frame, 1), dtype=np.float64)
    rgb = np.asarray(frame.images[1], dtype=np.int16)[..., :3]
    x, y, z = points.transpose(2, 0, 1)
    workspace = (
        np.isfinite(points).all(axis=2)
        & (np.abs(x) <= .24) & (.03 <= y) & (y <= .43)
        & (.448 <= z) & (z <= .72) & (z <= frame.geometry["eef"][2] - .045)
    )
    # Suppress steep depth boundaries before filling small holes in the dark
    # mask. This keeps a lid separate from shadows on the basket and tabletop.
    finite_z = np.where(np.isfinite(z), z, np.inf)
    steep = ndimage.maximum_filter(finite_z, size=3) - ndimage.minimum_filter(finite_z, size=3)
    locally_flat = workspace & (steep <= .0050)
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    proposals = []
    for ceiling, color_spread in ((94, 19), (123, 27)):
        dark_neutral = (high <= ceiling) & (low >= 12) & (high - low <= color_spread)
        seeds = dark_neutral & locally_flat
        labels, _ = ndimage.label(seeds, structure=np.ones((3, 3)))
        for label, window in enumerate(ndimage.find_objects(labels), 1):
            nr, nc = (axis.stop - axis.start for axis in window)
            if min(nr, nc) < 9 or nr * nc < 100:
                continue
            mask = labels[window] == label
            filled = ndimage.binary_fill_holes(mask)
            if filled.sum() <= 1.12 * mask.sum():
                mask = filled & locally_flat[window]
            candidate = _proposal(points[window], mask, tuple(axis.start for axis in window))
            if candidate is not None:
                proposals.append(candidate)

    ranked = sorted(proposals, key=lambda item: (-item["confidence"], -item["surface_pixels"]))
    distinct = []
    for candidate in ranked:
        center = np.asarray(candidate["center_m"])
        if not any(np.linalg.norm(center - item["center_m"]) < .014 for item in distinct):
            distinct.append(candidate)
    return distinct
