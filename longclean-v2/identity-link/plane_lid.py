"""Sensor-only lid fallback: isolate coplanar patches before the frozen circle fit."""

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lid-geometry"), str(ROOT)]
from lid import _proposal, inspect_lids as inspect_original_lids
from vision import cloud


_CONNECT = np.ones((3, 3), dtype=bool)
_CROSS_5 = np.zeros((5, 5), dtype=bool)
_CROSS_5[2, 2] = _CROSS_5[0, 2] = _CROSS_5[4, 2] = True
_CROSS_5[2, 0] = _CROSS_5[2, 4] = True
_PLANE_TOL = .0018


def _uniform_rows(rows, limit):
    return rows[np.linspace(0, len(rows) - 1, min(len(rows), limit), dtype=int)]


def _planes(points, mask):
    """Select a bounded set of planes using local measured normals and z bins."""
    xyz = points[mask]
    sample = _uniform_rows(xyz, 3600)
    ref = np.median(sample, axis=0)
    normals, levels = [], []
    # Get normals from short, entirely observed horizontal and vertical spans.
    core = np.argwhere(ndimage.binary_erosion(mask, structure=_CROSS_5))
    if len(core):
        r, c = _uniform_rows(core, 56).T
        horizontal = points[r, c + 2] - points[r, c - 2]
        vertical = points[r + 2, c] - points[r - 2, c]
        cross = np.cross(horizontal, vertical)
        length = np.linalg.norm(cross, axis=1)
        keep = (length > 1e-9) & (np.abs(cross[:, 2]) > .90 * length)
        if np.any(keep):
            normal = cross[keep] / length[keep, None]
            normal *= np.sign(normal[:, 2:3])
            normals.extend(normal)
            levels.extend(np.sum((points[r[keep], c[keep]] - ref) * normal, axis=1))

    # Nearly horizontal tops are common. Dense height modes back up a noisy
    # finite difference without making all samples vote for nearby clutter.
    bins = np.rint((sample[:, 2] - ref[2]) / .0012).astype(int)
    distinct, counts = np.unique(bins, return_counts=True)
    for i in np.argsort(-counts, kind="stable")[:8]:
        if counts[i] < 20:
            break
        normals.append(np.array([0., 0., 1.]))
        levels.append(float(np.median(sample[bins == distinct[i], 2]) - ref[2]))
    if not normals:
        return []

    nrm = np.asarray(normals)
    lvl = np.asarray(levels)
    signatures = np.rint(np.column_stack((nrm[:, :2] / .035, lvl / .0013))).astype(int)
    _, first = np.unique(signatures, axis=0, return_index=True)
    first.sort()
    nrm, lvl = nrm[first], lvl[first]
    support = np.sum(np.abs((sample - ref) @ nrm.T - lvl) <= _PLANE_TOL, axis=0)
    chosen = []
    for j in np.argsort(-support, kind="stable"):
        # Reject hypotheses that are practically the same over a lid-sized patch.
        if support[j] < 60:
            break
        if any(abs(lvl[j] - prev_lvl) + .033 * np.linalg.norm(nrm[j] - prev_nrm) < .0020
               for prev_nrm, prev_lvl in chosen):
            continue
        chosen.append((nrm[j], lvl[j]))
        if len(chosen) == 4:
            break
    return [(normal, float(normal @ ref + level)) for normal, level in chosen]


def _windows(mask, minimum=100, limit=4):
    labels, total = ndimage.label(mask, structure=_CONNECT)
    counts = np.bincount(labels.ravel(), minlength=total + 1)
    windows = ndimage.find_objects(labels)
    ranked = sorted(range(1, total + 1), key=lambda k: (-counts[k], k))
    emitted = 0
    for label in ranked:
        if counts[label] < minimum or emitted == limit:
            return
        window = windows[label - 1]
        if min(axis.stop - axis.start for axis in window) < 9:
            continue
        emitted += 1
        yield window, (labels[window] == label)


def local_plane_lids(points, rgb, hand_z):
    """Propose alternatives near the basket from only measured wrist RGB-D."""
    x, y, z = np.moveaxis(points, -1, 0)
    # Restrict the fallback search to the landing workspace, slightly outside
    # downstream center limits so a complete perimeter can still be observed.
    area = (np.isfinite(points).all(axis=2) & (np.abs(x) <= .120)
            & (.145 <= y) & (y <= .386) & (.458 <= z) & (z <= .608)
            & (z <= hand_z - .045))
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    alternatives = []
    for ceiling, spread in ((94, 19), (123, 27)):
        dark = area & (high <= ceiling) & (low >= 12) & (high - low <= spread)
        for region, component in _windows(dark):
            xyz = points[region]
            offset = tuple(axis.start for axis in region)
            for nrm, level in _planes(xyz, component):
                coplanar = np.abs(np.sum(xyz * nrm, axis=2) - level) <= _PLANE_TOL
                for patch, linked in _windows(component & coplanar, limit=2):
                    patch_points = xyz[patch]
                    filled = ndimage.binary_fill_holes(linked)
                    if filled.sum() <= 1.12 * linked.sum():
                        linked = filled & coplanar[patch] & area[region][patch]
                    image_offset = tuple(a + b.start for a, b in zip(offset, patch))
                    fit = _proposal(patch_points, linked, image_offset)
                    if fit is not None:
                        alternatives.append({**fit, "proposal_method": "local_plane"})
    return sorted(alternatives, key=lambda d: (-d["confidence"], -d["surface_pixels"]))


def inspect_lids(frame):
    """Keep all original proposals, adding only distinct, unchanged-fit circles."""
    proposals = list(inspect_original_lids(frame))
    points = np.asarray(cloud(frame, 1), dtype=np.float64)
    rgb = np.asarray(frame.images[1], dtype=np.int16)[..., :3]
    extras = 0
    for fit in local_plane_lids(points, rgb, frame.geometry["eef"][2]):
        if any(np.linalg.norm(np.subtract(fit["center_m"], old["center_m"])) < .014 for old in proposals):
            continue
        proposals.append(fit)
        extras += 1
        if extras == 4:
            break
    return proposals
