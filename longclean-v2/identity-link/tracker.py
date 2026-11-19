"""Soup-only observation tracker. Identity and landing are separate evidence."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "lid-geometry")]
from common import SOUP
from lid import inspect_lids
from vision import cloud


def inks(image):
    r, g, b = np.moveaxis(np.asarray(image, dtype=np.int16)[..., :3], -1, 0)
    blue = (b > r + 18) & (b > g + 6) & (b > 52)
    yellow = (r > 70) & (g > 58) & (g > b + 35) & (r > b + 44)
    return blue, yellow


def seen_pattern(frame, camera, area, minimum):
    xyz = cloud(frame, camera)
    groups = [xyz[mask & area] for mask in inks(frame.images[camera])]
    counts = [len(group) for group in groups]
    if min(counts) < minimum:
        return None
    blue, yellow = [np.median(g, axis=0) for g in groups]
    if np.linalg.norm(blue - yellow) > .060:
        return None
    return (blue + yellow) / 2, counts, groups


def held_pattern(frame):
    hand = frame.geometry["eef"]
    for camera in (1, 0):
        d = cloud(frame, camera) - hand
        roi = np.isfinite(d).all(axis=2) & (np.linalg.norm(d[..., :2], axis=-1) < .057)
        roi &= (-.111 < d[..., 2]) & (d[..., 2] < .027)
        view = seen_pattern(frame, camera, roi, 40 if camera == 1 else 15)
        if view is not None:
            return view[0], camera, view[1]
    return None


@dataclass
class Observed:
    step: int
    eef: np.ndarray
    center: np.ndarray
    camera: int


@dataclass
class Pending:
    step: int
    origin: np.ndarray
    hand: np.ndarray
    before: tuple
    prior: dict = None


def depth_comparison(points, frame, camera):
    """Compare a current world surface with the old depth rays through it.

    With LeRobot's stored double flip, rows increase in optical y while
    columns decrease in optical x. An old surface in front of the new point
    is occlusion, not evidence that the footprint used to be empty.
    """
    intr = frame.geometry["intrinsics"][camera]
    ext = frame.geometry["extrinsics"][camera]
    depth = frame.geometry["depths"][camera]
    h, w = depth.shape
    xyz = (points - ext[:3, 3]) @ ext[:3, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > .001)]
    if not len(xyz):
        return None
    x, y, z = xyz.T
    rows = np.rint(intr[1, 1] * y / z + intr[1, 2]).astype(int)
    cols = w - 1 - np.rint(intr[0, 0] * x / z + intr[0, 2]).astype(int)
    valid = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
    old = depth[rows[valid], cols[valid]]
    diff = old - z[valid]
    diff = diff[np.isfinite(old) & (old > .001)]
    if len(diff) < max(10, len(points) // 3):
        return None
    return {
        "usable": int(len(diff)),
        "behind_ratio": float(np.mean(diff > .011)),
        "same_ratio": float(np.mean(np.abs(diff) <= .008)),
    }


def novelty(points, before):
    # Evaluate a deterministic, spatially spread subset without consulting any
    # mask or simulator metadata. Include both old camera perspectives.
    if len(points) < 22:
        return False, []
    inds = np.linspace(0, len(points) - 1, num=min(100, len(points))).astype(int)
    sample = points[inds]
    views = []
    for stamp, frame in before:
        for camera in (0, 1):
            stats = depth_comparison(sample, frame, camera)
            if stats is not None:
                views.append({"step": stamp, "camera": camera, **stats})
    visible_new = any(r["behind_ratio"] >= .58 for r in views)
    already_present = any(r["same_ratio"] >= .37 for r in views)
    return visible_new and not already_present, views


def support_in_basket(xyz, hand, origin):
    x, y, z = np.moveaxis(xyz, -1, 0)
    separated = (z < hand[2] - .070) | (np.linalg.norm(xyz[..., :2] - hand[:2], axis=-1) > .064)
    return (np.isfinite(xyz).all(axis=-1) & separated & (-.055 < x) & (x < .065)
            & (.195 < y) & (y < .335) & (.469 < z) & (z < .569)
            & (np.linalg.norm(xyz[..., :2] - origin, axis=-1) < .070))


def evidence(frame, pending):
    """Propose the single associated landing, or explicitly abstain."""
    hand = frame.geometry["eef"]
    views, candidates = [], []
    for camera in (1, 0):
        xyz = cloud(frame, camera)
        roi = support_in_basket(xyz, hand, pending.origin)
        hit = seen_pattern(frame, camera, roi, 24 if camera == 1 else 12)
        if hit is None:
            continue
        center, count, groups = hit
        # Include both colors, with an equal share so one abundant ink cannot
        # drown the evidence for the other color or mix two distant objects.
        points = np.concatenate([g[np.linspace(0, len(g)-1, min(48, len(g))).astype(int)]
                                 for g in groups])
        new, report = novelty(points, pending.before)
        views.append({"mode": "label", "camera": camera, "pixels": count, "new": new, "depth": report})
        if new and np.linalg.norm(center[:2] - pending.origin) <= .063:
            candidates.append(("identity_label", center))
    xyz = cloud(frame, 1)
    for lid in inspect_lids(frame):
        center = np.array(lid["center_m"])
        x, y, z = center
        r = lid["radius_m"]
        rim_clearance = min(x + .055, .065 - x, y - .195, .335 - y)
        if (not .470 < z < .575 or rim_clearance < .55 * r or hand[2] - z < .065
                or np.linalg.norm(center[:2] - pending.origin) > .064):
            continue
        delta = xyz - center
        support = ((np.linalg.norm(delta[..., :2], axis=-1) <= .70 * r)
                   & (np.abs(delta @ lid["normal_xyz"]) < .0035))
        new, report = novelty(xyz[support], pending.before)
        views.append({"mode": "lid", "radius_m": r, "new": new, "depth": report})
        if new:
            candidates.append(("identity_lid", center))
    if not candidates:
        return None, views, "no_novel_associated_surface"
    chosen = candidates[0]
    if any(np.linalg.norm(center[:2] - chosen[1][:2]) > .038 or abs(center[2] - chosen[1][2]) > .055
           for _, center in candidates[1:]):
        return None, views, "ambiguous_landing"
    return chosen, views, None


class IdentityObserver:
    """Confirm soup only; no instruction identity, simulated state or truth inputs."""

    def __init__(self):
        self.complete = set()
        self.transport = deque(maxlen=16)
        self.clear = deque(maxlen=4)
        self.pending = None
        self.grasp_gap = None
        self.last_step = None

    def observe(self, frame, actions, step):
        if SOUP in self.complete:
            return []
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("Frames must be in strictly increasing control-step order")
        self.last_step = step
        hand = frame.geometry["eef"]
        opening = any(a[-1] < -.5 for a in actions)
        events = []

        def record(name, **info):
            events.append({"object": SOUP, "event": name, "step": step, **info})

        if self.pending is None and hand[1] < .160 and (not self.clear or step >= self.clear[-1][0] + 6):
            self.clear.append((step, frame))

        if self.pending is None and not opening and .035 <= frame.gap <= .073 and .50 < hand[2] < .84:
            self.grasp_gap = frame.gap
            hit = held_pattern(frame)
            if hit is not None:
                center, camera, pixels = hit
                self.transport.append(Observed(step, hand.copy(), center, camera))
                record("identity_sample", camera=camera, pixels=pixels, xyz_m=center.tolist())

        if self.pending is None and opening and frame.gap >= .067:
            latest = self.transport[-1] if self.transport else None
            if latest is not None:
                # The first useful observation must come from a moving hand
                # before entering the basket, not hand pixels over old cargo.
                links = [p for p in self.transport if p.step < latest.step and step - p.step <= 92
                         and p.eef[1] < .177
                         and np.linalg.norm(latest.eef - p.eef) >= .031
                         and np.linalg.norm(latest.center - p.center) >= .025
                         and np.linalg.norm((latest.center - p.center) - (latest.eef - p.eef)) <= .049]
                # Use sensed cargo-to-hand offset to extrapolate the last
                # observed center to the opening, rather than commanded label.
                origin = hand[:2] + latest.center[:2] - latest.eef[:2]
                near_basket = -.105 < origin[0] < .112 and .170 < origin[1] < .358
                fresh = step - latest.step <= 25 and np.linalg.norm(hand - latest.eef) <= .115
                changed = self.grasp_gap is not None and frame.gap >= self.grasp_gap + .005
                before = tuple((s, f) for s, f in self.clear if 5 <= step - s <= 135)
                if fresh and links and near_basket and changed and before:
                    self.pending = Pending(step, origin, hand.copy(), before)
                    record("identity_pending", transport_steps=[links[0].step, latest.step],
                           origin_xy_m=origin.tolist(), baseline_steps=[s for s, _ in before])
                else:
                    record("identity_release_rejected", fresh=bool(fresh), transport_link=bool(links),
                           near_basket=bool(near_basket), opened=bool(changed), baseline=bool(before))
            self.transport.clear()
            self.grasp_gap = None
        elif self.pending is None and (frame.gap >= .077 or frame.gap < .029):
            self.transport.clear()
            self.grasp_gap = None

        if self.pending is None:
            return events
        pending = self.pending
        age = step - pending.step
        if age > 40 or (not opening and frame.gap < .061):
            self.pending = None
            record("identity_abstained", reason="timeout" if age > 40 else "gripper_reclosed")
            return events
        if frame.gap < .067 or age < 3:
            return events
        candidate, views, failure = evidence(frame, pending)
        record("identity_landing_sample", release_step=pending.step,
               candidate=None if candidate is None else candidate[0], views=views, reason=failure)
        if candidate is None:
            if failure == "ambiguous_landing":
                pending.prior = None
            return events
        source, center = candidate
        prior = pending.prior
        if (prior is not None and source == prior["source"] and 5 <= step - prior["step"] <= 20
                and np.linalg.norm(center - prior["center"]) <= .025):
            self.complete.add(SOUP)
            self.pending = None
            record("in_basket", source=source, release_step=pending.step,
                   evidence_step=prior["step"], xyz_m=center.tolist())
        elif (prior is None or step - prior["step"] > 20 or source != prior["source"]
              or np.linalg.norm(center - prior["center"]) > .025):
            pending.prior = {"step": step, "source": source, "center": center}
        return events
