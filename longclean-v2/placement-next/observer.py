"""Passive RGB-D placement observer for the two remaining scene-01 objects."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from common import LABELS, SAUCE, SOUP
from vision import basket_pixels, cloud, color_mask

TARGETS = (SAUCE, SOUP)


@dataclass
class Release:
    target: str
    step: int
    origin: np.ndarray
    baseline: object
    old_points: object


def near_basket(eef):
    x, y, z = eef
    return bool(-.09 < x < .10 and .17 < y < .36 and .51 < z < .82)


def grasp_vote(frame):
    """Look only around the fingers; require a winner among the unambiguous inks."""
    counts = {t: [] for t in TARGETS}
    for i, image in enumerate(frame.images):
        offset = cloud(frame, i) - frame.geometry["eef"]
        roi = (np.linalg.norm(offset[..., :2], axis=-1) < .052) & (-.10 < offset[..., 2]) & (offset[..., 2] < -.008)
        for target in TARGETS:
            counts[target].append(int(np.count_nonzero(roi & color_mask(image, LABELS[target]))))
    scores = {t: max(counts[t]) for t in TARGETS}
    winner = max(scores, key=scores.get)
    other = next(t for t in TARGETS if t != winner)
    minimum = 12 if winner == SAUCE else 22
    return winner if scores[winner] >= minimum and scores[winner] > 1.5 * scores[other] else None


def colored_pocket(frame, camera, target, origin):
    return basket_pixels(frame, camera, origin) & color_mask(frame.images[camera], LABELS[target])


def make_release(target, step, origin, baseline):
    label_points = []
    for i in range(2):
        label_points.append(cloud(baseline, i)[colored_pocket(baseline, i, target, origin)])
    return Release(target, step, origin, baseline, cKDTree(np.concatenate(label_points)))


def deposit(frame, release):
    """Require new target pixels inside the pocket and separated from the fingers."""
    counts, centers = [], []
    for i in range(2):
        xyz = cloud(frame, i)
        mask = colored_pocket(frame, i, release.target, release.origin)
        offset = xyz - frame.geometry["eef"]
        mask &= (offset[..., 2] < -.073) | (np.linalg.norm(offset[..., :2], axis=-1) > .065)
        if i == 0:
            mask &= release.baseline.geometry["depths"][0] - frame.geometry["depths"][0] > .009
        points = xyz[mask]
        if i == 1 and len(points):
            points = points[release.old_points.query(points)[0] > .012]
        counts.append(len(points))
        centers.append(np.median(points, axis=0) if len(points) else None)

    # Soup ink partly overlaps the cheese carton. Require the fixed depth change,
    # with a lower pixel count only if the wrist independently sees the same area.
    if (release.target == SOUP and counts[0] >= 18 and counts[1] >= 60
            and np.linalg.norm(centers[0] - centers[1]) < .055):
        return counts, centers[0]
    # Sauce's green ink is distinct; allow the wrist when the fixed view is occluded.
    qualifies = (counts[0] >= 34, counts[1] >= 45 and release.target == SAUCE)
    channel = next((i for i, good in enumerate(qualifies) if good), None)
    return counts, None if channel is None else centers[channel]


class PlacementObserver:
    """Apply observations only; the instruction and simulator audit are not inputs."""
    def __init__(self):
        self.complete = set()
        self.pending = None
        self.votes = deque(maxlen=4)
        self.clear_view = None
        self.previous = None

    def observe(self, frame, actions, step):
        eef = frame.geometry["eef"]
        opening = any(a[-1] < -.5 for a in actions)
        over_basket = near_basket(eef)
        result = []
        if self.clear_view is None or (eef[1] < .155 and self.pending is None):
            self.clear_view = frame

        if not opening and .035 <= frame.gap <= .071 and .52 < eef[2] < .82:
            self.votes.append((step, grasp_vote(frame), eef.copy()))
        if opening and frame.gap >= .061:
            if self.pending is None and over_basket and self.previous is not None:
                for target in TARGETS:
                    seen = [(s, pos) for s, t, pos in self.votes if t == target and s >= step - 35]
                    if (target not in self.complete and len(seen) >= 2
                            and step - seen[-1][0] <= 20
                            and np.linalg.norm(seen[-1][1] - seen[0][1]) >= .025
                            and not any(t is not None and t != target and s > seen[0][0]
                                        for s, t, _ in self.votes)):
                        self.pending = make_release(target, step, self.previous.geometry["eef"][:2], self.clear_view)
                        result.append({"object": target, "event": "pending_release"})
                        break
            self.votes.clear()
        elif frame.gap >= .075 or frame.gap < .03:
            self.votes.clear()

        if self.pending is not None:
            event = self.pending
            counts, center = deposit(frame, event)
            age = step - event.step
            # Wait for a later observation. A confirming frame must have depth and identity
            # evidence of its own; repeated open commands alone cannot confirm a deposit.
            if age > 0 and center is not None and frame.gap >= .067:
                self.complete.add(event.target)
                result.append({"object": event.target, "event": "in_basket",
                               "release_step": event.step, "pixels": counts})
                self.pending = None
            elif age >= 30:
                result.append({"object": event.target, "event": "abstained", "pixels": counts})
                self.pending = None
            else:
                result.append({"object": event.target, "event": "post_release_sample", "pixels": counts})
        self.previous = frame
        return result
