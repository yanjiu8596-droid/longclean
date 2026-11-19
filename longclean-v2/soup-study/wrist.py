"""Supplement the frozen observer with an observation-only soup channel."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from probe import inks
from common import SOUP
from observer import PlacementObserver, grasp_vote, near_basket
from vision import cloud


def wrist_tracks(frame):
    xyz = cloud(frame, 1)
    delta = xyz - frame.geometry["eef"]
    near = (np.linalg.norm(delta[..., :2], axis=-1) < .055) & (-.10 < delta[..., 2]) & (delta[..., 2] < .02)
    return [xyz[near & ink] for ink in inks(frame.images[1])]


def wrist_soup_held(frame):
    blue, yellow = wrist_tracks(frame)
    return (len(blue) >= 45 and len(yellow) >= 45
            and np.linalg.norm(np.median(blue, axis=0) - np.median(yellow, axis=0)) < .075)


def wrist_pocket(frame, origin, *, detached, camera=1):
    xyz = cloud(frame, camera)
    x, y, z = np.moveaxis(xyz, -1, 0)
    region = ((-.055 < x) & (x < .065) & (.195 < y) & (y < .335)
              & (.485 < z) & (z < .561) & (np.linalg.norm(xyz[..., :2] - origin, axis=-1) < .085))
    if detached:
        d = xyz - frame.geometry["eef"]
        region &= (d[..., 2] < -.072) | (np.linalg.norm(d[..., :2], axis=-1) > .065)
    return [xyz[region & ink] for ink in inks(frame.images[camera])]


class WristSoup:
    def __init__(self):
        self.base = PlacementObserver()
        self.done = False
        self.transport = deque(maxlen=5)
        self.clear = None
        self.release = None
        self.previous = None

    def observe(self, frame, actions, step):
        events = self.base.observe(frame, actions, step)
        # Requests, instruction identity, and simulator truth are not accepted as inputs.
        combined = [e for e in events if e["event"] == "in_basket"]
        if any(e["object"] == SOUP for e in combined):
            if self.done:
                combined = [e for e in combined if e["object"] != SOUP]
            self.done = True
            self.release = None
        if not self.done:
            combined.extend(self.supplement(frame, actions, step))
        self.previous = frame
        return combined

    def supplement(self, frame, actions, step):
        output = []
        eef = frame.geometry["eef"]
        opened = any(a[-1] < -.5 for a in actions)
        if self.clear is None or (eef[1] < .155 and self.release is None):
            self.clear = frame
        if not opened and .035 <= frame.gap <= .072 and .52 < eef[2] < .82:
            visual = grasp_vote(frame)
            held = visual == SOUP or (visual is None and wrist_soup_held(frame))
            self.transport.append((step, held, eef.copy()))
        if opened and frame.gap >= .061:
            recent = [(s, p) for s, accepted, p in self.transport if accepted and step - s <= 36]
            if (self.release is None and self.previous is not None and near_basket(eef) and len(recent) >= 2
                    and step - recent[-1][0] <= 20 and np.linalg.norm(recent[-1][1] - recent[0][1]) >= .025):
                origin = self.previous.geometry["eef"][:2]
                # Remove wrist surfaces previously seen as well as stationary basket
                # contents visible in the fixed view, in the common world frame.
                old_wrist = wrist_pocket(self.clear, origin, detached=False)
                old_fixed = wrist_pocket(self.clear, origin, detached=False, camera=0)
                old = [cKDTree(np.concatenate((a, b))) for a, b in zip(old_wrist, old_fixed, strict=True)]
                self.release = (step, origin, old, None)
                output.append({"object": SOUP, "event": "wrist_pending"})
            self.transport.clear()
        elif frame.gap >= .075 or frame.gap < .03:
            self.transport.clear()
        if self.release is None:
            return output
        began, origin, old, prior = self.release
        if step - began > 35 or (not opened and frame.gap < .062):
            self.release = None
            return output + [{"object": SOUP, "event": "wrist_abstained"}]
        points = wrist_pocket(frame, origin, detached=True)
        new = [xyz[tree.query(xyz)[0] > .012] for xyz, tree in zip(points, old, strict=True)]
        counts = [len(xyz) for xyz in new]
        output.append({"object": SOUP, "event": "wrist_sample", "pixels": counts})
        if frame.gap < .067 or min(counts) < 24:
            return output
        blue, yellow = (np.median(xyz, axis=0) for xyz in new)
        if np.linalg.norm(blue - yellow) > .060:
            return output
        center = (blue + yellow) / 2
        if prior is not None and 6 <= step - prior[0] <= 20 and np.linalg.norm(center - prior[1]) < .025:
            self.done = True
            self.release = None
            output.append({"object": SOUP, "event": "in_basket", "source": "wrist_soup", "pixels": counts})
        elif prior is None or step - prior[0] > 20 or np.linalg.norm(center - prior[1]) >= .025:
            self.release = (began, origin, old, (step, center))
        return output
