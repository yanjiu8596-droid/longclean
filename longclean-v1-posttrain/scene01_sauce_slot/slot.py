"""Find an observed floor-level slot, then align and lower the held can."""
import numpy as np

import geometry
from low_place import clouds, fit_rim, clear_depth


def grid(rim, basis, occupant, params):
    """Positive margins denote separation of the payload disk from both obstacles."""
    step = params["grid_m"]
    radius = params["payload_radius_m"]
    need = radius + params["occupant_radius_m"]
    half, margin = params["inner_half_width_m"], params["margin_m"]
    axis = np.arange(-np.ceil(half * 2 / step), np.ceil(half * 2 / step) + 1) * step
    coords = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    uv = coords @ basis
    wall = half - np.abs(uv).max(axis=1) - radius
    can = np.linalg.norm(coords + rim[:2] - occupant[:2], axis=1) - need
    clearance = np.minimum(wall, can)
    band = np.abs(coords[:, 1]) <= params["centerline_y_half_width_m"]
    output = {}
    for label, allowed in (("centerline", band), ("full_aperture", np.ones(len(band), bool))):
        best = int(np.argmax(np.where(allowed, clearance, -np.inf)))
        wall_ok = allowed & (wall >= margin)
        output[label] = {"grid_candidates_inside_walls": int(wall_ok.sum()),
                         "feasible": int((wall_ok & (can >= margin)).sum()),
                         "best_center_xy": (rim[:2] + coords[best]).tolist(),
                         "wall_clearance_mm": float(1000 * wall[best]), "can_clearance_mm": float(1000 * can[best]),
                         "max_min_clearance_mm": float(1000 * clearance[best])}
    # A continuous upper bound independent of the grid spacing and of the goal band.
    edge = half - radius - margin
    corners = np.array([(x, y) for x in (-edge, edge) for y in (-edge, edge)]) @ basis.T + rim[:2]
    output["continuous_max_center_distance_mm"] = float(1000 * np.max(np.linalg.norm(corners - occupant[:2], axis=1)))
    output["required_center_distance_mm"] = 1000 * (need + margin)
    legal = coords[band & (wall >= margin)] + rim[:2]
    for idx in np.argsort(np.linalg.norm(legal - occupant[:2], axis=1), kind="stable"):
        distances = np.linalg.norm(legal - legal[idx], axis=1)
        far = int(np.argmax(distances))
        if distances[far] >= need + margin:
            output["two_target_layout_hint"] = {"soup_xy": legal[idx].tolist(), "sauce_xy": legal[far].tolist(),
                "first_can_displacement_mm": float(1000 * np.linalg.norm(legal[idx] - occupant[:2]))}
            break
    return output


def poses(frame, rim, views, previous):
    state = {k: frame[k] for k in ("site_position", "site_rotation", "gripper_qpos")}
    blue = []
    for xyz, rgb, mask in views:
        r, g, b = np.moveaxis(rgb, -1, 0)
        inside = (np.linalg.norm(xyz[..., :2] - rim[:2], axis=-1) < .088)
        below = (xyz[..., 2] < rim[2] - .018) & (xyz[..., 2] > rim[2] - .14)
        blue.append(xyz[mask & inside & below & (b > r + 18) & (b > g + 6) & (b > 52)])
    seed = rim - np.array([0., 0., .085])
    if sum(map(len, blue)) >= 8:
        ink = np.median(np.vstack(blue), axis=0)
        seed = (ink + seed) / 2
    soup = geometry.fit(frame, {"xyz": seed.tolist(), "color": "soup"}, state, previous=previous)
    return soup, [len(p) for p in blue]


class Placement:
    def __init__(self, frames, params):
        self.phase = "original"
        self.decision = {"admitted": False, "reason": "rim_unobserved", "views": []}
        fork = frames[-1]
        anchor = fork["site_position"]
        fits = []
        last_rim = None
        previous_soup = None
        for frame in frames:
            view = clouds(frame)
            rim_fit = fit_rim(view, anchor, last_rim)
            row = {"step": int(frame["step"])}
            self.decision["views"].append(row)
            if rim_fit is None:
                continue
            _, rim, basis, edges = rim_fit
            last_rim = rim
            row.update(rim=rim.tolist(), basis=basis.tolist(), edge_support=edges.tolist())
            soup, blue = poses(frame, rim, view, previous_soup)
            row.update(blue_points=blue, soup=soup)
            if soup is not None:
                previous_soup = soup
                fits.append((rim, basis, np.array(soup["center"]), row["step"]))
        if not fits:
            if last_rim is not None:
                self.decision["reason"] = "soup_pose_unobserved"
            return
        rim, basis, _, _ = fits[-1]
        occupant = np.median([f[2] for f in fits], axis=0)
        table = grid(rim, basis, occupant, params)
        self.decision.update(grid=table, occupant=occupant.tolist(), rim=rim.tolist(), basis=basis.tolist())
        if len(fits) < 2 or fits[-1][3] != int(fork["step"]):
            self.decision["reason"] = "insufficient_time_support"
            return
        center_spread = np.ptp(np.stack([f[2][:2] for f in fits]), axis=0)
        rim_spread = np.ptp(np.stack([f[0][:2] for f in fits]), axis=0)
        self.decision.update(soup_span_mm=(1000*center_spread).tolist(), rim_span_mm=(1000*rim_spread).tolist())
        if max(center_spread) > .010 or max(rim_spread) > .009:
            self.decision["reason"] = "inconsistent_multiframe_fit"
            return
        if not table["centerline"]["feasible"]:
            self.decision["reason"] = "no_clear_floor_slot"
            return
        target = np.asarray(table["centerline"]["best_center_xy"])
        state = {k: fork[k] for k in ("site_position", "site_rotation", "gripper_qpos")}
        payload = geometry.fit(fork, {"xyz": anchor.tolist(), "color": "sauce"}, state)
        if payload is None:
            self.decision["reason"] = "payload_pose_unobserved"
            return
        self.rotation = state["site_rotation"]
        target = target - (np.asarray(payload["center"]) - anchor)[:2]
        low = clear_depth(clouds(fork), state, rim, basis, target)
        if low is None:
            self.decision["reason"] = "hand_sweep_unclear"
            return
        self.target = np.r_[target, rim[2] + low]
        self.high = max(anchor[2] + .012, rim[2] + .090)
        self.start = anchor.copy()
        self.phase = "raise"
        self.stable = 0
        self.open_step = None
        self.decision.update(admitted=True, reason="observed_slot_and_hand_clearance", payload=payload,
                             hand_target=self.target.tolist())

    def action(self, step, state, original):
        if not self.decision["admitted"]:
            return original, {"phase": "original", "gate": self.decision["reason"]}
        pos = state["site_position"]
        rot = state["site_rotation"]
        if self.phase == "raise" and pos[2] >= self.high - .004:
            self.phase = "align"
        delta = np.linalg.norm(pos[:2] - self.target[:2])
        if self.phase == "align" and delta <= .004 and abs(pos[2] - self.high) <= .005:
            self.phase = "lower"
        if self.phase == "lower":
            self.stable = self.stable + 1 if delta <= .004 and abs(pos[2] - self.target[2]) <= .004 else 0
            if self.stable >= 3:
                self.phase = "open"
                self.open_step = step
        if self.phase == "open" and step - self.open_step >= 14:
            self.phase = "retreat"
        goal = self.target.copy()
        if self.phase == "raise":
            goal[:2] = self.start[:2]
        if self.phase in ("align", "raise", "retreat"):
            goal[2] = self.high
        error = np.clip(goal - pos, -.005, .005)
        r = self.rotation @ rot.T
        turn = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
        command = np.r_[error/.05, np.clip(turn, -.15, .15), -1. if self.phase in ("open", "retreat") else 1.]
        return command, {"phase": self.phase, "target": goal.tolist(), "xy_error_mm": float(delta * 1000)}
