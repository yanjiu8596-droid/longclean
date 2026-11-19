"""Audit front-camera pose registration in the measured gripper frame."""

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from placement_orientation_rgbd_probe import angle_deg, bilinear, checked_member, rotation, sensor
from near_table_placement_pair import REFINED_ROTATION_TOL_DEG, REFINED_XY_TOL_M, placement_error
from soup_placement_feasibility_v2 import Z_TOL
from common import SOUP, sha, write


# Degrees and millimetres in the least-squares parameter vector.
LIMIT = np.array([2.5] * 3 + [2.] * 3)
DIFFERENCE = .025 * np.eye(6)
OFFSETS = (1, 2, 4)
MIN_PIXELS = 24
COHORTS = ("development", "replay")


@dataclass(frozen=True)
class View:
    intr: np.ndarray
    cam_rotation: np.ndarray
    cam_position: np.ndarray
    grip_rotation: np.ndarray
    grip_position: np.ndarray
    image: np.ndarray
    smooth: np.ndarray
    label_points: np.ndarray
    label_color: np.ndarray

    @classmethod
    def read(cls, raw):
        rgb, depth, intr, camera, camera_xyz, offset = sensor(raw, "camera1")
        grip = raw["camera2_extrinsic"][:3, :3]
        smooth = np.hypot(*np.gradient(depth)) * (depth.shape[0] / 256.) <= .0025
        red, green, blue = np.moveaxis(rgb, -1, 0)
        ink = ((blue > red + 18) & (blue > green + 6) & (blue > 52)
               | (red > 70) & (green > 58) & (green > blue + 35) & (red > blue + 44))
        near = (np.linalg.norm(offset[..., :2], axis=-1) < .044) & (np.abs(offset[..., 2]) < .052)
        mask = ink & smooth & near
        brightness = np.maximum(12., rgb.sum(axis=2))
        color = np.stack((4. * np.log(brightness), 25. * red / brightness, 25. * blue / brightness), axis=-1)
        return cls(intr, camera[:3, :3], camera[:3, 3], grip, raw["eef_pos"],
                   np.dstack((color, depth)), smooth, offset[mask] @ grip, color[mask])

    def project(self, points):
        camera = (points @ self.grip_rotation.T + self.grip_position - self.cam_position) @ self.cam_rotation
        homogeneous = camera @ self.intr.T
        xy = homogeneous[:, :2] / homogeneous[:, 2:]
        return xy, camera[:, 2]


def visible_template(reference, target):
    xy, depth = target.project(reference.label_points)
    h, w = target.smooth.shape
    margin = 4 * (h / 256.)
    inside = ((xy[:, 0] > margin) & (xy[:, 0] < w - margin - 1)
              & (xy[:, 1] > margin) & (xy[:, 1] < h - margin - 1) & (depth > 0))
    ids = np.flatnonzero(inside)
    proj = xy[ids]
    col, row = np.rint(proj).astype(int).T
    observed = bilinear(target.image, proj)
    ids = ids[target.smooth[row, col] & (np.abs(observed[:, 3] - depth[ids]) < .003)]
    return reference.label_points[ids], reference.label_color[ids]


def pair_fit(first, second, *, use_color):
    source, source_color = visible_template(first, second)
    target, target_color = visible_template(second, first)
    counts = [len(source), len(target)]
    result = {"pixels": counts, "relative_rotation_deg": None}
    if (len(source) < MIN_PIXELS * first.smooth.size / 256 ** 2
            or len(target) < MIN_PIXELS * second.smooth.size / 256 ** 2):
        return {**result, "status": "insufficient_shared_texture"}

    def view_residual(view, xyz, color):
        xy, depth = view.project(xyz)
        observed = bilinear(view.image, xy)
        geometry = (observed[:, 3] - depth) * 2500.
        if not use_color:
            return geometry
        photometric = observed[:, :3] - color
        photometric -= np.median(photometric, axis=0)
        return np.column_stack((geometry, photometric)).ravel()

    def residual(parameters):
        rotate = rotation(parameters[:3])
        move = parameters[3:] / 1000.
        forward = view_residual(second, source @ rotate.T + move, source_color)
        inverse = view_residual(first, (target - move) @ rotate, target_color)
        return np.r_[forward, inverse]

    def cost(errors):
        return float(np.mean(2. * (np.sqrt(1. + errors ** 2) - 1.)))

    parameters = np.zeros(6)
    values = residual(parameters)
    initial = cost(values)
    for _ in range(12):
        jacobian = np.column_stack([(residual(parameters + step) - residual(parameters - step)) / .05
                                    for step in DIFFERENCE])
        weighted = jacobian.T / np.sqrt(1. + values ** 2)
        change = np.linalg.solve(weighted @ jacobian + .1 * np.eye(6), -(weighted @ values))
        for scale in (1., .5, .25, .125):
            trial = np.clip(parameters + scale * change, -LIMIT, LIMIT)
            candidate = residual(trial)
            if cost(candidate) + 1e-10 < cost(values):
                parameters, values = trial, candidate
                break
        else:
            break
    result.update(initial_cost=initial, cost=cost(values), translation_mm=parameters[3:].tolist())
    if np.any(np.abs(parameters) >= LIMIT - .02):
        return {**result, "status": "fit_at_bound"}
    return {**result, "status": "tracked", "relative_rotation_deg": parameters[:3].tolist()}


def temporal_fit(views, steps, direct):
    links = {(i, i + d) for i in range(len(views)) for d in OFFSETS if i + d < len(views)}
    links.add((0, len(views) - 1))
    measured = []
    for i, j in sorted(links):
        pair = direct if (i, j) == (0, len(views) - 1) else pair_fit(views[i], views[j], use_color=True)
        measured.append({"i": i, "j": j, "steps": [steps[i], steps[j]], **pair})

    good = [(edge["i"], edge["j"], edge) for edge in measured if edge["status"] == "tracked"]
    connected = {0}
    while True:
        extended = connected | {b for i, j, _ in good for a, b in ((i, j), (j, i)) if a in connected}
        if extended == connected:
            break
        connected = extended
    last = len(views) - 1
    diagnostic = {"connected_frames": len(connected), "edge_results": measured, "relative_rotation_deg": None}
    if last not in connected:
        return {**diagnostic, "status": "final_unreachable"}
    good = [(i, j, edge) for i, j, edge in good if i in connected and j in connected]
    nodes = {i: k for k, i in enumerate(sorted(connected))}
    equations = np.zeros((len(good), len(nodes)))
    for index, (left, right, _) in enumerate(good):
        equations[index, nodes[left]], equations[index, nodes[right]] = -1., 1.
    # Gripper-relative drift is small. Combine rotation vectors in the local
    # tangent space while anchoring the first frame to zero.
    equations = equations[:, 1:]
    offsets = np.array([edge["relative_rotation_deg"] for _, _, edge in good])
    support = np.sqrt([min(edge["pixels"]) for _, _, edge in good])
    weights = support.copy()
    for _ in range(5):
        solved, _, _, _ = np.linalg.lstsq(equations * weights[:, None], offsets * weights[:, None], rcond=None)
        error_deg = np.linalg.norm(equations @ solved - offsets, axis=1)
        weights = support * np.sqrt(np.minimum(1., .25 / np.maximum(error_deg, 1e-12)))
    return {**diagnostic, "status": "tracked", "used_links": len(good),
            "median_link_residual_deg": float(np.median(error_deg)),
            "relative_rotation_deg": solved[nodes[last] - 1].tolist()}


def score_fit(result, reference, final, true_world_rotation):
    if result["relative_rotation_deg"] is None:
        return {**result, "angle_deg": None, "error_deg": None, "decision": "abstain"}
    relative = rotation(result["relative_rotation_deg"])
    estimated = final.grip_rotation @ relative @ reference.grip_rotation.T
    angle = angle_deg(estimated)
    return {**result, "angle_deg": angle, "error_deg": angle_deg(estimated.T @ true_world_rotation),
            "decision": "allow" if angle <= REFINED_ROTATION_TOL_DEG else "block"}


def audit_cohort(archive, parent):
    index = json.load(archive.extractfile(parent + "/orientation-pairs.json"))
    rows = []
    for seed, arms in sorted(index.items()):
        if not arms:
            continue
        folder = f"{parent}/pairs/{seed}/fixed"
        result = json.loads(checked_member(archive, folder + "/result.json", arms["fixed"]))
        raw_trace = checked_member(archive, folder + "/physics.jsonl", result["file_sha256"]["physics.jsonl"])
        trace = {row["step"]: row for row in map(json.loads, raw_trace.splitlines())}
        start = min(trace)
        end = next(s["end_step"] for s in result["phases"] if s["phase"] == "lower")
        views, steps = [], []
        for frame in result["frames"]:
            step = frame["step"]
            if not start <= step <= end:
                continue
            if frame["state_sha256"] != trace[step]["state_sha256"] or not trace[step]["held"][SOUP]:
                raise ValueError("A frame did not join the held-can trace")
            payload = checked_member(archive, folder + "/" + frame["file"], frame["sha256"])
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                views.append(View.read(data))
                steps.append(step)
        if steps != sorted(set(steps)) or steps[0] != start or steps[-1] != end:
            raise ValueError("Expected both archived held-can endpoints in order")

        reference, final = views[0], views[-1]
        depth = pair_fit(reference, final, use_color=False)
        direct = pair_fit(reference, final, use_color=True)
        temporal = temporal_fit(views, steps, direct)

        # Only scoring reads object pose or physical release outcomes.
        ref_rotation = np.asarray(trace[start]["body_poses"][SOUP]["rotation"])
        true_world = np.asarray(trace[end]["body_poses"][SOUP]["rotation"]) @ ref_rotation.T
        error = placement_error(trace[end], np.asarray(result["target_m"]), ref_rotation)
        if error["xy_error_m"] > REFINED_XY_TOL_M or error["abs_z_error_m"] > Z_TOL:
            raise ValueError("The selected checkpoint does not meet the pose benchmark's XYZ limits")
        methods = {"gripper_prior": {"relative_rotation_deg": [0., 0., 0.]},
                   "front_depth": depth, "front_rgbd_pair": direct, "front_rgbd_temporal": temporal}
        row = {"seed": int(seed), "frames": steps, "xyz_error": error,
               "truth_angle_deg": error["soup_rotation_error_deg"], "truth_open_allowed": error["within_tolerance"],
               "methods": {k: score_fit(fit, reference, final, true_world) for k, fit in methods.items()}}
        rows.append(row)
        print(seed, len(steps), f'true={row["truth_angle_deg"]:.4f}',
              {k: None if v["angle_deg"] is None else (round(v["angle_deg"], 4), round(v["error_deg"], 4))
               for k, v in row["methods"].items()}, flush=True)
    return rows


def summarize(rows):
    summary = {}
    for method in rows[0]["methods"]:
        entries = [row["methods"][method] for row in rows]
        errors = [entry["error_deg"] for entry in entries if entry["error_deg"] is not None]
        summary[method] = {
            "coverage": len(errors), "total": len(rows),
            "unsafe_allow": sum(not row["truth_open_allowed"] and entry["decision"] == "allow"
                                for row, entry in zip(rows, entries, strict=True)),
            "allowed_true_in_range": sum(row["truth_open_allowed"] and entry["decision"] == "allow"
                                         for row, entry in zip(rows, entries, strict=True)),
            "median_rotation_error_deg": None if not errors else float(np.median(errors)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cohort", choices=COHORTS, action="append")
    args = parser.parse_args()
    expected = args.bundle.with_name(args.bundle.name + ".sha256").read_text(encoding="ascii").split()[0]
    with args.bundle.open("rb") as bundle:
        digest = hashlib.file_digest(bundle, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("Evidence archive digest mismatch")
    groups = {}
    with tarfile.open(args.bundle, "r:gz") as archive:
        parents = [name.rsplit("/", 1)[0] for name in archive.getnames() if name.endswith("/orientation-pairs.json")]
        for parent in parents:
            cohort = "development" if parent.endswith("-dev") else "replay"
            if cohort in (args.cohort or COHORTS):
                rows = audit_cohort(archive, parent)
                groups[cohort] = {"rows": rows, "summary": summarize(rows)}
    write(args.output, {"evidence_sha256": digest, "source_sha256": sha(Path(__file__)),
                        "opening_angle_limit_deg": REFINED_ROTATION_TOL_DEG, "groups": groups})


if __name__ == "__main__":
    main()
