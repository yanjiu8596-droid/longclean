"""Joint sensor-only CAD and texture fit of a held object's relative rotation."""

from dataclasses import dataclass

import numpy as np

from placement_orientation_rgbd_probe import bilinear, rotation
from placement_rgbd_pose_sensor import (
    CAMERA_KEYS, DEPTH_SCALE_M, GUARD_BAND_DEG, RAY_PIXEL_OFFSET,
    SUPPORT_MIN, initial_pose, measured_rays, vector_degrees,
)
from near_table_placement_pair import REFINED_ROTATION_TOL_DEG


DEPTH_POINTS = 360
TEXTURE_POINTS = 192
TEXTURE_MIN = 22
TEXTURE_WEIGHT = .28
SHARED_ROTATION_PRIOR = .080
RELATIVE_ROTATION_PRIOR = .065
PARAMETER_LIMIT = np.array([6.] * 3 + [4.] * 3 + [4.] * 6)
DIFFERENCE = .035 * np.eye(12)


@dataclass(frozen=True)
class Camera:
    intrinsics: np.ndarray
    world_rotation: np.ndarray
    world_position: np.ndarray
    image: np.ndarray

    @classmethod
    def read(cls, raw, key):
        rgb = raw[key][::-1].astype(np.float64)
        brightness = np.maximum(12., rgb.sum(axis=-1))
        appearance = np.stack((32. * rgb[..., 0] / brightness,
                               32. * rgb[..., 2] / brightness, 3. * np.log(brightness)), axis=-1)
        extrinsic = raw[key + "_extrinsic"]
        return cls(raw[key + "_intrinsic"], extrinsic[:3, :3], extrinsic[:3, 3],
                   np.dstack((appearance, raw[key + "_depth"][::-1])))

    def project(self, points):
        camera = (points - self.world_position) @ self.world_rotation
        xyz = camera @ self.intrinsics.T
        return xyz[:, :2] / np.maximum(xyz[:, 2:], 1e-8) - RAY_PIXEL_OFFSET, camera[:, 2]

    def interior(self, pixels, depth, margin):
        h, w = self.image.shape[:2]
        return ((pixels[:, 0] >= margin) & (pixels[:, 0] < w - margin - 1)
                & (pixels[:, 1] >= margin) & (pixels[:, 1] < h - margin - 1) & (depth > 0.))


@dataclass(frozen=True)
class TextureTerm:
    source: int
    target: int
    view: str
    points: np.ndarray
    appearance: np.ndarray
    camera: Camera

    def residual(self, centers, rotations):
        moved = ((self.points - centers[self.source]) @ rotations[self.source]
                 @ rotations[self.target].T + centers[self.target])
        pixels, z = self.camera.project(moved)
        visible = self.camera.interior(pixels, z, 1)
        h, w = self.camera.image.shape[:2]
        clipped = np.clip(pixels, [1., 1.], [w - 2.1, h - 2.1])
        observed = bilinear(self.camera.image, clipped)
        color = observed[:, :3] - self.appearance
        if np.any(visible):
            color[:, 2] -= np.median(color[visible, 2])
        color[:, 2] *= .35
        color = np.where(visible[:, None], color, 6.)
        return (TEXTURE_WEIGHT * color).ravel()


def texture_terms(observations, centers, rotations, groups):
    cameras = [{view: Camera.read(obs, view) for view in CAMERA_KEYS} for obs in observations]
    terms = []
    for source, target in ((0, 1), (1, 0)):
        for index, view in enumerate(CAMERA_KEYS):
            origin, rays, depth = groups[source][index]
            points = origin + rays * depth[:, None]
            first, second = cameras[source][view], cameras[target][view]
            shifted = (points - centers[source]) @ rotations[source] @ rotations[target].T + centers[target]
            sampled, first_depth = first.project(points)
            projected, second_depth = second.project(shifted)
            inside = first.interior(sampled, first_depth, 2) & second.interior(projected, second_depth, 7)
            ix = np.flatnonzero(inside)
            observed = bilinear(second.image, projected[ix])
            ix = ix[np.abs(observed[:, 3] - second_depth[ix]) <= .0025]
            if len(ix) < TEXTURE_MIN:
                continue
            first_appearance = bilinear(first.image, sampled[ix])[:, :3]
            # Chroma rejects grey background and retains the wrapped label in
            # the wrist view even when that label is too dark for the front mask.
            chroma = np.abs(first_appearance[:, :2] - 32. / 3.).max(axis=1)
            colored = chroma >= 1.05
            ix, first_appearance = ix[colored], first_appearance[colored]
            if len(ix) < TEXTURE_MIN:
                continue
            keep = np.linspace(0, len(ix) - 1, min(len(ix), TEXTURE_POINTS), dtype=int)
            terms.append(TextureTerm(source, target, view, points[ix[keep]], first_appearance[keep], second))
    return terms


def held_poses(centers, grip, mount, parameters):
    shared = rotation(parameters[:3]) @ mount
    orientations = (grip[0] @ shared, grip[1] @ rotation(parameters[3:6]) @ shared)
    return centers + parameters[6:].reshape(2, 3) / 1000., orientations


def depth_errors(model, centers, rotations, groups):
    errors, bias = [], []
    for center, attitude, group in zip(centers, rotations, groups, strict=True):
        model.pose(center, attitude)
        pair, offsets = [], []
        for view, (origin, rays, recorded) in zip(CAMERA_KEYS, group, strict=True):
            predicted = model.depth(origin, rays)
            valid = predicted > 0.
            residual = predicted - recorded
            offset = float(np.median(residual[valid])) if view == "camera1" and np.any(valid) else 0.
            pair.append(np.where(valid, residual - offset, .006))
            offsets.append(offset)
        errors.append(pair)
        bias.append(offsets)
    return errors, bias


def solve_joint(observations, calibration, model, *, include_depth_control):
    from scipy.optimize import least_squares

    centers, initial = zip(*(initial_pose(o, calibration) for o in observations), strict=True)
    centers = np.asarray(centers)
    grip = [o["camera2_extrinsic"][:3, :3] for o in observations]
    mount = np.asarray(calibration["grip_rotation"])
    gaps = [float(np.abs(frame["gripper_qpos"]).sum()) for frame in observations]
    groups, counts = [], []
    for center, attitude, observation in zip(centers, initial, observations, strict=True):
        model.pose(center, attitude)
        views = [measured_rays(observation, key, center, model, DEPTH_POINTS) for key in CAMERA_KEYS]
        groups.append([points for points, _ in views])
        counts.append([count for _, count in views])
    support = {"depth_support_pixels": counts, "gripper_gap_m": [v if np.isfinite(v) else None for v in gaps]}
    unavailable = (any(not np.isfinite(gap) or not .025 <= gap <= .085 for gap in gaps)
                   or any(n < threshold for count in counts for n, threshold in zip(count, SUPPORT_MIN, strict=True)))
    if unavailable:
        result = {**support, "angle_deg": None, "decision": "abstain", "raw_decision": "abstain",
                  "reason": "insufficient_two_view_support_or_gripper_gap"}
        return {"joint_texture": result, **({"joint_depth": result} if include_depth_control else {})}

    output, seed_parameters = {}, np.zeros(12)
    for method in ("joint_depth", "joint_texture"):
        with_texture = method == "joint_texture"
        terms = []
        if with_texture:
            predicted_centers, predicted_rotations = held_poses(centers, grip, mount, seed_parameters)
            terms = texture_terms(observations, predicted_centers, predicted_rotations, groups)
            if not terms:
                output[method] = {**support, "angle_deg": None, "decision": "abstain", "raw_decision": "abstain",
                                  "reason": "no_shared_label_texture"}
                continue

        def residual(parameters):
            fitted_centers, fitted_rot = held_poses(centers, grip, mount, parameters)
            depths, _ = depth_errors(model, fitted_centers, fitted_rot, groups)
            values = [r / DEPTH_SCALE_M * np.sqrt(DEPTH_POINTS / len(r)) for pair in depths for r in pair]
            values.append(np.sqrt(DEPTH_POINTS) *
                          np.r_[SHARED_ROTATION_PRIOR * parameters[:3], RELATIVE_ROTATION_PRIOR * parameters[3:6]])
            values.extend(t.residual(fitted_centers, fitted_rot) for t in terms)
            return np.concatenate(values)

        def jacobian(parameters):
            return np.column_stack([(residual(parameters + v) - residual(parameters - v)) / .070
                                    for v in DIFFERENCE])

        result = least_squares(residual, seed_parameters, jac=jacobian, bounds=(-PARAMETER_LIMIT, PARAMETER_LIMIT),
                               loss="soft_l1", max_nfev=28, ftol=4e-5, xtol=4e-5)
        seed_parameters = result.x
        fitted_centers, fitted_rot = held_poses(centers, grip, mount, result.x)
        depths, bias = depth_errors(model, fitted_centers, fitted_rot, groups)
        medians = [[float(np.median(np.abs(r)) * 1000.) for r in group] for group in depths]
        data = {**support, "fit_parameters": result.x.tolist(), "cost": float(result.cost),
                "evaluations": int(result.nfev), "median_depth_residual_mm": medians,
                "texture_terms": [{"source": t.source, "target": t.target, "camera": t.view, "points": len(t.points)}
                                  for t in terms],
                "fitted_axial_bias_mm": (np.asarray(bias) * 1000.).tolist(),
                "reference_fit": {"rotation": fitted_rot[0].tolist(), "center_m": fitted_centers[0].tolist()},
                "endpoint_fit": {"rotation": fitted_rot[1].tolist(), "center_m": fitted_centers[1].tolist()}}
        if max(max(pair) for pair in medians) > .40 or np.any(np.abs(result.x) >= PARAMETER_LIMIT - .02):
            output[method] = {**data, "angle_deg": None, "decision": "abstain", "raw_decision": "abstain",
                              "reason": "joint_fit_residual_or_parameter_bound"}
            continue
        correction = vector_degrees(fitted_rot[0] @ fitted_rot[1].T)
        angle = float(np.linalg.norm(correction))
        raw = "allow" if angle <= REFINED_ROTATION_TOL_DEG else "block"
        near = abs(angle - REFINED_ROTATION_TOL_DEG) <= GUARD_BAND_DEG
        output[method] = {**data, "angle_deg": angle, "raw_decision": raw, "decision": "abstain" if near else raw,
                          "reason": "near_angle_boundary" if near else None, "correction_world_deg": correction.tolist()}
    return output if include_depth_control else {"joint_texture": output["joint_texture"]}
