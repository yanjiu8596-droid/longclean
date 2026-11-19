"""Fit an isolated CAD mesh to two measured depth images near a held-soup grasp."""

from pathlib import Path

import numpy as np

from placement_orientation_feedback import CAP_DEG, position_ok, rotation_angle
from placement_orientation_rgbd_probe import rotation
from near_table_placement_pair import REFINED_ROTATION_TOL_DEG, placement_error
from stage4_fixture import require
from common import SOUP, sha


CAMERA_KEYS = ("camera1", "camera2")
SAMPLE_LIMIT = 768
SUPPORT_MIN = (48, 160)
SUPPORT_RANGE_M = .0032
DEPTH_SCALE_M = .00020
FIT_LIMIT = np.array([6.] * 3 + [4.] * 3)
GUARD_BAND_DEG = .10
RAY_PIXEL_OFFSET = .5


def vector_degrees(matrix):
    angle, skew = rotation_angle(matrix, np.eye(3))
    size = np.linalg.norm(skew)
    return np.degrees(angle) * skew / size if size > 1e-12 else np.zeros(3)


class SoupMesh:
    """The tracker owns a one-object model, with no scene poses or simulator state."""

    def __init__(self, mesh_path):
        import mujoco

        self.api = mujoco
        self.mesh_sha256 = sha(mesh_path)
        self.model = mujoco.MjModel.from_xml_string(
            """<mujoco model="soup_depth">
              <asset><mesh name="soup_mesh" file="soup.msh" scale=".01 .01 .01"/></asset>
              <worldbody>
                <body name="soup"><freejoint/>
                  <geom name="surface" type="mesh" mesh="soup_mesh" group="1" contype="0" conaffinity="0"/>
                </body>
              </worldbody>
            </mujoco>""",
            {"soup.msh": Path(mesh_path).read_bytes()},
        )
        self.data = mujoco.MjData(self.model)
        self.groups = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

    def pose(self, center, attitude):
        self.data.qpos[:3] = center
        self.api.mju_mat2Quat(self.data.qpos[3:], np.asarray(attitude, dtype=np.float64).reshape(-1))
        self.api.mj_kinematics(self.model, self.data)

    def depth(self, origin, rays):
        # Unnormalized camera rays have camera z=1, so the intersection
        # parameter is axial depth in metres, the same quantity as the archive.
        n = len(rays)
        dist = np.empty(n, dtype=np.float64)
        ids = np.empty(n, dtype=np.int32)
        self.api.mj_multiRay(
            self.model, self.data, np.ascontiguousarray(origin), np.ascontiguousarray(rays.reshape(-1)),
            self.groups, True, -1, ids, dist, n, 3.,
        )
        return dist


def asset_path():
    from libero.libero import get_libero_path

    return Path(get_libero_path("assets")) / "stable_hope_objects/alphabet_soup/visual/textured_vis.msh"


def initial_pose(observation, calibration):
    wrist = observation["camera2_extrinsic"][:3, :3]
    center = observation["eef_pos"] + wrist @ np.asarray(calibration["grip_center_m"])
    attitude = wrist @ np.asarray(calibration["grip_rotation"])
    return center, attitude


def measured_rays(observation, camera, center, model, max_points):
    depth = observation[camera + "_depth"][::-1].astype(np.float64)
    k = observation[camera + "_intrinsic"]
    camera_world = observation[camera + "_extrinsic"]
    rows, cols = np.indices(depth.shape)
    cam = np.stack(((cols + RAY_PIXEL_OFFSET - k[0, 2]) / k[0, 0],
                    (rows + RAY_PIXEL_OFFSET - k[1, 2]) / k[1, 1], np.ones_like(depth)), axis=-1)
    rays = cam @ camera_world[:3, :3].T
    origin = camera_world[:3, 3].copy()
    xyz = origin + rays * depth[..., None]
    smooth = np.hypot(*np.gradient(depth)) < .003
    near = np.linalg.norm(xyz - center, axis=-1) <= .052
    allowed = np.isfinite(depth) & (depth > 0.) & smooth & near
    if camera == "camera1":
        rgb = observation[camera][::-1].astype(np.int16)
        red, green, blue = np.moveaxis(rgb, -1, 0)
        ink = ((blue > red + 18) & (blue > green + 6) & (blue > 52)
               | (red > 70) & (green > 58) & (green > blue + 35) & (red > blue + 44))
        allowed &= ink
    measured, direction = depth[allowed], rays[allowed]
    predicted = model.depth(origin, direction) if len(direction) else np.empty(0)
    supported = (predicted > 0.) & (np.abs(predicted - measured) <= SUPPORT_RANGE_M)
    count = int(supported.sum())
    if count > max_points:
        indexes = np.linspace(0, count - 1, max_points, dtype=int)
    else:
        indexes = slice(None)
    return (origin, direction[supported][indexes], measured[supported][indexes]), count


def fit(observation, calibration, model):
    from scipy.optimize import least_squares

    center, attitude = initial_pose(observation, calibration)
    gap = float(np.abs(observation["gripper_qpos"]).sum())
    if not np.isfinite(gap) or gap < .025 or gap > .085:
        return {"status": "invalid_gripper_gap", "gripper_gap_m": gap}
    model.pose(center, attitude)
    prepared = [measured_rays(observation, name, center, model, SAMPLE_LIMIT) for name in CAMERA_KEYS]
    counts = [n for _, n in prepared]
    provenance = {"gripper_gap_m": gap, "initial_center_m": center.tolist(),
                  "initial_rotation": attitude.tolist(), "support_pixels": counts}
    if any(count < minimum for count, minimum in zip(counts, SUPPORT_MIN, strict=True)):
        return {**provenance, "status": "insufficient_two_view_support"}
    groups = [term for term, _ in prepared]

    def errors(parameters):
        model.pose(center + parameters[3:] / 1000., rotation(parameters[:3]) @ attitude)
        result, biases = [], []
        for view, (origin, rays, observed) in zip(CAMERA_KEYS, groups, strict=True):
            depth = model.depth(origin, rays)
            valid = depth > 0.
            delta = depth - observed
            bias = float(np.median(delta[valid])) if view == "camera1" and np.any(valid) else 0.
            result.append(np.where(valid, delta - bias, .006))
            biases.append(bias)
        return result, biases

    def residual(parameters):
        terms, _ = errors(parameters)
        return np.concatenate([r / DEPTH_SCALE_M * np.sqrt(SAMPLE_LIMIT / len(r)) for r in terms])

    step = .035 * np.eye(6)

    def derivative(parameters):
        return np.column_stack([(residual(parameters + v) - residual(parameters - v)) / .070 for v in step])

    result = least_squares(residual, np.zeros(6), jac=derivative, bounds=(-FIT_LIMIT, FIT_LIMIT),
                           loss="soft_l1", max_nfev=36, ftol=2e-5, xtol=2e-5)
    params = result.x
    final_errors, biases = errors(params)
    medians = [float(np.median(np.abs(item)) * 1000.) for item in final_errors]
    data = {**provenance, "median_depth_residual_mm": medians, "fitted_axial_bias_mm": [b * 1000. for b in biases],
            "evaluations": int(result.nfev),
            "center_m": (center + params[3:] / 1000.).tolist(),
            "rotation": (rotation(params[:3]) @ attitude).tolist(), "correction_from_init": params.tolist(),
            "cost": float(result.cost)}
    if max(medians) > .40:
        return {**data, "status": "depth_fit_residual"}
    if np.any(np.abs(params) >= FIT_LIMIT - .02):
        return {**data, "status": "fit_at_bound"}
    return {**data, "status": "fit"}


def estimate(reference, endpoint, calibration, model, reference_fit=None):
    earlier = fit(reference, calibration, model) if reference_fit is None else reference_fit
    later = fit(endpoint, calibration, model)
    output = {"reference_fit": earlier, "endpoint_fit": later}
    if earlier["status"] != "fit" or later["status"] != "fit":
        return {**output, "angle_deg": None, "decision": "abstain", "raw_decision": "abstain",
                "reason": "pose_fit_unavailable"}
    correction = np.asarray(earlier["rotation"]) @ np.asarray(later["rotation"]).T
    vector = vector_degrees(correction)
    angle = float(np.linalg.norm(vector))
    raw = "allow" if angle <= REFINED_ROTATION_TOL_DEG else "block"
    uncertainty = abs(angle - REFINED_ROTATION_TOL_DEG) <= GUARD_BAND_DEG
    return {**output, "angle_deg": angle, "raw_decision": raw, "decision": "abstain" if uncertainty else raw,
            "reason": "near_angle_boundary" if uncertainty else None, "correction_world_deg": vector.tolist()}


def score_estimate(estimate, fork, sample, target):
    a, b = (row["body_poses"][SOUP] for row in (fork, sample))
    reference, current = np.asarray(a["rotation"]), np.asarray(b["rotation"])
    truth_rotation = reference @ current.T
    true_vector = vector_degrees(truth_rotation)
    true_angle = float(np.linalg.norm(true_vector))
    error = placement_error(sample, np.asarray(target), reference)
    require(abs(error["soup_rotation_error_deg"] - true_angle) < 1e-6, "Truth rotation convention differs")
    truth_gate = "allow" if true_angle <= REFINED_ROTATION_TOL_DEG else "block"
    row = {"truth": truth_gate, "xyz_qualified": position_ok(error), "held": sample["held"][SOUP],
           "truth_angle_deg": true_angle, "truth_world_correction_deg": true_vector.tolist(),
           "native_position_error": error, "sensor": estimate}
    if estimate["angle_deg"] is None:
        return row
    estimated = np.asarray(estimate["correction_world_deg"])
    size = float(np.linalg.norm(estimated))
    bounded = estimated * min(1., CAP_DEG / max(size, 1e-12))
    remaining = np.linalg.norm(vector_degrees(rotation(bounded) @ current @ reference.T))
    residual = float(remaining)
    dot = float(estimated @ true_vector)
    direction = None if min(size, true_angle) < 1e-8 else dot / (size * true_angle)
    return {**row, "rotation_error_deg": float(np.linalg.norm(vector_degrees(rotation(estimated) @ truth_rotation.T))),
            "direction_cosine": direction, "bounded_residual_deg": residual,
            "bounded_reduces_angle": residual + 1e-8 < true_angle,
            "bounded_recovers_under_limit": residual <= REFINED_ROTATION_TOL_DEG}
