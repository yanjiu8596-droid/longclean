"""Score a wrist RGB-D template against sealed held-can release checkpoints."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from near_table_placement_pair import REFINED_ROTATION_TOL_DEG, REFINED_XY_TOL_M, placement_error
from placement_orientation_feedback import rotation_angle
from soup_placement_feasibility_v2 import Z_TOL
from common import SOUP, write


def rotation(deg):
    x, y, z = np.deg2rad(deg)
    vector = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm([x, y, z])
    return np.eye(3) + np.sinc(angle / np.pi) * vector + .5 * np.sinc(.5 * angle / np.pi) ** 2 * (vector @ vector)


def angle_deg(matrix):
    angle, _ = rotation_angle(matrix, np.eye(3))
    return float(np.rad2deg(angle))


def sensor(data, camera):
    # For a top-left image the vertically flipped raw render agrees with the camera matrix.
    rgb = data[camera][::-1].astype(np.float64)
    depth = data[camera + "_depth"][::-1].astype(np.float64)
    intrinsic = data[camera + "_intrinsic"]
    extrinsic = data[camera + "_extrinsic"]
    row, col = np.indices(depth.shape)
    rays = np.stack(((col - intrinsic[0, 2]) / intrinsic[0, 0],
                     (row - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(depth)), axis=-1)
    points = rays * depth[..., None]
    world = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    return rgb, depth, intrinsic, extrinsic, points, world - data["eef_pos"]


def bilinear(image, pixels):
    x, y = pixels.T
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    # The template starts well inside the image and corrections are bounded.
    if np.any((ix < 0) | (ix >= image.shape[1] - 1) | (iy < 0) | (iy >= image.shape[0] - 1)):
        raise ValueError("The projected template left the image")
    u, v = (x - ix)[:, None], (y - iy)[:, None]
    return ((1. - v) * ((1. - u) * image[iy, ix] + u * image[iy, ix + 1])
            + v * ((1. - u) * image[iy + 1, ix] + u * image[iy + 1, ix + 1]))


def feature_count(data, camera):
    rgb, _, _, _, _, offset = sensor(data, camera)
    r, g, b = np.moveaxis(rgb, -1, 0)
    blue = (b > r + 18) & (b > g + 6) & (b > 52)
    yellow = (r > 70) & (g > 58) & (g > b + 35) & (r > b + 44)
    near = (np.linalg.norm(offset[..., :2], axis=-1) < .044) & (np.abs(offset[..., 2]) < .052)
    return int((near & (blue | yellow)).sum())


def fit_wrist(reference, current):
    rgb0, dep0, _, extrinsic0, points, offset = sensor(reference, "camera2")
    rgb1, dep1, intr, extrinsic1, _, _ = sensor(current, "camera2")
    row, col = np.indices(dep0.shape)
    cap = (np.linalg.norm(offset[..., :2], axis=-1) < .038) & (.025 < offset[..., 2]) & (offset[..., 2] < .043)
    neutral = (np.ptp(rgb0, axis=2) <= 16) & (rgb0.min(axis=2) >= 8)
    smooth = np.hypot(*np.gradient(dep0)) < .0012
    border = (row >= 12) & (col >= 12) & (row < dep0.shape[0] - 12) & (col < dep0.shape[1] - 12)
    selected = np.argwhere(cap & neutral & smooth & border & (np.abs(dep0 - dep1) < .003))
    count = len(selected)
    if count < 60:
        return {"template_pixels": count, "estimate_deg": None}
    rr, cc = selected[np.linspace(0, count - 1, min(count, 900), dtype=int)].T
    initial = points[rr, cc]
    center = initial.mean(axis=0)
    reference_rgb = rgb0[rr, cc]
    centered_rgb = reference_rgb - reference_rgb.mean(axis=0)
    reference_variance = np.maximum(1., np.sum(centered_rgb ** 2, axis=0))
    image = np.dstack((rgb1, dep1))

    def residual(parameters):
        points1 = (initial - center) @ rotation(parameters[:3]).T + center + parameters[3:] / 1000.
        projected = points1 @ intr.T
        observed = bilinear(image, projected[:, :2] / projected[:, 2:])
        centered = observed[:, :3] - observed[:, :3].mean(axis=0)
        gain = np.clip(np.sum(centered * centered_rgb, axis=0) / reference_variance, .5, 2.)
        color = (centered - gain * centered_rgb) / 6.
        depth = ((observed[:, 3] - points1[:, 2]) * 2000.)[:, None]
        return np.column_stack((color, depth)).ravel()

    # Keep the sampled pixels fixed so moving them out of a foreground patch raises the loss.
    def cost(errors):
        return np.mean(2. * (np.sqrt(1. + errors ** 2) - 1.))

    step = .025 * np.eye(6)
    parameters = np.zeros(6)
    values = residual(parameters)
    starting = cost(values)
    for _ in range(15):
        derivative = np.stack([(residual(parameters + col) - residual(parameters - col)) / .05 for col in step], axis=1)
        weights = 1. / np.sqrt(1. + values ** 2)
        linear = derivative.T * weights
        delta = np.linalg.solve(linear @ derivative + .1 * np.eye(6), -(linear @ values))
        for scale in (1., .5, .25, .125):
            proposal = np.clip(parameters + scale * delta, [-3.] * 3 + [-2.] * 3, [3.] * 3 + [2.] * 3)
            candidate = residual(proposal)
            if cost(candidate) < cost(values):
                parameters, values = proposal, candidate
                break
        else:
            break
    change = extrinsic1[:3, :3] @ rotation(parameters[:3]) @ extrinsic0[:3, :3].T
    return {"template_pixels": count, "fit_pixels": len(rr), "estimate_deg": angle_deg(change),
            "rotation_world": change.tolist(), "rotation_wrist_deg": parameters[:3].tolist(),
            "translation_wrist_mm": parameters[3:].tolist(),
            "loss_at_identity": float(starting), "final_loss": float(cost(values))}


def checked_member(archive, name, expected_sha):
    data = archive.extractfile(name).read()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"Archive member digest mismatch: {name}")
    return data


def audit(archive, parent):
    plan = json.load(archive.extractfile(parent + "/orientation-pairs.json"))
    rows = []
    for seed in sorted(plan):
        if not plan[seed]:
            continue
        folder = f"{parent}/pairs/{seed}/fixed"
        result = json.loads(checked_member(archive, folder + "/result.json", plan[seed]["fixed"]))
        trace_bytes = checked_member(archive, folder + "/physics.jsonl", result["file_sha256"]["physics.jsonl"])
        trace = list(map(json.loads, trace_bytes.splitlines()))
        lower = next(stage for stage in result["phases"] if stage["phase"] == "lower")
        fork, final = trace[0], trace[lower["end_step"] - trace[0]["step"]]
        ref = np.array(fork["body_poses"][SOUP]["rotation"])
        true = np.array(final["body_poses"][SOUP]["rotation"]) @ ref.T
        ideal_axis = np.array(final["body_poses"][SOUP]["rotation"])[:, 2] @ ref[:, 2]
        cap = float(np.rad2deg(np.arccos(np.clip(ideal_axis, -1., 1.))))
        error = placement_error(final, np.asarray(result["target_m"]), ref)
        xyz_ok = error["xy_error_m"] <= REFINED_XY_TOL_M and error["abs_z_error_m"] <= Z_TOL
        frames = {frame["step"]: frame for frame in result["frames"]}
        shots = []
        for state in (fork, final):
            frame = frames[state["step"]]
            if frame["state_sha256"] != state["state_sha256"]:
                raise ValueError("The RGB-D frame did not match its physical state")
            payload = checked_member(archive, f"{folder}/{frame['file']}", frame["sha256"])
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                shots.append(dict(data))
        fitted = fit_wrist(*shots)
        if fitted["estimate_deg"] is not None:
            fitted["rotation_error_deg"] = angle_deg(np.asarray(fitted["rotation_world"]).T @ true)
            fitted["passes_angle_guard"] = fitted["estimate_deg"] <= REFINED_ROTATION_TOL_DEG
        rows.append({"seed": int(seed), "fork_step": fork["step"], "lower_state_step": final["step"],
                     "xyz_ok": xyz_ok, "opened": result["opening"] is not None,
                     "truth_total_deg": error["soup_rotation_error_deg"], "truth_lid_axis_change_deg": cap,
                     "ideal_normal_guard_agrees": (cap <= REFINED_ROTATION_TOL_DEG) == error["within_tolerance"]
                         if xyz_ok else None,
                     "color_pixels": {camera: [feature_count(shot, camera) for shot in shots]
                                      for camera in ("camera1", "camera2")}, "wrist_template": fitted})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sidecar = args.bundle.with_name(args.bundle.name + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    with args.bundle.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected != actual:
        raise ValueError("Evidence archive digest mismatch")
    with tarfile.open(args.bundle, "r:gz") as archive:
        parents = sorted(name.rsplit("/", 1)[0] for name in archive.getnames() if name.endswith("/orientation-pairs.json"))
        groups = {("development" if parent.endswith("-dev") else "independent"): audit(archive, parent) for parent in parents}
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.output, {"evidence_sha256": actual, "script_sha256": script_sha,
                        "opening_angle_limit_deg": REFINED_ROTATION_TOL_DEG, "groups": groups})
    for group, rows in groups.items():
        for row in rows:
            print(group, row["seed"], row["xyz_ok"],
                  "truth, ideal normal, wrist angle", row["truth_total_deg"], row["truth_lid_axis_change_deg"],
                  row["wrist_template"]["estimate_deg"], row["wrist_template"].get("rotation_error_deg"),
                  row["color_pixels"], row["wrist_template"]["template_pixels"], flush=True)


if __name__ == "__main__":
    main()
