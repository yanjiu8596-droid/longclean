"""Measure near-release CAD render matching on replayed RGB-D checkpoints."""

import argparse
from contextlib import contextmanager
from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from actual_physics_simulator import load_native
from near_table_placement_pair import REFINED_ROTATION_TOL_DEG, compare_native, placement_error
from placement_orientation_feedback import DEV, rotation_angle
from placement_orientation_rgbd_probe import angle_deg, rotation
from soup_placement_feasibility_v2 import pinned, read_lines
from stage4_fixture import require, validate_study
from common import CAMERAS, SOUP, array_hash, read, sha, write


WIDTH = 256
ANGLES_DEG = (.9, 1.1)
GRID_MM = (-.30, -.15, 0., .15, .30)
SHIFT_MM = ((.12, -.08, .11), (-.12, .19, -.14), (.20, .07, -.18), (-.18, -.16, .16))
AXIS_TILT_DEG = 8.
MIN_MARGIN = .10
VIEWS = ("camera1", "camera2")
SCALES = np.array([4., 4., 4., .0002], dtype=np.float32)


def hypotheses():
    return np.array(list(product(ANGLES_DEG, GRID_MM, GRID_MM, GRID_MM)), dtype=float)


def native_assets():
    from libero.libero import get_libero_path

    base = Path(get_libero_path("assets")) / "stable_hope_objects" / "alphabet_soup"
    paths = ("alphabet_soup.xml", "visual/textured_vis.msh", "texture_map.png")
    return {name: sha(base / name) for name in paths}


@contextmanager
def replay_endpoint(native, bddl, seed, slot, result, scratch):
    inputs = {name: pinned(slot / name, digest) for name, digest in result["file_sha256"].items()}
    trace = read_lines(inputs["physics.jsonl"])
    start = trace[0]["step"]
    end = next(row["end_step"] for row in result["phases"] if row["phase"] == "lower")
    frame, = [old for old in result["frames"] if old["step"] == end]
    pinned(slot / frame["file"], frame["sha256"])
    prefix = read_lines(inputs["replayed-actions.jsonl"])
    require(prefix[-1]["step"] == start and start < end <= trace[-1]["step"], "Invalid held interval")
    with TemporaryDirectory(prefix="cad-observation-", dir=scratch) as scratch_path:
        scene = native.Simulator(bddl, Path(scratch_path), seed, False)
        try:
            for expected in prefix:
                require(scene.frame == expected["start_step"], "Changed native action boundary")
                returned = scene.advance(expected["actions"])
                actual = {**expected, **returned, "actions": expected["actions"][:returned["executed"]]}
                compare_native(actual, expected, [], False)
            require(array_hash(scene.env.get_sim_state()) == trace[0]["state_sha256"], "Changed held fork")
            for state in trace[1:end - start + 1]:
                scene.obs, _, done, _ = scene.env.step(np.asarray(state["action"], dtype=np.float32))
                scene.frame += 1
                require(scene.frame == state["step"] and bool(done) == state["environment_done"]
                        and array_hash(scene.env.get_sim_state()) == state["state_sha256"],
                        f"Changed native state at step {scene.frame}")
            captured = scene.capture()
            endpoint = trace[end - start]
            require(scene.frame == end and endpoint["held"][SOUP]
                    and captured["state_sha256"] == frame["state_sha256"] == endpoint["state_sha256"]
                    and captured["observation_sha256"] == frame["observation_sha256"],
                    "Near-release RGB-D replay differs from the archive")
            yield scene, trace[0], endpoint, frame, len(prefix)
        finally:
            scene.close()


def expand_mask(mask, steps):
    for _ in range(steps):
        padded = np.pad(mask, 1)
        rows, cols = mask.shape
        mask = np.logical_or.reduce([padded[j:j+rows, i:i+cols] for j, i in product(range(3), repeat=2)])
    return mask


def matching_view(sim, utils, camera_key):
    camera = CAMERAS[camera_key].removesuffix("_image")
    rgb, raw_depth = sim.render(width=WIDTH, height=WIDTH, camera_name=camera, depth=True)
    return rgb, utils.get_real_depth_map(sim, raw_depth)


def capture_case(scene, fork, endpoint, dest, reference):
    import mujoco

    env = scene.env.env
    sim, obj = env.sim, env.objects_dict[SOUP]
    data = sim.data._data
    visual_ids = [sim.model.geom_name2id(name) for name in obj.visual_geoms]
    require(visual_ids and all(i >= 0 for i in visual_ids), "Missing visual soup geometry")
    body = env.obj_body_id[SOUP]
    original_geom_pos = data.geom_xpos.copy()
    original_geom_mat = data.geom_xmat.copy()
    visual_pos = original_geom_pos[visual_ids]
    visual_rot = original_geom_mat[visual_ids].reshape(-1, 3, 3)
    reference_rot, actual_rot = (np.asarray(row["body_poses"][SOUP]["rotation"]) for row in (fork, endpoint))
    center = np.asarray(endpoint["body_poses"][SOUP]["position_m"])
    require(np.allclose(data.xpos[body], center, atol=1e-10, rtol=0)
            and np.allclose(data.xmat[body].reshape(3, 3), actual_rot, atol=1e-10, rtol=0),
            "Archived body pose differs from the renderer state")
    _, skew = rotation_angle(actual_rot, reference_rot)
    axis = skew / np.linalg.norm(skew)
    turn_about = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    turn_about /= np.linalg.norm(turn_about)
    utils = scene.camera_utils

    def pose(angle, shift, direction):
        target = center + np.asarray(shift) / 1000.
        rotated = rotation(angle * direction) @ reference_rot
        transform = rotated @ actual_rot.T
        # The offscreen scene reads these derived transforms. Leave the rest of
        # the rendered scene and all physics coordinates at the native checkpoint.
        data.geom_xpos[visual_ids] = (visual_pos - center) @ transform.T + target
        data.geom_xmat[visual_ids] = (transform @ visual_rot).reshape(-1, 9)

    def raw_views():
        return {view: matching_view(sim, utils, view) for view in VIEWS}

    native = raw_views()
    with np.load(scene.output / "observation.npz", allow_pickle=False) as original:
        for view in VIEWS:
            rgb, depth = native[view]
            require(np.array_equal(rgb, original[view]) and np.array_equal(depth, original[view + "_depth"]),
                    f"Direct {view} renderer differs from recorded RGB-D")
    nominal_masks = {view: [] for view in VIEWS}
    try:
        for angle in ANGLES_DEG:
            pose(angle, (0., 0., 0.), axis)
            for view in VIEWS:
                camera = CAMERAS[view].removesuffix("_image")
                seg = sim.render(width=WIDTH, height=WIDTH, camera_name=camera, segmentation=True)
                nominal_masks[view].append((seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
                                           & np.isin(seg[..., 1], visual_ids))
        pixels = {view: np.nonzero(expand_mask(np.logical_or.reduce(masks), 2))
                  for view, masks in nominal_masks.items()}
        require(all(len(rc[0]) >= 32 for rc in pixels.values()), "The model is not visible in both views")

        def flatten(frames, view):
            rgb, depth = frames[view]
            return np.column_stack((rgb[pixels[view]], depth[pixels[view]])).astype(np.float32)

        arrays = {}
        for view, rc in pixels.items():
            arrays[f"{view}_rc"] = np.stack(rc, axis=1)
            arrays[f"{view}_partial"] = rc[1] <= np.median(rc[1])
        bank = hypotheses()
        template = {view: [] for view in VIEWS}
        for angle, *shift in bank:
            pose(angle, shift, axis)
            views = raw_views()
            for view in VIEWS:
                template[view].append(flatten(views, view))
        for view in VIEWS:
            arrays[f"{view}_template"] = np.stack(template[view])
        rows, observation = [], {view: [] for view in VIEWS}
        perturbations = [("nominal", (0., 0., 0.), 0.)]
        perturbations.extend(("off_grid", shift, 0.) for shift in SHIFT_MM)
        perturbations.extend((("axis_tilt", SHIFT_MM[0], AXIS_TILT_DEG),
                              ("axis_tilt", SHIFT_MM[1], -AXIS_TILT_DEG)))
        settings = [(kind, angle, shift, tilt) for angle in ANGLES_DEG for kind, shift, tilt in perturbations]
        order = np.random.default_rng(reference["seed"]).permutation(len(settings))
        for i in order:
            kind, angle, shift, tilt = settings[i]
            pose(angle, shift, rotation(tilt * turn_about) @ axis)
            views = raw_views()
            for view in VIEWS:
                observation[view].append(flatten(views, view))
            rows.append({"condition": kind, "angle_deg": angle, "offset_mm": list(shift), "axis_tilt_deg": tilt})
        for view in VIEWS:
            observation[view].append(flatten(native, view))
            arrays[f"{view}_observation"] = np.stack(observation[view])
        rows.append({"condition": "native", "angle_deg": angle_deg(actual_rot @ reference_rot.T),
                     "offset_mm": [0., 0., 0.], "axis_tilt_deg": 0.})
    finally:
        data.geom_xpos[visual_ids] = visual_pos
        data.geom_xmat[visual_ids] = original_geom_mat[visual_ids]
        require(array_hash(scene.env.get_sim_state()) == endpoint["state_sha256"]
                and np.array_equal(data.geom_xpos, original_geom_pos) and np.array_equal(data.geom_xmat, original_geom_mat),
                "Rendering failed to restore the native visual geometry")
    restored = raw_views()
    require(all(np.array_equal(a, b) for view in VIEWS for a, b in zip(native[view], restored[view], strict=True)),
            "Restored endpoint pixels differ")
    np.savez_compressed(dest, **arrays)
    return {**reference, "pack": dest.name, "pack_sha256": sha(dest), "angle_axis_world": axis.tolist(),
            "reference_rotation": reference_rot.tolist(), "oracle_center_m": center.tolist(),
            "visible_pixels_by_angle": {view: [int(m.sum()) for m in masks] for view, masks in nominal_masks.items()},
            "observations": rows}


def capture(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") and os.environ.get("MUJOCO_GL") == "egl"
            and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "Pin the idle render device first")
    bddl, _ = validate_study(args.study.resolve())
    source_index = args.source / "orientation-pairs.json"
    index = read(source_index)
    require(set(index) == set(map(str, DEV)), "Unexpected source fixtures")
    args.output.mkdir(parents=True, exist_ok=False)
    native = load_native()
    native.ensure_libero_config()
    assets = native_assets()
    cases = []
    for seed in sorted(DEV):
        arm = args.source / "pairs" / str(seed) / "fixed"
        fixed_digest = index[str(seed)]["fixed"]
        result = read(pinned(arm / "result.json", fixed_digest))
        with replay_endpoint(native, bddl, seed, arm, result, args.output) as (scene, fork, last, frame, prefix):
            error = placement_error(last, np.asarray(result["target_m"]),
                                    np.asarray(fork["body_poses"][SOUP]["rotation"]))
            metadata = {"seed": seed, "fork_step": fork["step"], "step": last["step"], "prefix_intervals": prefix,
                        "endpoint_sha256": last["state_sha256"], "fork_sha256": fork["state_sha256"],
                        "source_result_sha256": fixed_digest, "frame_sha256": frame["sha256"], "native_error": error}
            cases.append(capture_case(scene, fork, last, args.output / f"{seed}.npz", metadata))
        print("CAD_RENDER_OK", seed, metadata["fork_step"], metadata["step"], cases[-1]["visible_pixels_by_angle"],
              cases[-1]["pack_sha256"], flush=True)
    write(args.output / "capture.json", {"cases": cases, "source_pair_sha256": sha(source_index),
          "capture_source_sha256": sha(Path(__file__)), "asset_sha256": assets, "bddl_sha256": sha(bddl),
          "width": WIDTH, "angles_deg": list(ANGLES_DEG), "position_grid_mm": list(GRID_MM)})


def best_classes(loss, bank, fixed):
    selection = (bank[:, 1:] == 0.).all(axis=1) if fixed else np.ones(len(bank), dtype=bool)
    winners = []
    for angle in ANGLES_DEG:
        indices = np.flatnonzero(selection & (bank[:, 0] == angle))
        match = int(indices[np.argmin(loss[indices])])
        winners.append({"angle_deg": angle, "offset_mm": bank[match, 1:].tolist(), "loss": float(loss[match])})
    below, above = (row["loss"] for row in winners)
    margin = abs(below - above) / max(1e-12, below + above)
    decision = "below" if below < above else "above"
    return {"decision": decision if margin >= MIN_MARGIN else "abstain", "relative_margin": margin,
            "classes": winners}


def score_pack(pack, case):
    bank = hypotheses()
    nominal = (bank[:, 1:] == 0.).all(axis=1)
    separation = {}
    for view in VIEWS:
        first, second = pack[f"{view}_template"][nominal]
        diff = np.abs(first - second)
        separation[view] = {"support_pixels": len(diff), "rgb_changed_pixels": int(np.any(diff[:, :3] > 0, axis=1).sum()),
                            "depth_changed_pixels": int((diff[:, 3] > 1e-6).sum()),
                            "mean_depth_difference_mm": float(diff[:, 3].mean() * 1000.)}
    rows = []
    for i, meta in enumerate(case["observations"]):
        full, partial, depth = {}, {}, {}
        for view in VIEWS:
            diff = np.abs(pack[f"{view}_template"] - pack[f"{view}_observation"][i])
            normalized = np.minimum(diff / SCALES, 10.)
            pixel_loss = .5 * normalized[..., :3].mean(axis=-1) + .5 * normalized[..., 3]
            full[view] = pixel_loss.mean(axis=1)
            partial[view] = pixel_loss[:, pack[f"{view}_partial"]].mean(axis=1)
            depth[view] = normalized[..., 3].mean(axis=1)
        both = (full["camera1"] + full["camera2"]) / 2.
        masked = (partial["camera1"] + partial["camera2"]) / 2.
        both_depth = (depth["camera1"] + depth["camera2"]) / 2.
        methods = {name: best_classes(loss, bank, fixed) for name, loss, fixed in (
            ("front_fixed", full["camera1"], True), ("wrist_fixed", full["camera2"], True),
            ("dual_fixed", both, True), ("dual_translation", both, False),
            ("dual_translation_partial", masked, False), ("dual_depth_translation", both_depth, False))}
        truth = "below" if meta["angle_deg"] <= REFINED_ROTATION_TOL_DEG else "above"
        rows.append({**meta, "seed": case["seed"], "truth": truth, "methods": methods})
    return {"seed": case["seed"], "separation": separation, "rows": rows}


def summarize(cases):
    rows = [row for case in cases for row in case["rows"]]
    output = {}
    for condition in ("nominal", "off_grid", "axis_tilt", "native"):
        group = [row for row in rows if row["condition"] == condition]
        output[condition] = {}
        for method in group[0]["methods"]:
            pairs = [(row["truth"], row["methods"][method]["decision"]) for row in group]
            output[condition][method] = {"total": len(group),
                "correct": sum(truth == decision for truth, decision in pairs),
                "false_allow": pairs.count(("above", "below")), "false_block": pairs.count(("below", "above")),
                "abstain": sum(decision == "abstain" for _, decision in pairs)}
    return output


def score(args):
    metadata = read(args.source / "capture.json")
    require(metadata["width"] == WIDTH and metadata["angles_deg"] == list(ANGLES_DEG)
            and metadata["position_grid_mm"] == list(GRID_MM), "The scoring protocol differs from the capture")
    cases = []
    for case in metadata["cases"]:
        with np.load(pinned(args.source / case["pack"], case["pack_sha256"]), allow_pickle=False) as pack:
            cases.append(score_pack(pack, case))
    summary = summarize(cases)
    write(args.output, {"capture_sha256": sha(args.source / "capture.json"),
                        "scoring_source_sha256": sha(Path(__file__)), "position_grid_mm": list(GRID_MM),
                        "margin_threshold": MIN_MARGIN, "summary": summary, "cases": cases})
    for condition, methods in summary.items():
        print("CAD_SCORE", condition, methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "score"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    if args.mode == "capture" and args.study is None:
        parser.error("--study is required for capture")
    (capture if args.mode == "capture" else score)(args)


if __name__ == "__main__":
    main()
