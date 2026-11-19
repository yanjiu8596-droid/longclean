"""Exact-action soup execution replay with evaluator-only contact and pose sidecars."""
import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys


HERE = Path(__file__).resolve().parent
AUDIT_SHA = "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4"
PHYSICS_SHA = "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8"
LAYOUTS = (931301, 931303)
END, FIRST_IMAGE, STRIDE = 450, 300, 5


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def checked(root, name, digest):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()) and path.is_file()
            and sha(path) == digest, f"Changed or unsafe evidence: {name}")
    return path


def simulator_main():
    """Called only by the native simulator interpreter; never writes model state."""
    require(sha(HERE / "actual_physics_simulator.py") == PHYSICS_SHA, "Changed frozen grasp instrumentation")
    import actual_physics_simulator as physics
    import numpy as np
    native = physics.instrument(physics.load_native())
    soup, basket = native.SOUP, "basket_1"

    def geoms(model, root):
        result = set()
        for geom, body in enumerate(model.geom_bodyid):
            parent = int(body)
            while parent not in (root, 0):
                parent = int(model.body_parentid[parent])
            if parent == root and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                result.add(geom)
        require(bool(result), f"Missing collision geometry for body {root}")
        return result

    def bounds(sim, group):
        low, high = [], []
        for geom in sorted(group):
            box = sim.model._model.geom_aabb[geom]
            rotation = np.asarray(sim.data.geom_xmat[geom]).reshape(3, 3)
            center = np.asarray(sim.data.geom_xpos[geom]) + rotation @ box[:3]
            half = np.abs(rotation) @ box[3:]
            low.append(center - half)
            high.append(center + half)
        return np.min(low, axis=0), np.max(high, axis=0)

    class ContactSimulator(native.Simulator):
        def __init__(self, bddl, output, seed, video):
            self._contacts_path = output / "contacts.jsonl"
            with self._contacts_path.open("x", encoding="utf-8"):
                pass
            super().__init__(bddl, output, seed, video)
            self._body_ids = {obj: self.env.env.obj_body_id[obj] for obj in (*native.GOALS, basket)}
            self._groups = {obj: geoms(self.env.env.sim.model, body) for obj, body in self._body_ids.items()}

        def capture(self, **metrics):
            result = super().capture(**metrics)
            before = native.array_hash(self.env.get_sim_state())
            require(before == result["state_sha256"], "Native capture state changed")
            sim = self.env.env.sim
            model, data = sim.model._model, sim.data._data
            basket_low, basket_high = bounds(sim, self._groups[basket])
            contacts = []
            for index in range(sim.data.ncon):
                contact = sim.data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first not in self._groups[soup] and second not in self._groups[soup]:
                    continue
                other = second if first in self._groups[soup] else first
                body = int(sim.model.geom_bodyid[other])
                geom_name = native.mujoco.mj_id2name(model, native.mujoco.mjtObj.mjOBJ_GEOM, other)
                body_name = native.mujoco.mj_id2name(model, native.mujoco.mjtObj.mjOBJ_BODY, body)
                owner = next((obj for obj, group in self._groups.items() if other in group), None)
                if owner is None:
                    names = f"{geom_name} {body_name}".lower()
                    owner = "table" if "table" in names else "gripper_or_robot" if any(x in names for x in ("finger", "gripper", "robot")) else "other"
                band = None
                if owner == basket:
                    band = ("upper_rim_band" if contact.pos[2] >= basket_high[2] - .019 else
                            "lower_floor_band" if contact.pos[2] <= basket_low[2] + .019 else "middle_or_side_band")
                force = np.zeros(6, dtype=np.float64)
                native.mujoco.mj_contactForce(model, data, index, force)
                contacts.append({"index": index, "geom1": first, "geom2": second, "other_geom_id": other,
                    "other_geom_name": geom_name, "other_body_name": body_name, "other_object": owner,
                    "basket_world_height_band_hint": band, "position_m": np.asarray(contact.pos).tolist(),
                    "distance_m": float(contact.dist), "contact_frame": np.asarray(contact.frame).reshape(3, 3).tolist(),
                    "contact_force_frame_N_Nm": force.tolist(), "normal_force_N": float(force[0])})
            poses = {obj: {"position_m": np.asarray(sim.data.body_xpos[body]).tolist(),
                           "rotation_matrix": np.asarray(sim.data.body_xmat[body]).reshape(3, 3).tolist()}
                     for obj, body in self._body_ids.items()}
            after = native.array_hash(self.env.get_sim_state())
            require(after == before, "Read-only contacts or poses changed simulator state")
            sample = {"step": result["step"], "state_sha256": before, "state_sha256_after_read": after,
                "goals": result["audit"]["goals"], "held_soup": result["audit"]["held_soup"],
                "grip_gap_m": float(np.abs(self.obs["robot0_gripper_qpos"]).sum()),
                "eef_observed_xyz_m": self.obs["robot0_eef_pos"].tolist(), "body_poses": poses,
                "basket_collision_bounds_m": {"low": basket_low.tolist(), "high": basket_high.tolist()},
                "soup_contacts": contacts}
            with self._contacts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
            return result

    native.Simulator = ContactSimulator
    sys.argv.remove("--simulator")
    native.main()


def worker_class(api):
    class ContactWorker(api.Worker):
        def __init__(self, bddl, target, seed):
            self.path = target
            target.mkdir(parents=True, exist_ok=False)
            args = [str(api.SIM_PYTHON), "-B", "-u", str(Path(__file__).resolve()), "--simulator",
                    "--output", str(target), "--bddl", str(bddl), "--seed", str(seed)]
            self.log = (target / "simulator.log").open("w")
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
            self.trace = (target / "actions.jsonl").open("x", buffering=1)
            self.npz = target / "observation.npz"
            self.current = self.recv()
    return ContactWorker


def main():
    from stage4_fixture import GPU, validate_gpu, validate_study
    from common import write
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--card", required=True, choices=GPU, type=int)
    cli.add_argument("--layout", required=True, choices=LAYOUTS, type=int)
    for name in ("study", "run", "audit", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    root, out = args.run.resolve(), args.output.resolve()
    require(sha(args.audit) == AUDIT_SHA and sha(HERE / "actual_physics_simulator.py") == PHYSICS_SHA, "Changed source audit or simulator wrapper")
    audit = read(args.audit)
    require(audit["schema"] == "v2-dense-online-audit/1" and audit["valid_native_terminal_records"] == 12, "Expected complete source audit")
    entry, = [r for r in audit["records"] if r["layout_seed"] == args.layout and r["arm"] == "adaptive1"]
    source_path = checked(root, entry["result"], entry["result_sha256"])
    result = read(source_path)
    actions_path = checked(source_path.parent, "actions.jsonl", result["actions_sha256"])
    grasp_path = checked(source_path.parent, "grasp-audit.jsonl", result["grasp_audit_sha256"])
    originals = [r for r in rows(actions_path) if r["step"] <= END]
    require(originals and originals[-1]["step"] == END and result["terminal"]["step"] >= END, "Missing cutoff anchor")
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    require(sha(bddl) == result["initial"]["bddl_sha256"], "Changed source scene")
    out.mkdir(parents=True, exist_ok=True)
    dest, marker = out / str(args.layout), out / f"soup-execution-{args.layout}.json"
    require(out != root and not dest.exists() and not marker.exists(), "Existing replay evidence or source output")
    metadata = {"schema": "v2-soup-execution-replay/1", "layout_seed": args.layout, "card": args.card,
        "source_arm": "adaptive1", "source_audit_sha256": AUDIT_SHA, "source_result": entry["result"],
        "source_result_sha256": entry["result_sha256"], "source_actions_sha256": result["actions_sha256"],
        "source_grasp_audit_sha256": result["grasp_audit_sha256"], "source_sha256": sha(Path(__file__)),
        "physics_wrapper_sha256": PHYSICS_SHA, "cutoff": END, "keyframe_start": FIRST_IMAGE, "keyframe_stride": STRIDE}
    write(marker, {**metadata, "complete": False})
    import run as api
    from PIL import Image
    from target_follow_audit import observation
    snapshots, verified = [], []
    with worker_class(api)(bddl, dest, args.layout) as worker:
        require(api.fingerprint(worker.current) == result["initial"], "Source initial state diverged")
        initial = worker.current
        (dest / "frames").mkdir()
        for old in originals:
            require(worker.current["step"] == old["start_step"] and not worker.terminal(), "Source interval/termination changed")
            used = worker.step(old["actions"], old["target"], old["instruction"])
            current = worker.current
            actual = {"start_step": old["start_step"], "target": old["target"], "instruction": old["instruction"], "actions": used, **current}
            require(actual == old, f"Full native source row diverged at {old['step']}")
            verified.append(actual)
            step = current["step"]
            if FIRST_IMAGE <= step <= END and step % STRIDE == 0:
                npz, png = dest / f"frames/{step}.npz", dest / f"frames/{step}.png"
                copyfile(worker.npz, npz)
                observation(npz, api.fingerprint(current))
                frame = api.Frame.read(npz)
                image = Image.new("RGB", (512, 256))
                for camera in range(2):
                    image.paste(frame.images[camera], (camera * 256, 0))
                image.save(png)
                snapshots.append({"step": step, "npz": npz.relative_to(dest).as_posix(), "npz_sha256": sha(npz),
                    "png": png.relative_to(dest).as_posix(), "png_sha256": sha(png), **api.fingerprint(current)})
        terminal = api.fingerprint(worker.current)
    require(rows(dest / "actions.jsonl") == verified and rows(dest / "grasp-audit.jsonl") == [r for r in rows(grasp_path) if r["step"] <= END],
            "Saved actions or independent grasp truth differ from original")
    contact_rows = rows(dest / "contacts.jsonl")
    for native, contact in zip([initial, *verified], contact_rows, strict=True):
        require(contact["step"] == native["step"] and contact["state_sha256"] == contact["state_sha256_after_read"] == native["state_sha256"]
                and contact["goals"] == native["audit"]["goals"] and contact["held_soup"] == native["audit"]["held_soup"]
                and all(pose["position_m"] == native["audit"]["positions"][obj] for obj, pose in contact["body_poses"].items()),
                "Read-only contact/pose sidecar disagrees with original native state")
    require([r["step"] for r in snapshots] == list(range(FIRST_IMAGE, END + 1, STRIDE)), "Missing selected keyframe")
    report = {**metadata, "initial": api.fingerprint(initial), "cutoff_fingerprint": terminal, "snapshots": snapshots,
        "actions_sha256": sha(dest / "actions.jsonl"), "grasp_audit_sha256": sha(dest / "grasp-audit.jsonl"),
        "contacts_sha256": sha(dest / "contacts.jsonl"), "verified_source_rows": len(verified), "contact_capture_rows": len(contact_rows),
        "source_metric_parity": "Every original native field, executed action and call boundary matched exactly.",
        "surface_hint_limit": "Basket labels are world-height bands within 19 mm of collision AABB extrema, not certified semantic rim/floor labels. Inspect actual geom names, points, normals and contact forces.",
        "scope": "Evaluator-only exact-action replay through step 450; no model, observer, scheduler or new control action."}
    write(dest / "report.json", report)
    write(marker, {**metadata, "complete": True, "report": f"{args.layout}/report.json", "report_sha256": sha(dest / "report.json")})
    print("SOUP_EXECUTION_REPLAY", args.layout, len(verified), "source rows", len(snapshots), "frames", flush=True)


if __name__ == "__main__":
    simulator_main() if "--simulator" in sys.argv else main()
