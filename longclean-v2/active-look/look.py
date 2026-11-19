"""Paired placement trajectories with a shared release and different camera motion."""
from pathlib import Path
from shutil import copyfile
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "soup-study"))
from probe import inks
from common import CHEESE, SAUCE, SOUP, V1, read, sha, write
from run import Worker, fingerprint
import run
from vision import Frame, cloud

# Worker looks up its executable here, for this runner only.
run.D = HERE
FORK, END = 260, 295
BLIND = tuple(range(927100, 927108))
CONTROLS = {927100, 927102, 927104, 927106}
ITEMS = (SOUP, "basket_1", CHEESE, SAUCE)


def clamp_offset(d, limit):
    return np.asarray(d) * min(1.0, limit / max(1e-9, np.linalg.norm(d)))


def sight(frame, mask):
    xyz = cloud(frame, 1)
    x, y, z = np.moveaxis(xyz, -1, 0)
    d = xyz - frame.geometry["eef"]
    pocket = ((-.055 < x) & (x < .065) & (.195 < y) & (y < .335) & (.485 < z) & (z < .561)
              & ((d[..., 2] < -.072) | (np.linalg.norm(d[..., :2], axis=-1) > .065)))
    blue, yellow = inks(frame.images[1])
    counts = [int((mask & ink).sum()) for ink in (blue, yellow)]
    pocket_counts = [int((mask & pocket & ink).sum()) for ink in (blue, yellow)]
    return {"pixels": int(mask.sum()), "label_pixels": counts, "pocket_label_pixels": pocket_counts,
            "usable": min(pocket_counts) >= 24}


def trace(seed, condition, arm, bddl, asset, out):
    with Worker(bddl, out, seed) as worker:
        a = worker.current["audit"]
        initial = fingerprint(worker.current)
        safe = np.array(a["site_xyz"])
        safe[2] = max(.74, safe[2])
        basket = np.array(a["positions"]["basket_1"])
        grab = np.array(a["positions"][SOUP]) + asset["grasp_xyz_offset_m"]
        high = grab + [0, 0, .16]
        side = -1 if seed % 2 else 1
        over = np.array([basket[0] + side * .022, basket[1] - (.025 if condition == "inside" else .096), .67])
        lower = over.copy()
        lower[2] = .555 if condition == "inside" else .60
        rotation = np.array(asset["base_rotation_matrix"])

        def drive(point, grip):
            now = worker.current["audit"]
            dp = 1.4 * (point - np.asarray(now["site_xyz"])) / .05
            dr = Rotation.from_matrix(rotation @ np.asarray(now["site_rotation"]).T).as_rotvec() / .5
            action = np.r_[np.clip(np.r_[dp, dr], -.85, .85), grip].tolist()
            worker.step([action], None, "controlled camera visibility study")

        stages = [(safe, 16, -1), (high, 33, -1), (grab, 44, -1), (grab, 24, 1),
                  (high, 29, 1), (over, 44, 1), (lower, 28, 1), (lower, 20, -1), (over, 22, -1)]
        held = False
        for i, (point, steps, grip) in enumerate(stages):
            if i == 4 and not worker.current["audit"]["held_soup"]:
                return {"initial": initial, "eligible": False, "reason": "grasp failed"}
            if i == 7:
                held = worker.current["audit"]["held_soup"]
            for _ in range(steps):
                drive(point, grip)
        assert worker.current["step"] == FORK
        a = worker.current["audit"]
        expected = condition == "inside"
        eligible = held and not a["held_soup"] and a["goals"][SOUP] == expected
        checkpoint = fingerprint(worker.current)
        if not eligible:
            return {"initial": initial, "checkpoint": checkpoint, "eligible": False,
                    "reason": "shared release did not reach expected state", "fork_audit": a}
        frame = Frame.read(worker.npz)
        ext = frame.geometry["extrinsics"][1]
        direction, camera = ext[:3, 2], ext[:3, 3]
        # Camera-axis compensation uses the commanded drop, not the actual object location.
        if direction[2] >= -.4:
            raise RuntimeError("Wrist optical axis is not downward")
        hit = camera + (.51 - camera[2]) / direction[2] * direction
        offset = clamp_offset(over[:2] - hit[:2], .055)
        target = over.copy()
        if arm == "lateral":
            target[:2] += offset
        fork_xyz = {k: np.array(a["positions"][k]) for k in ITEMS}
        movement = {k: 0.0 for k in ITEMS}
        changed, samples = [], []
        view_offset = []
        for step in range(FORK + 1, END + 1):
            drive(target, -1)
            state = worker.current["audit"]
            for k in ITEMS:
                movement[k] = max(movement[k], float(np.linalg.norm(np.array(state["positions"][k]) - fork_xyz[k])))
            if state["goals"] != a["goals"] or state["held_soup"]:
                changed.append(step)
            if "mask_pixels" not in worker.current:
                continue
            frame = Frame.read(worker.npz)
            mask = np.load(out / "score-mask.npy", allow_pickle=False)
            assert mask.sum() == worker.current["mask_pixels"]
            samples.append({"step": step, **sight(frame, mask), "eef": frame.geometry["eef"].tolist()})
            view_offset.append(float(np.linalg.norm(frame.geometry["eef"][:2] - over[:2])))
            copyfile(worker.npz, out / f"{step}.npz")
            copyfile(out / "score-mask.npy", out / f"{step}-mask.npy")
        late = [s for s in samples if s["step"] >= 275]
        assert len(late) == 5
        return {"initial": initial, "checkpoint": checkpoint, "eligible": True,
                "planned_lateral_xy_m": offset.tolist(), "actual_last_offset_m": view_offset[-1],
                "max_shift_m": movement, "goals_changed_or_soup_regrasped_at": changed,
                "safe": not changed and max(movement.values()) <= .005,
                "visible_ink": any(a["usable"] and b["usable"] for a, b in zip(late, late[1:])),
                "max_visible_instance_pixels": max(s["pixels"] for s in late), "samples": samples}


def main():
    phase, card, output = sys.argv[1:]
    card, out = int(card), Path(output).resolve()
    plan = read(ROOT / "study4/plan.json")
    bddl = ROOT / "study4" / plan["bddl"]
    assert sha(bddl) == plan["bddl_sha256"]
    sources = ("look.py", "simulator.py", "launch.sh")
    code = {n: sha(HERE / n) for n in sources}
    if phase == "blind":
        freeze = read(HERE / "freeze.json")
        assert freeze["source"] == code and freeze["layouts"] == list(BLIND)
    else:
        assert phase == "smoke"
    baseline_sources = read(ROOT / "study4/frozen.json")["code"]
    assert all(sha(ROOT / n) == h for n, h in baseline_sources.items())
    out.mkdir(parents=True, exist_ok=True)
    cases = [(927090, "inside"), (927090, "outside")] if phase == "smoke" else [
        (seed, condition) for i, seed in enumerate(BLIND) if i % 2 == card - 2
        for condition in (("inside", "outside") if seed in CONTROLS else ("inside",))]
    asset = read(V1 / "grasp-reference.json")
    results = []
    for seed, condition in cases:
        base = out / f"{seed}-{condition}"
        arms = {}
        for method in ("vertical", "lateral"):
            arms[method] = trace(seed, condition, method, bddl, asset, base / method)
        a, b = arms["vertical"], arms["lateral"]
        assert a["initial"] == b["initial"] and a.get("checkpoint") == b.get("checkpoint")
        assert a["eligible"] == b["eligible"]
        results.append({"layout": seed, "condition": condition, "arms": arms})
        print(seed, condition, a["eligible"], a.get("visible_ink"), b.get("visible_ink"),
              a.get("safe"), b.get("safe"), flush=True)
    write(out / f"card-{card}.json", {"phase": phase, "source": code, "pairs": results})


if __name__ == "__main__":
    main()
