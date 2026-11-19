"""Generate physical soup-can rim failures in the unchanged LIBERO scene.

The scripted arm uses audited geometry to create test fixtures. Geometry, truth,
and contacts never go to the visual observer. No qpos, geom, or scene edits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import CHEESE, COMMIT, GPU, REPO, SAUCE, SOUP, V1, read, sha, write
from collect_counters import FOLLOW, bounds, fingerprint, gates, subtree_geoms


# Mid-sides and corner approaches, fixed before the development run.
PROFILES = ((-1., 0.), (1., 0.), (0., -1.), (0., 1.),
            (-1., -.8), (1., -.8), (-1., .8), (1., .8))


def schedule(phase, card, profile):
    start = 930000 if phase == "dev" else 930020
    return [{"layout": start + i, "case": "rim", "slot": i, "card": 2 + i % 2,
             "profile": i if profile is None else profile}
            for i in range(len(PROFILES)) if 2 + i % 2 == card]


def basket_surface(sim, basket_geom, limits, np, mujoco):
    """Vertical collision-surface samples with no change to the MuJoCo model."""
    low, high = limits
    # Follow an upward measured collision top, not the outer AABB x/y edge.
    axis = [np.arange(low[i] + .001, high[i], .003) for i in (0, 1)]
    extent = np.array([len(v) for v in axis])
    world = np.full((*extent, 3), np.nan)
    model, data = sim.model._model, sim.data._data
    hitid = np.full(1, -1, dtype=np.int32)
    ray = np.array([0., 0., -1.])
    zmin = high[2] - .019
    for i, x in enumerate(axis[0]):
        for j, y in enumerate(axis[1]):
            cursor = np.array([x, y, high[2] + .004])
            # Other bodies (usually absent here) are not basket support.
            for _ in range(6):
                distance = mujoco.mj_ray(model, data, cursor, ray, None, 1, -1, hitid)
                if not np.isfinite(distance) or distance < 0 or cursor[2] - distance < zmin:
                    break
                if int(hitid[0]) in basket_geom:
                    world[i, j] = [x, y, cursor[2] - distance]
                    break
                cursor[2] -= distance + .0003
    return world


def choose_lip(world, limits, profile, np):
    pts = world[np.isfinite(world[..., 2])]
    if len(pts) < 4:
        return None
    low, high = limits
    mid = .5 * (low + high)
    half = .5 * (high - low)
    anchor = np.asarray(PROFILES[profile])
    scored = []
    for i, point in enumerate(pts):
        relative = (point[:2] - mid[:2]) / half[:2]
        if np.abs(relative).max() < .62:
            continue
        delta = pts[:, :2] - point[:2]
        neighbors = (np.max(np.abs(delta), axis=1) <= .010) & (np.abs(pts[:, 2] - point[2]) <= .003)
        d = delta[neighbors]
        # Opposing near-coplanar surface samples distinguish a crest capable
        # of bearing a nearly centered can from an outer sloping edge.
        balance = min((d[:, 0] < 0).sum(), (d[:, 0] > 0).sum()) + min((d[:, 1] < 0).sum(), (d[:, 1] > 0).sum())
        cost = float(np.linalg.norm(relative - anchor)) - .010 * min(20, int(balance)) - .002 * min(30, len(d))
        scored.append((cost, i, int(balance), len(d)))
    if not scored:
        return None
    _, i, balance, count = min(scored)
    return {"xyz": pts[i].tolist(), "support_balance": balance, "near_level_grid_points": count,
            "grid_resolution_m": .003, "near_level_tolerance_m": .003,
            "profile_anchor": anchor.tolist()}


def capture(job, bddl, out, asset, stride, np, Rotation, mujoco, Simulator):
    folder = out / str(job["layout"])
    frames_dir = folder / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    (folder / "trace").mkdir()
    scene = Simulator(bddl, folder / "trace", job["layout"], False)
    try:
        env = scene.env.env
        sim = env.sim
        robot = env.robots[0]
        site = robot.controller.eef_name
        robot_root = robot.robot_model.root_body
        robot_ids = subtree_geoms(sim.model, sim.model.body_name2id(robot_root))
        gripper_ids = {sim.model.geom_name2id(name) for name in robot.gripper.contact_geoms}
        robot_ids |= gripper_ids
        basket_ids = subtree_geoms(sim.model, env.obj_body_id["basket_1"])
        soup_ids = subtree_geoms(sim.model, env.obj_body_id[SOUP])
        limits = bounds(sim, basket_ids, np)
        world = basket_surface(sim, basket_ids, limits, np, mujoco)
        lip = choose_lip(world, limits, job["profile"], np)
        full, frames, coarse, actions, actions5, transitions, stages = [], [], [], [], [], [], {}

        def audit():
            rim, robot_touch = [], False
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]
                pair = {int(c.geom1), int(c.geom2)}
                if not pair & soup_ids or c.dist > .001:
                    continue
                robot_touch |= bool(pair & robot_ids)
                if pair & basket_ids and c.pos[2] >= limits[1][2] - .019:
                    rim.append({"xyz_m": c.pos.tolist(), "separation_m": float(c.dist),
                                "basket_geom": next(g for g in pair if g in basket_ids)})
            return {"step": scene.frame, "goals": scene.goals(), "positions": scene.xyz(),
                    "held": {k: bool(env._check_grasp(robot.gripper, env.objects_dict[k]))
                             for k in (SOUP, SAUCE, CHEESE)},
                    "eef_xyz": scene.obs["robot0_eef_pos"].tolist(),
                    "gap": float(np.abs(scene.obs["robot0_gripper_qpos"]).sum()),
                    "rim_contact": bool(rim), "rim_hits": rim, "robot_contact": robot_touch}

        def record(stream):
            a = full[-1]
            current = scene.capture(executed=len(actions), done=False, transitions=list(transitions),
                                    completed_regrasp_steps=0)
            (folder / "trace" / "observation.npz").replace(frames_dir / f"{scene.frame}.npz")
            current["audit"] |= {k: a[k] for k in ("gap", "eef_xyz")}
            row = {"start_step": scene.frame - len(actions), "target": None,
                   "instruction": "physical release challenge", "actions": actions[:], **current}
            frames.append(row)
            if actions:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            actions.clear()
            transitions.clear()

        initial = audit()
        full.append(initial)
        coarse.append({"step": 0, "audit": initial, "actions": []})
        previous = initial["goals"]

        def drive(name, target, grip, count, rotation, stream, stop=None, slow=False):
            nonlocal previous
            stages[name] = scene.frame
            for _ in range(count):
                if all(previous.values()):
                    break
                dpos = 1.4 * (target - sim.data.get_site_xpos(site)) / .05
                native_rot = np.asarray(sim.data.get_site_xmat(site)).reshape(3, 3)
                drot = Rotation.from_matrix(rotation @ native_rot.T).as_rotvec() / .5
                if slow:
                    dpos = np.clip(dpos, [-.08, -.08, -.03], [.08, .08, .08])
                executed = np.r_[np.clip(np.r_[dpos, drot], -.85, .85), grip].astype(np.float32)
                scene.obs, _, _, _ = scene.env.step(executed)
                scene.frame += 1
                a = audit()
                full.append(a)
                transitions.extend({"step": scene.frame, "object": key, "in_basket": bool(value)}
                                   for key, value in a["goals"].items() if value != previous[key])
                previous = a["goals"]
                actions.append(executed.tolist())
                actions5.append(executed.tolist())
                if scene.frame % 5 == 0:
                    coarse.append({"step": scene.frame, "audit": a, "actions": actions5[:]})
                    actions5.clear()
                if scene.frame % stride == 0:
                    record(stream)
                if stop is not None and stop(a):
                    break

        center = np.asarray(initial["positions"][SOUP])
        safe = np.asarray(sim.data.get_site_xpos(site)).copy()
        safe[2] = max(.74, safe[2])
        rotation = np.asarray(asset["base_rotation_matrix"])
        grab = center + asset["grasp_xyz_offset_m"]
        high = grab + [0., 0., .16]
        with (folder / "trace" / "actions.jsonl").open("x", encoding="utf-8") as stream:
            record(stream)
            if lip is not None:
                for name, target, grip, count in (("safe", safe, -1, 16), ("above_grasp", high, -1, 33),
                                                  ("approach", grab, -1, 44), ("grasp", grab, 1, 24)):
                    drive(name, target, grip, count, rotation, stream)
            if full[-1]["held"][SOUP]:
                drive("lift", high, 1, 29, rotation, stream)
            if full[-1]["held"][SOUP]:
                crest = np.asarray(lip["xyz"])
                held_center = np.asarray(full[-1]["positions"][SOUP])
                offset = np.asarray(sim.data.get_site_xpos(site)) - held_center
                bottom, _ = bounds(sim, soup_ids, np)
                above = crest + offset + [0., 0., held_center[2] - bottom[2] + .065]
                drive("transport", above, 1, 51, rotation, stream)
                if full[-1]["held"][SOUP]:
                    # Update the measured suspended clearance before descending.
                    tcp = np.asarray(sim.data.get_site_xpos(site)).copy()
                    now = np.asarray(full[-1]["positions"][SOUP])
                    bottom, _ = bounds(sim, soup_ids, np)
                    down = np.r_[tcp[:2] + crest[:2] - now[:2], tcp[2] + crest[2] - bottom[2] - .002]
                    drive("low_speed_descent", down, 1, 70, rotation, stream, slow=True,
                          stop=lambda a: a["rim_contact"] or not a["held"][SOUP])
                    if full[-1]["held"][SOUP] and full[-1]["rim_contact"]:
                        latched = np.asarray(sim.data.get_site_xpos(site)).copy()
                        angle = np.asarray(sim.data.get_site_xmat(site)).reshape(3, 3).copy()
                        drive("touch_dwell", latched, 1, 10, angle, stream, slow=True)
                        # A slow release can still slip; the measured gate, not
                        # the aim waypoint, decides whether it is a valid case.
                        drive("open", latched, -1, 20, angle, stream, slow=True)
                        upwards = latched + [0., 0., .095]
                        drive("retract", upwards, -1, 40, angle, stream)
            if actions:
                record(stream)
        quality = gates("rim", SOUP, initial, full, coarse, stages)
        measured = quality["measured_open_step"]
        tail = full[measured + FOLLOW - 4:measured + FOLLOW + 1] if measured is not None else []
        xyz = np.asarray([row["positions"][SOUP] for row in tail])
        spread = float(np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1))) if len(xyz) else None
        checks = quality["checks"]
        checks["robot_clear_last_5"] = len(tail) == 5 and not any(row["robot_contact"] for row in tail)
        checks["tail_displacement_at_most_5mm"] = len(tail) == 5 and spread <= .005
        quality["eligible"] = all(checks.values())
        quality["tail_displacement_m"] = spread
        write(folder / "physics-audit.json", {"controlled_object": SOUP, "stages": stages,
                                             "basket_collision_box": [v.tolist() for v in limits],
                                             "lip": lip, "steps": full, "release_quality": quality})
        write(folder / "fixture.json", {**job, "initial": fingerprint(frames[0]),
                                      "end": fingerprint(frames[-1]), "initial_audit": frames[0]["audit"],
                                      "end_audit": frames[-1]["audit"], "frames": "frames",
                                      "actions": "trace/actions.jsonl", "physics_audit": "physics-audit.json",
                                      "frame_steps": [r["step"] for r in frames], "release_quality": quality})
        return {"path": f'{job["layout"]}/fixture.json', "initial_sha256": frames[0]["state_sha256"],
                "layout": job["layout"], "case": "rim", **quality}
    finally:
        scene.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("card", type=int, choices=(2, 3))
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--phase", choices=("dev", "blind"), required=True)
    p.add_argument("--profile", type=int, choices=range(len(PROFILES)))
    p.add_argument("--save-stride", type=int, choices=(1, 5), default=1)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.phase == "blind" and a.profile is None:
        p.error("freeze one successful development profile before independent collection")
    jobs = schedule(a.phase, a.card, a.profile)
    if a.list:
        print(json.dumps(jobs))
        return
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != GPU[a.card]
            or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(a.card)
            or os.environ.get("MUJOCO_GL") != "egl"):
        raise RuntimeError("EGL and CUDA must name the same approved, idle GPU")
    study, out = a.study.resolve(), a.out.resolve()
    lock, plan = read(study / "frozen.json"), read(study / "plan.json")
    bddl = study / plan["bddl"]
    if (sha(study / "plan.json") != lock["plan_sha256"] or sha(bddl) != plan["bddl_sha256"]
            or not all(sha(ROOT / name) == lock["code"][name] for name in ("common.py", "simulator.py"))):
        raise RuntimeError("Frozen scenario or simulator has changed")
    if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("Unexpected repository revision")
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from simulator import Simulator, ensure_libero_config

    ensure_libero_config()
    reference = V1 / "grasp-reference.json"
    asset = read(reference)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"manifest-{a.card}.json"
    if path.exists() or any((out / str(job["layout"])).exists() for job in jobs):
        raise FileExistsError("Collection already exists; choose a fresh output root")
    manifest = {"schema": "physical-release-profiled-rim-v1", "card": a.card,
                "phase": a.phase, "planned": jobs, "complete": False, "records": [],
                "collector_sha256": sha(Path(__file__)), "gate_sha256": sha(HERE / "collect_counters.py"),
                "bddl_sha256": sha(bddl), "grasp_reference_sha256": sha(reference),
                "frame_stride": a.save_stride, "qualification_stride": 5, "followup_steps": FOLLOW,
                "commit": COMMIT, "profile": a.profile}
    write(path, manifest)
    for job in jobs:
        result = capture(job, bddl, out, asset, a.save_stride, np, Rotation, mujoco, Simulator)
        manifest["records"].append(result)
        write(path, manifest)
        print(job["layout"], job["profile"], "rim", "eligible", result["eligible"],
              [k for k, v in result["checks"].items() if not v], flush=True)
    manifest["eligible_by_case"] = {"rim": sum(row["eligible"] for row in manifest["records"])}
    manifest["complete"] = True
    write(path, manifest)


if __name__ == "__main__":
    main()
