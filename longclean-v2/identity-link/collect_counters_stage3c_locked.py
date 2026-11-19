"""Collect four real-physics negative release fixtures with the frozen LIBERO scene.

Run under SIM_PYTHON. This scripted, state-guided controller only creates
counterexamples. The scorers must receive the realized frames/actions, never its
target, object coordinates, contact bits, or acceptance flags.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import CHEESE, COMMIT, GPU, REPO, SAUCE, SOUP, V1, read, sha, write


CASES = ("outside", "rim", "other_can", "identity_unknown")
STRIDE = 5
FOLLOW = 30


def schedule(phase):
    return [
        {"layout": 929300 + i if phase == "dev" else 929320 + 8 * i + slot,
         "case": case, "slot": slot, "card": 2 + ((i if phase == "dev" else slot) % 2)}
        for i, case in enumerate(CASES)
        for slot in range(1 if phase == "dev" else 8)
    ]


def subtree_geoms(model, root):
    result = set()
    for gid, body in enumerate(model.geom_bodyid):
        parent = int(body)
        while parent not in (root, 0):
            parent = int(model.body_parentid[parent])
        if parent == root and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            result.add(gid)
    if not result:
        raise RuntimeError(f"No collision geoms found for body {root}")
    return result


def bounds(sim, geom_ids, np):
    lows, highs = [], []
    for g in sorted(geom_ids):
        geom = sim.model._model.geom_aabb[g]
        rot = np.asarray(sim.data.geom_xmat[g]).reshape(3, 3)
        center = np.asarray(sim.data.geom_xpos[g]) + rot @ geom[:3]
        half = np.abs(rot) @ geom[3:]
        lows.append(center - half)
        highs.append(center + half)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def fingerprint(f):
    return {k: f[k] for k in ("step", "raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256")}


def opening_window(frames, start):
    if start is None:
        return None
    for left, right in zip(frames, frames[1:]):
        step = right["step"]
        x, y, z = right["audit"]["eef_xyz"]
        if (start < step <= start + 15 and .031 < left["audit"]["gap"] < .072
                and right["audit"]["gap"] >= .067
                and any(a[-1] == -1 for a in right["actions"])
                and -.105 < x < .11 and .15 < y < .37 and .50 < z < .83):
            return step
    return None


def gates(case, carried, initial, step_rows, frames, stages):
    commanded = stages.get("open")
    opening = opening_window(frames, commanded)
    by_step = {row["step"]: row for row in step_rows}
    pre = by_step.get(commanded)
    after = [by_step[i] for i in range(opening, opening + FOLLOW + 1) if i in by_step] if opening else []
    tail = after[-1] if after else None
    full_follow = len(after) == FOLLOW + 1
    complete_goals = full_follow and all(not a["goals"][SOUP] for a in after)
    free_after = bool(tail and not tail["held"][carried])
    release_held = bool(pre and pre["held"][carried])
    moved = bool(tail and sum((x - y)**2 for x, y in
                              zip(tail["positions"][carried], initial["positions"][carried])) >= .05**2)
    stable_rim = bool(full_follow and all(a["rim_contact"] and not a["held"][SOUP] for a in after[-5:]))
    touched_rim = any(a["rim_contact"] and not a["held"][SOUP] for a in after)
    qualifies = {"held_at_release_command": release_held, "measured_near_open": opening is not None,
                 "complete_30step_window": full_follow, "soup_false_through_window": complete_goals,
                 "released_by_end": free_after, "carried_at_least_5cm": moved}
    if case == "outside":
        dx, dy = (tail["positions"][SOUP][i] - tail["positions"]["basket_1"][i]
                  for i in (0, 1)) if tail else (0., 0.)
        qualifies["outside_xy"] = abs(dx) > .070 or abs(dy) > .085
        qualifies["rim_did_not_catch"] = not stable_rim
        qualifies["other_can_not_in"] = full_follow and not any(a["goals"][SAUCE] for a in after)
    elif case == "rim":
        qualifies["unheld_rim_contact_last_5"] = stable_rim
        qualifies["other_can_not_in"] = full_follow and not any(a["goals"][SAUCE] for a in after)
    elif case == "other_can":
        qualifies["sauce_in_final_20"] = full_follow and all(
            a["goals"][SAUCE] and not a["held"][SAUCE] for a in after[-20:])
        qualifies["soup_never_held"] = not any(a["held"][SOUP] for a in step_rows)
    else:
        qualifies["soup_never_held"] = not any(a["held"][SOUP] for a in step_rows)
        qualifies["other_can_not_in"] = full_follow and not any(a["goals"][SAUCE] for a in after)
        qualifies["cheese_in_final_20"] = full_follow and all(
            a["goals"][CHEESE] and not a["held"][CHEESE] for a in after[-20:])
    return {"eligible": all(qualifies.values()), "checks": qualifies,
            "open_command_after_step": commanded, "measured_open_step": opening,
            "rim_contact_in_window": touched_rim,
            "rim_contact_class": ("supported_at_30" if stable_rim else
                                  "touch_then_no_support" if touched_rim else "none"),
            "unheld_rim_contact_steps": [a["step"] for a in after if a["rim_contact"] and not a["held"][SOUP]]}


def capture_case(job, bddl, out, asset, np, Rotation, Simulator):
    case, slot = job["case"], job["slot"]
    carried = {"outside": SOUP, "rim": SOUP, "other_can": SAUCE, "identity_unknown": CHEESE}[case]
    folder = out / str(job["layout"])
    (folder / "frames").mkdir(parents=True, exist_ok=False)
    (folder / "trace").mkdir()
    scene = Simulator(bddl, folder / "trace", job["layout"], False)
    try:
        env = scene.env.env
        sim = env.sim
        robot = env.robots[0]
        site = robot.controller.eef_name
        root = env.obj_body_id
        basket_geom = subtree_geoms(sim.model, root["basket_1"])
        soup_geom = subtree_geoms(sim.model, root[SOUP])
        carried_geom = subtree_geoms(sim.model, root[carried])
        basket_low, basket_high = bounds(sim, basket_geom, np)
        stages, frames, step_rows, block, transitions = {}, [], [], [], []
        previous = scene.goals()

        def audit():
            sample = scene.xyz()
            rim_hits = []
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]
                pair = {c.geom1, c.geom2}
                if pair & basket_geom and pair & soup_geom and c.dist <= .001 and c.pos[2] >= basket_high[2] - .019:
                    rim_hits.append({"z_m": float(c.pos[2]), "separation_m": float(c.dist)})
            return {"step": scene.frame, "goals": scene.goals(), "positions": sample,
                    "held": {k: bool(env._check_grasp(robot.gripper, env.objects_dict[k]))
                             for k in (SOUP, SAUCE, CHEESE)},
                    "eef_xyz": scene.obs["robot0_eef_pos"].tolist(),
                    "gap": float(np.abs(scene.obs["robot0_gripper_qpos"]).sum()),
                    "rim_contact": bool(rim_hits), "rim_hits": rim_hits}

        def record(stream):
            a = step_rows[-1]
            observed = scene.capture(executed=len(block), done=False, transitions=list(transitions),
                                     completed_regrasp_steps=0)
            (folder / "trace" / "observation.npz").replace(folder / "frames" / f"{scene.frame}.npz")
            observed["audit"] |= {"gap": a["gap"], "eef_xyz": a["eef_xyz"]}
            row = {"start_step": scene.frame - len(block), "target": None,
                   "instruction": "physical release challenge", "actions": block[:], **observed}
            frames.append(row)
            if block:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            block.clear()
            transitions.clear()

        def drive(stage, target, grip, count, rotation, stream):
            nonlocal previous
            stages[stage] = scene.frame
            for _ in range(count):
                if all(previous.values()):
                    break
                dpos = 1.4 * (target - sim.data.get_site_xpos(site)) / .05
                native_rot = np.asarray(sim.data.get_site_xmat(site)).reshape(3, 3)
                drot = Rotation.from_matrix(rotation @ native_rot.T).as_rotvec() / .5
                executed = np.r_[np.clip(np.r_[dpos, drot], -.85, .85), grip].astype(np.float32)
                scene.obs, _, _, _ = scene.env.step(executed)
                scene.frame += 1
                a = audit()
                step_rows.append(a)
                transitions.extend({"step": scene.frame, "object": key, "in_basket": bool(value)}
                                   for key, value in a["goals"].items() if value != previous[key])
                previous = a["goals"]
                block.append(executed.tolist())
                if scene.frame % STRIDE == 0:
                    record(stream)

        initial = audit()
        step_rows.append(initial)
        basket = np.asarray(initial["positions"]["basket_1"])
        center = np.asarray(initial["positions"][carried])
        safe = np.asarray(sim.data.get_site_xpos(site)).copy()
        safe[2] = max(.74, safe[2])
        rotation = np.asarray(asset["base_rotation_matrix"])
        grab = center + asset["grasp_xyz_offset_m"]
        if carried == CHEESE:
            low, high = bounds(sim, carried_geom, np)
            grab = (low + high) * .5 + [0, 0, .007]
            # First vary opposite basket sides, then choose the orthogonal jaw approach.
            if (high - low)[0] > (high - low)[1]:
                rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ rotation
        high = grab + [0, 0, .16]
        sign = -1 if (job["layout"] % 2) else 1
        with (folder / "trace" / "actions.jsonl").open("x", encoding="utf-8") as stream:
            record(stream)
            for name, target, grip, count in (("safe", safe, -1, 16), ("above_grasp", high, -1, 33),
                                              ("approach", grab, -1, 44), ("grasp", grab, 1, 24)):
                drive(name, target, grip, count, rotation, stream)
            if step_rows[-1]["held"][carried]:
                drive("lift", high, 1, 29, rotation, stream)
            if step_rows[-1]["held"][carried]:
                over = np.array([basket[0] + sign * .022, basket[1] - .025, .67])
                lower = over.copy()
                lower[2] = .555
                if case == "outside":
                    over[1] = lower[1] = basket[1] - .096
                    lower[2] = .60
                elif case == "rim":
                    # The collision shape only sets the waypoint; real contacts
                    # and released body dynamics decide eligibility later.
                    object_center = np.asarray(step_rows[-1]["positions"][SOUP])
                    object_low, _ = bounds(sim, carried_geom, np)
                    tcp_minus_center = np.asarray(sim.data.get_site_xpos(site)) - object_center
                    lip = (basket_low[0] if sign < 0 else basket_high[0]) - sign * (.006 + .004 * (slot // 2))
                    target_center = np.array([lip, basket[1] - .015,
                                              basket_high[2] + (object_center[2] - object_low[2]) + .009])
                    lower = target_center + tcp_minus_center
                    over = lower.copy()
                    over[2] = max(.67, lower[2] + .06)
                drive("transport", over, 1, 44, rotation, stream)
                drive("lower", lower, 1, 28, rotation, stream)
                drive("open", lower, -1, 20, rotation, stream)
                drive("retract", over, -1, 36, rotation, stream)
            if block:
                record(stream)
        quality = gates(case, carried, initial, step_rows, frames, stages)
        write(folder / "physics-audit.json", {"controlled_object": carried, "stages": stages,
                                             "basket_collision_box": [basket_low.tolist(), basket_high.tolist()],
                                             "steps": step_rows, "release_quality": quality})
        write(folder / "fixture.json", {**job, "initial": fingerprint(frames[0]),
                                      "end": fingerprint(frames[-1]), "initial_audit": frames[0]["audit"],
                                      "end_audit": frames[-1]["audit"], "frames": "frames",
                                      "actions": "trace/actions.jsonl", "physics_audit": "physics-audit.json",
                                      "frame_steps": [r["step"] for r in frames], "release_quality": quality})
        return {"path": f'{job["layout"]}/fixture.json', "initial_sha256": frames[0]["state_sha256"],
                "layout": job["layout"], "case": case, **quality}
    finally:
        scene.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("card", type=int, choices=(2, 3))
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--phase", choices=("dev", "blind"), required=True)
    p.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    p.add_argument("--list", action="store_true", help="List the frozen schedule without loading physics")
    a = p.parse_args()
    jobs = [job for job in schedule(a.phase) if job["card"] == a.card and job["case"] in a.cases]
    if a.list:
        print(json.dumps(jobs))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU[a.card] or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(a.card):
        raise RuntimeError("EGL and CUDA must name the same approved, idle GPU")
    study, out = a.study.resolve(), a.out.resolve()
    lock, plan = read(study / "frozen.json"), read(study / "plan.json")
    bddl = study / plan["bddl"]
    if (sha(study / "plan.json") != lock["plan_sha256"] or sha(bddl) != plan["bddl_sha256"]
            or not all(sha(ROOT / name) == lock["code"][name] for name in ("common.py", "simulator.py"))):
        raise RuntimeError("Frozen scenario or simulator has changed")
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Unexpected repository revision")
    import numpy as np
    from scipy.spatial.transform import Rotation
    from simulator import Simulator, ensure_libero_config
    ensure_libero_config()
    asset = V1 / "grasp-reference.json"
    ref = read(asset)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"manifest-{a.card}.json"
    if path.exists() or any((out / str(job["layout"])).exists() for job in jobs):
        raise FileExistsError("Collection already exists; choose a fresh output root")
    manifest = {"schema": "physical-release-v1", "card": a.card, "phase": a.phase, "planned": jobs,
                "complete": False, "records": [], "collector_sha256": sha(Path(__file__)),
                "bddl_sha256": sha(bddl), "grasp_reference_sha256": sha(asset), "frame_stride": STRIDE,
                "followup_steps": FOLLOW, "commit": commit}
    write(path, manifest)
    for job in jobs:
        result = capture_case(job, bddl, out, ref, np, Rotation, Simulator)
        manifest["records"].append(result)
        write(path, manifest)
        print(job["layout"], job["case"], "eligible", result["eligible"],
              [k for k, v in result["checks"].items() if not v], flush=True)
    manifest["eligible_by_case"] = dict(Counter(r["case"] for r in manifest["records"] if r["eligible"]))
    manifest["complete"] = True
    write(path, manifest)


if __name__ == "__main__":
    main()
