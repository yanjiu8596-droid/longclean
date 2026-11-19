"""Evaluator-only placement feasibility from one sealed native soup grasp."""

import argparse
import inspect
import json
from math import dist, hypot
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import BASKET, BUDGET, CHEESE, GOALS, SAUCE, SOUP, array_hash, read, sha, write


HERE = Path(__file__).resolve().parent
AUDIT_SHA = "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4"
SOURCE_REPORT_SHA = "72f8e63eeca8774b836778723558aad7c2c04e93520e1420fd0265e1fbe46d4c"
PHYSICS_SHA = "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8"
LAYOUT, FORK, STABLE = 931301, 400, 20
OFFSETS = ((0., 0.), (.03, 0.), (-.03, 0.), (0., .03), (0., -.03))
PHASE_CAPS = {"lift": 30, "transfer": 45, "lower": 135, "open": 20, "retreat": 40, "settle": 40}
XY_TOL, Z_TOL, SETTLED_TOL = .005, .010, .005
PLACED = (CHEESE, SAUCE)


def read_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def pinned(path, digest):
    require(path.is_file() and sha(path) == digest, f"Changed source artifact: {path}")
    return path


def fingerprint(state):
    return {k: state[k] for k in ("step", "raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256")}


def source_case(run, audit, report_path):
    a = read(pinned(audit, AUDIT_SHA))
    require(a["schema"] == "v2-dense-online-audit/1" and a["valid_native_terminal_records"] == 12,
            "Unexpected or incomplete source audit")
    entry, = [r for r in a["records"] if r["layout_seed"] == LAYOUT and r["arm"] == "adaptive1"]
    result_path = (run / entry["result"]).resolve()
    require(result_path.is_relative_to(run), "Source episode is outside its result root")
    result = read(pinned(result_path, entry["result_sha256"]))
    report = read(pinned(report_path, SOURCE_REPORT_SHA))
    require(report["layout_seed"] == LAYOUT and report["source_arm"] == "adaptive1"
            and report["source_audit_sha256"] == AUDIT_SHA and report["source_result_sha256"] == entry["result_sha256"]
            and report["source_actions_sha256"] == result["actions_sha256"] == entry["actions_sha256"]
            and report["source_grasp_audit_sha256"] == result["grasp_audit_sha256"] == entry["grasp_audit_sha256"]
            and report["initial"] == result["initial"], "Contact replay and original episode disagree")
    originals = [r for r in read_lines(pinned(result_path.parent / "actions.jsonl", entry["actions_sha256"]))
                 if r["step"] <= FORK]
    grasp = [r for r in read_lines(pinned(result_path.parent / "grasp-audit.jsonl", entry["grasp_audit_sha256"]))
             if r["step"] <= FORK]
    anchors = [r for r in read_lines(pinned(report_path.parent / "actions.jsonl", report["actions_sha256"]))
               if r["step"] <= FORK]
    sidecar = read_lines(pinned(report_path.parent / "contacts.jsonl", report["contacts_sha256"]))
    at_fork, = [r for r in sidecar if r["step"] == FORK]
    require(originals and originals[-1]["step"] == FORK and originals == anchors
            and [r["step"] for r in grasp] == [0, *[r["step"] for r in originals]]
            and grasp[-1]["state_sha256"] == at_fork["state_sha256"] == originals[-1]["state_sha256"],
            "The exact original replay lacks a matching step-400 fork")
    require(grasp[-1]["held_by_target"] == {obj: obj == SOUP for obj in GOALS}
            and originals[-1]["audit"]["goals"] == at_fork["goals"] == {obj: obj != SOUP for obj in GOALS},
            "The selected source fork is not the known soup grasp with two completed objects")
    return {"entry": entry, "initial": result["initial"], "actions": originals, "grasp": grasp, "contact_fork": at_fork}


def site_formula(pos, rotation, size, point):
    """Reproduce the installed SiteObject.in_box calculation and strict bounds."""
    import numpy as np

    pos, point = np.asarray(pos), np.asarray(point)
    extent = np.abs(np.asarray(rotation) @ np.asarray(size))
    low, high = pos - extent, pos + extent
    low[2] -= .01
    return {"total_size": extent.tolist(), "lower": low.tolist(), "upper": high.tolist(),
            "lower_margin_m": (point - low).tolist(), "upper_margin_m": (high - point).tolist(),
            "strictly_inside": bool(np.all(point > low) and np.all(point < high))}


def site_snapshot(scene, goals):
    import numpy as np

    inner = scene.env.env
    region = inner.object_states_dict[BASKET]
    actual_site = inner.object_sites_dict[region.object_name]
    require(type(region).__name__ == "SiteObjectState" and region.object_name == BASKET,
            "Expected the native LIBERO basket site state")
    pos = inner.sim.data.get_site_xpos(region.object_name).tolist()
    rotation = inner.sim.data.get_site_xmat(region.object_name).reshape(3, 3).tolist()
    size = np.asarray(actual_site.size).tolist()
    site_id = inner.sim.model.site_name2id(region.object_name)
    site_body, basket_body = int(inner.sim.model.site_bodyid[site_id]), int(inner.obj_body_id["basket_1"])
    objects = {}
    for obj in GOALS:
        state = inner.object_states_dict[obj]
        point = inner.sim.data.body_xpos[inner.obj_body_id[state.object_name]].tolist()
        formula = site_formula(pos, rotation, size, point)
        inside = bool(region.check_contain(state))
        contact_result = bool(region.check_contact(state))
        require(contact_result and inside == formula["strictly_inside"] == goals[obj],
                f"Copied strict in_box semantics differ from native predicate for {obj} at {scene.frame}")
        objects[obj] = {"body_center_m": point, "library_check_contain": inside, "library_check_contact": contact_result,
                        "official_in_predicate": goals[obj], **formula}
    states_source = Path(inspect.getfile(type(region))).resolve()
    object_source = Path(inspect.getfile(type(actual_site))).resolve()
    predicates_source = states_source.parents[1] / "predicates" / "base_predicates.py"
    return {"name": region.object_name, "parent_name": region.parent_name, "state_type": type(region).__name__,
            "site_object_type": type(actual_site).__name__, "position_m": pos, "rotation": rotation,
            "size": size, "site_id": site_id, "site_body_id": site_body, "basket_body_id": basket_body,
            "basket_body_position_m": inner.sim.data.body_xpos[basket_body].tolist(),
            "basket_body_rotation": inner.sim.data.body_xmat[basket_body].reshape(3, 3).tolist(),
            "objects": objects, "sources": {key: {"path": str(path), "sha256": sha(path)}
                for key, path in (("object_states", states_source), ("site_object", object_source), ("predicates", predicates_source))}}


def target_error(body, target):
    x, y, z = (float(value) - float(wanted) for value, wanted in zip(body, target, strict=True))
    xy = hypot(x, y)
    return {"xyz_error_m": [x, y, z], "xy_error_m": xy, "abs_z_error_m": abs(z),
            "within_tolerance": xy <= XY_TOL and abs(z) <= Z_TOL}


def physical_sample(scene):
    inner = scene.env.env
    robot = inner.robots[0]
    sim = inner.sim
    before = array_hash(scene.env.get_sim_state())
    held = {name: bool(inner._check_grasp(robot.gripper, inner.objects_dict[name])) for name in GOALS}
    record = {"step": scene.frame, "state_sha256": before, "goals": scene.goals(), "held": held,
              "body_poses": {name: {"position_m": sim.data.body_xpos[inner.obj_body_id[name]].tolist(),
                                    "rotation": sim.data.body_xmat[inner.obj_body_id[name]].reshape(3, 3).tolist()}
                             for name in GOALS},
              "site_xyz_m": sim.data.get_site_xpos(robot.controller.eef_name).tolist(),
              "site_rotation": sim.data.get_site_xmat(robot.controller.eef_name).reshape(3, 3).tolist(),
              "measured_grip_gap_m": sum(abs(x) for x in scene.obs["robot0_gripper_qpos"].tolist())}
    require(array_hash(scene.env.get_sim_state()) == before, "Evaluator sampling changed native state")
    return record


def settled_outcome(history, target, opening_error, completed_phases):
    require(bool(history), "The fork sample is required")
    detach_pairs = [(left, r) for left, r in zip(history, history[1:]) if left["held"][SOUP] and not r["held"][SOUP]]
    first = detach_pairs[0] if detach_pairs else None
    commanded = next((pair for pair in detach_pairs if pair[1].get("phase") in {"open", "retreat", "settle"}), None)
    detach = first[1]["step"] if first else None
    opened_detach = commanded[1]["step"] if commanded else None
    detach_error = None if first is None else {key: {"step": row["step"],
        **target_error(row["body_poses"][SOUP]["position_m"], target)}
        for key, row in zip(("last_held", "first_unheld"), first, strict=True)}
    tail = history[-STABLE:]
    poses = [r["body_poses"][SOUP]["position_m"] for r in tail]
    tail_errors = [target_error(point, target) for point in poses]
    tail_worst = {"max_xy_error_m": max(e["xy_error_m"] for e in tail_errors),
                  "max_abs_z_error_m": max(e["abs_z_error_m"] for e in tail_errors)}
    spread = max(dist(a, b) for a in poses for b in poses)
    settled = (detach is not None and len(tail) == STABLE and tail[0]["step"] >= detach
               and all(all(r["goals"].values()) and not any(r["held"].values()) for r in tail) and spread <= SETTLED_TOL)
    reversals = [{"step": r["step"], "object": obj} for left, r in zip(history, history[1:])
                 for obj in PLACED if left["goals"][obj] and not r["goals"][obj]]
    displacement = {obj: max(dist(r["body_poses"][obj]["position_m"], history[0]["body_poses"][obj]["position_m"])
                              for r in history) for obj in PLACED}
    achieved = bool(opening_error and opening_error["within_tolerance"] and detach_error and detach == opened_detach
                    and all(e["within_tolerance"] for e in detach_error.values()))
    maintained = len(tail) == STABLE and all(e["within_tolerance"] for e in tail_errors)
    return {"first_held_to_unheld_step": detach, "first_held_to_unheld_after_open_command_step": opened_detach,
            "target_error_at_first_detach": detach_error, "tail_max_target_error": tail_worst,
            "tail_steps": [r["step"] for r in tail], "tail_soup_max_pairwise_displacement_m": spread,
            "physical_completion": settled, "specified_target_achieved_at_controlled_release": achieved,
            "specified_target_maintained_in_tail": maintained,
            "strict_success": settled and achieved and maintained and completed_phases and not reversals,
            "completed_object_goal_reversals": reversals, "completed_object_max_displacement_m": displacement}


def replay_fork(scene, source, dest):
    initial = scene.capture()
    require(fingerprint(initial) == source["initial"], "Source initial state diverged")
    with (dest / "prefix-actions.jsonl").open("x", encoding="utf-8") as stream:
        for original in source["actions"]:
            require(scene.frame == original["start_step"], "Original IPC boundary differs")
            current = scene.advance(original["actions"])
            actual = {"start_step": original["start_step"], "target": original["target"],
                      "instruction": original["instruction"], "actions": original["actions"][:current["executed"]], **current}
            require(actual == original, f"Source native response differs at {original['step']}")
            stream.write(json.dumps(actual, allow_nan=False) + "\n")
    require(read_lines(dest / "grasp-audit.jsonl") == source["grasp"], "Original all-object grasp sidecar differs")
    fork = physical_sample(scene)
    require(scene.frame == FORK and fork["state_sha256"] == current["state_sha256"], "Incorrect common fork")
    require(fork["held"] == {obj: obj == SOUP for obj in GOALS}
            and fork["goals"] == {obj: obj != SOUP for obj in GOALS}, "Incorrect grasp or completed objects at fork")
    contact = source["contact_fork"]
    require(all(pose["position_m"] == contact["body_poses"][name]["position_m"]
                and pose["rotation"] == contact["body_poses"][name]["rotation_matrix"]
                for name, pose in fork["body_poses"].items()), "Original contact-side body poses differ at fork")
    return current, fork


def archive_frame(scene, dest, sample, captured):
    step = scene.frame
    path = dest / "frames" / f"{step}.npz"
    copyfile(dest / "observation.npz", path)
    require(captured["state_sha256"] == sample["state_sha256"] and captured["audit"]["goals"] == sample["goals"],
            f"Frame refused provenance join at {step}")
    region = site_snapshot(scene, sample["goals"])
    require(array_hash(scene.env.get_sim_state()) == sample["state_sha256"], "Region queries changed simulator state")
    return {**fingerprint(captured), "file": path.relative_to(dest).as_posix(), "sha256": sha(path), "region": region}


def osc_control(current, waypoint, rotation, grip, slow):
    import numpy as np
    from scipy.spatial.transform import Rotation

    delta = 1.4 * (np.asarray(waypoint) - current["site_xyz_m"]) / .05
    angle = Rotation.from_matrix(np.asarray(rotation) @ np.asarray(current["site_rotation"]).T).as_rotvec() / .5
    control = np.clip(np.r_[delta, angle], -.85, .85)
    if slow:
        control[2] = max(-.03, control[2])
    return np.r_[control, grip].tolist()


def record_preview(snapshot, dest):
    import numpy as np
    from PIL import Image

    with np.load(dest / snapshot["file"], allow_pickle=False) as data:
        rgb = Image.fromarray(np.concatenate([data[k][::-1, ::-1] for k in ("camera1", "camera2")], axis=1))
    preview = dest / "final.png"
    rgb.save(preview)
    return {"file": preview.name, "sha256": sha(preview)}


def run_case(simulator, bddl, source, slot, dest):
    import numpy as np

    dest.mkdir(parents=True, exist_ok=False)
    (dest / "frames").mkdir()
    scene = simulator(bddl, dest, LAYOUT, False)
    try:
        original, fork = replay_fork(scene, source, dest)
        frames = [archive_frame(scene, dest, fork, original)]
        region = frames[0]["region"]
        center = np.asarray(region["position_m"])
        delta = (*OFFSETS[slot], 0.)
        target = center + delta
        soup = np.asarray(fork["body_poses"][SOUP]["position_m"])
        offset = np.asarray(fork["site_xyz_m"]) - soup
        rotation = fork["site_rotation"]
        high_z = max(.70, soup[2])
        lift, over = np.r_[soup[:2], high_z], np.r_[target[:2], high_z]
        history, stages = [fork], []
        first_official = None
        latest_capture = None
        error_at_open = None
        primary_stop = None
        lowered_before_loss = None
        executed_settle = False

        def preserve_current(record, done):
            nonlocal latest_capture
            if latest_capture is None or latest_capture["step"] != record["step"]:
                latest_capture = scene.capture(executed=1, done=bool(done))
            if frames[-1]["step"] != record["step"]:
                frames.append(archive_frame(scene, dest, record, latest_capture))
            return latest_capture

        def advance(phase, action, stream):
            nonlocal first_official, primary_stop
            require(scene.frame < FORK + sum(PHASE_CAPS.values()) <= BUDGET, "Controlled action exceeds the fixed budget")
            realized = np.asarray(action, dtype=np.float32)
            try:
                scene.obs, _, done, _ = scene.env.step(realized)
            except ValueError as error:
                if first_official is not None and "terminated episode" in str(error).lower():
                    primary_stop = "native_environment_refused_diagnostic_tail"
                    return None
                raise
            scene.frame += 1
            record = physical_sample(scene)
            official = all(record["goals"].values())
            record.update(phase=phase, action=realized.tolist(), environment_done=bool(done),
                          past_first_official_goal=first_official is not None)
            history.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            if official and first_official is None:
                capture = preserve_current(record, done)
                first_official = {"step": scene.frame, "held": record["held"], "environment_done": bool(done),
                                  "fingerprint": fingerprint(capture), "phase": phase, "frame": frames[-1]["file"]}
            if scene.frame % 5 == 0:
                preserve_current(record, done)
            if done and not official:
                primary_stop = "environment_done_without_all_goals"
            return record

        current = fork
        with (dest / "controlled-steps.jsonl").open("x", encoding="utf-8") as stream:
            for phase, body_waypoint in (("lift", lift), ("transfer", over), ("lower", target)):
                start = scene.frame
                reached = target_error(current["body_poses"][SOUP]["position_m"], body_waypoint)["within_tolerance"]
                for _ in range(PHASE_CAPS[phase]):
                    if reached or primary_stop:
                        break
                    moved = advance(phase, osc_control(current, body_waypoint + offset, rotation, 1, phase == "lower"), stream)
                    if moved is None:
                        break
                    current = moved
                    reached = target_error(current["body_poses"][SOUP]["position_m"], body_waypoint)["within_tolerance"]
                    if not current["held"][SOUP]:
                        if phase == "lower":
                            lowered_before_loss = scene.frame
                        else:
                            primary_stop = f"soup_grasp_lost_during_{phase}"
                        break
                error = target_error(current["body_poses"][SOUP]["position_m"], body_waypoint)
                stages.append({"phase": phase, "start_step": start, "end_step": scene.frame,
                               "body_target_m": body_waypoint.tolist(), "error_at_end": error, "grasp_at_end": current["held"][SOUP]})
                if not primary_stop and phase != "lower" and not reached:
                    primary_stop = f"{phase}_target_not_reached"
                if primary_stop:
                    break

            if not primary_stop:
                error_at_open = target_error(current["body_poses"][SOUP]["position_m"], target)
                open_at = np.asarray(current["site_xyz_m"])
                retreat_at = open_at.copy()
                retreat_at[2] = max(open_at[2] + .09, high_z + offset[2])
                for phase, waypoint in (("open", open_at), ("retreat", retreat_at), ("settle", None)):
                    start = scene.frame
                    for _ in range(PHASE_CAPS[phase]):
                        if primary_stop:
                            break
                        action = [0.] * 6 + [-1.] if waypoint is None else osc_control(current, waypoint, rotation, -1, False)
                        moved = advance(phase, action, stream)
                        if moved is None:
                            break
                        current = moved
                    stages.append({"phase": phase, "start_step": start, "end_step": scene.frame,
                                   "site_target_m": None if waypoint is None else waypoint.tolist(),
                                   "grasp_at_end": current["held"][SOUP]})
                executed_settle = stages[-1]["end_step"] - stages[-1]["start_step"] == PHASE_CAPS["settle"]
        if current["step"] != FORK:
            preserve_current(current, current["environment_done"])
        final_preview = record_preview(frames[-1], dest)
        outcome = settled_outcome(history, target, error_at_open, executed_settle and lowered_before_loss is None and not primary_stop)
        diagnostic_status = ("censored_native_terminal" if primary_stop == "native_environment_refused_diagnostic_tail"
                             else "phase_stopped" if primary_stop else "complete")
        report = {"schema": "v2-soup-placement-feasibility-case/1", "layout_seed": LAYOUT, "slot": slot,
                  "fork": frames[0], "fork_physics": fork, "offset_xy_m": list(OFFSETS[slot]), "body_target_m": target.tolist(),
                  "shared_site_minus_soup_translation_m": offset.tolist(), "shared_site_rotation": rotation,
                  "target_error_before_open": error_at_open, "target_error_at_end": target_error(current["body_poses"][SOUP]["position_m"], target),
                  "original_ipc_rows_exact": len(source["actions"]), "descent_grasp_lost_step": lowered_before_loss,
                  "first_official_all_goals": first_official, "diagnostic_post_official_control_count": sum(r.get("past_first_official_goal", False) for r in history),
                  "controlled_steps_executed": scene.frame - FORK, "phase_results": stages, "diagnostic_status": diagnostic_status,
                  "stop_reason": primary_stop or "finished_fixed_diagnostic", "final": frames[-1], "final_physics": current,
                  "frames": frames, "final_preview": final_preview, "original_replay_sha256": sha(dest / "prefix-actions.jsonl"),
                  "grasp_capture_sha256": sha(dest / "grasp-audit.jsonl"), "controlled_steps_sha256": sha(dest / "controlled-steps.jsonl"),
                  "scope": "Oracle diagnostic continuation from a seen physical checkpoint; original rollout results are unchanged.", **outcome}
        path = dest / "report.json"
        write(path, report)
        return path, report
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True, choices=GPU, type=int)
    for name in ("study", "run", "audit", "source-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    a = parser.parse_args()
    run, out, prior = a.run.resolve(), a.output.resolve(), a.source_report.resolve()
    require(not out.is_relative_to(run) and not prior.is_relative_to(out), "Use a separate diagnostic output")
    require(FORK + sum(PHASE_CAPS.values()) == 710 <= BUDGET, "Unexpected diagnostic budget")
    validate_gpu(a.card)
    bddl, _ = validate_study(a.study.resolve())
    source = source_case(run, a.audit.resolve(), prior)
    require(source["initial"]["bddl_sha256"] == sha(bddl), "Source scene differs")
    pinned(HERE / "actual_physics_simulator.py", PHYSICS_SHA)
    import actual_physics_simulator as physics

    native = physics.instrument(physics.load_native())
    native.ensure_libero_config()
    slots = [i for i in range(len(OFFSETS)) if i % 2 == a.card - 2]
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"soup-placement-plan-{a.card}.json"
    manifest_file = out / f"soup-placement-{a.card}.json"
    require(not plan_file.exists() and not manifest_file.exists()
            and all(not (out / f"slot-{i}").exists() for i in slots), "This assigned attempt has already started")
    plan = {"schema": "v2-soup-placement-feasibility-plan/1", "card": a.card, "source_layout": LAYOUT,
            "source_arm": "adaptive1", "source_audit_sha256": AUDIT_SHA, "source_contact_report_sha256": SOURCE_REPORT_SHA,
            "source_episode_sha256": source["entry"]["result_sha256"], "source_actions_sha256": source["entry"]["actions_sha256"],
            "source_grasp_sha256": source["entry"]["grasp_audit_sha256"], "source_bddl_sha256": sha(bddl),
            "physics_wrapper_sha256": PHYSICS_SHA, "runner_sha256": sha(Path(__file__)),
            "fork": fingerprint(source["actions"][-1]), "phase_caps": PHASE_CAPS, "maximum_diagnostic_step": 710,
            "native_total_budget": BUDGET, "stable_tail_steps": STABLE,
            "stable_position_tolerance_m": SETTLED_TOL, "opening_xy_tolerance_m": XY_TOL, "opening_z_tolerance_m": Z_TOL,
            "planned": [{"slot": i, "offset_xy_m": list(OFFSETS[i])} for i in slots],
            "scope": "Evaluator-only feasible placement test on one known state under unchanged LIBERO physics."}
    write(plan_file, plan)
    manifest = {"schema": "v2-soup-placement-feasibility-manifest/1", "card": a.card,
                "plan_sha256": sha(plan_file), "complete": False, "cases": []}
    write(manifest_file, manifest)
    for slot in slots:
        path, r = run_case(native.Simulator, bddl, source, slot, out / f"slot-{slot}")
        manifest["cases"].append({"slot": slot, "report": path.relative_to(out).as_posix(), "report_sha256": sha(path),
                                  "physical_completion": r["physical_completion"], "strict_success": r["strict_success"],
                                  "stop_reason": r["stop_reason"], "diagnostic_status": r["diagnostic_status"],
                                  "controlled_steps_executed": r["controlled_steps_executed"]})
        write(manifest_file, manifest)
        print("SOUP_PLACEMENT", slot, r["physical_completion"], r["strict_success"], r["stop_reason"], flush=True)
    manifest["complete"] = True
    write(manifest_file, manifest)


if __name__ == "__main__":
    main()
