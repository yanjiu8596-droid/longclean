"""Check the five fixed placement diagnostics against their native evidence."""
import argparse
from itertools import combinations
from math import dist, hypot, isfinite
from pathlib import Path

import numpy as np

from soup_placement_feasibility import (AUDIT_SHA, BUDGET, FORK, GOALS, HERE, LAYOUT, OFFSETS, PHASE_CAPS,
                                        PHYSICS_SHA, PLACED, SETTLED_TOL, SOUP, SOURCE_REPORT_SHA, STABLE,
                                        XY_TOL, Z_TOL, read_lines, source_case)
from stage4_fixture import require
from target_follow_audit import checked_path, observation, read, sha, stamp, write


CARD_SLOTS = {2: (0, 2, 4), 3: (1, 3)}
ORIGINAL_RUNNER_SHA = "731a621f42945ca39ca2057f5690b4152eef04361725c2c0ddecab5dba006951"
REPAIRED_RUNNER_SHA = "d8ef0f1f5df2525cdbb152ea1b62644bb533a6bd2d453e98e65e58c5d979fe7e"
PHASES = tuple(PHASE_CAPS)
OPEN_PHASES = {"open", "retreat", "settle"}
STOPPED = {"soup_grasp_lost_during_lift", "soup_grasp_lost_during_transfer",
           "lift_target_not_reached", "transfer_target_not_reached", "environment_done_without_all_goals"}


def coordinates(row, obj=SOUP):
    return row["body_poses"][obj]["position_m"]


def point_error(row, target):
    delta = [a - b for a, b in zip(coordinates(row), target, strict=True)]
    radial, height = hypot(*delta[:2]), abs(delta[2])
    return {"xyz_error_m": delta, "xy_error_m": radial, "abs_z_error_m": height,
            "within_tolerance": radial <= XY_TOL and height <= Z_TOL}


def check_phases(history, stages, target, stop):
    at = {r["step"]: r for r in history}
    end = history[-1]["step"]
    initial = history[0]
    soup = coordinates(initial)
    high = max(.70, soup[2])
    site_offset_z = initial["site_xyz_m"][2] - soup[2]
    waypoints = {"lift": [*soup[:2], high], "transfer": [*target[:2], high], "lower": target}
    names = [stage["phase"] for stage in stages]
    require(bool(names) and names == list(PHASES[:len(names)]), "Changed diagnostic phase order")
    opening, lost_on_descent, settled_steps = None, None, 0
    start = FORK
    for stage in stages:
        phase, finish = stage["phase"], stage["end_step"]
        require(stage["start_step"] == start and 0 <= finish - start <= PHASE_CAPS[phase]
                and finish in at and stage["grasp_at_end"] is at[finish]["held"][SOUP],
                f"Invalid phase boundaries or held state for {phase}")
        if phase in waypoints:
            expected = waypoints[phase]
            require(stage["body_target_m"] == expected and stage["error_at_end"] == point_error(at[finish], expected),
                    f"Wrong body waypoint or measured endpoint in {phase}")
        for step in range(start + 1, finish + 1):
            row = at[step]
            a = row["action"]
            cap = float(np.float32(.85))
            require(row["phase"] == phase and len(a) == 7
                    and all(type(v) in (int, float) and isfinite(v) and float(np.float32(v)) == v for v in a)
                    and all(-cap <= v <= cap for v in a[:6])
                    and a[-1] == (-1. if phase in OPEN_PHASES else 1.)
                    and (phase != "lower" or a[2] >= float(np.float32(-.03)))
                    and (phase != "settle" or a[:6] == [0.] * 6),
                    f"Invalid native float32 OSC command at {step}")
        if phase == "lower":
            lost_on_descent = next((step for step in range(start + 1, finish + 1) if not at[step]["held"][SOUP]), None)
            if lost_on_descent is not None:
                require(lost_on_descent == finish, "Descent continued despite losing the grasp")
        if phase == "open":
            opening = point_error(at[start], target)
            open_at = at[start]["site_xyz_m"]
            require(stage["site_target_m"] == open_at, "Opening did not hold the achieved end-effector position")
        if phase == "retreat":
            retreat_at = [*open_at[:2], max(open_at[2] + .09, high + site_offset_z)]
            require(stage["site_target_m"] == retreat_at, "Changed vertical retreat target")
        if phase == "settle":
            settled_steps = finish - start
            require(stage["site_target_m"] is None, "Unexpected commanded settle waypoint")
        start = finish
    require(start == end, "Phases do not cover all executed controls")
    if stop == "finished_fixed_diagnostic":
        require(names == list(PHASES) and settled_steps == PHASE_CAPS["settle"], "Normal finish lacks settle phase")
        status = "complete"
    elif stop == "native_environment_refused_diagnostic_tail":
        require(any(all(r["goals"].values()) for r in history), "Censoring lacks the triggering official point")
        status = "censored_native_terminal"
    else:
        require(stop in STOPPED, "Unrecognized phase stop")
        if stop == "environment_done_without_all_goals":
            require(history[-1]["environment_done"] and not all(history[-1]["goals"].values()), "Environment stop lacks evidence")
        status = "phase_stopped"
    return opening, lost_on_descent, settled_steps, status


def placement_metrics(history, target, opening, lost, status, settle_count):
    pairs = [(a, b) for a, b in zip(history, history[1:]) if a["held"][SOUP] and not b["held"][SOUP]]
    first = next(iter(pairs), None)
    controlled = next((b["step"] for a, b in pairs if b["phase"] in OPEN_PHASES), None)
    detach = first[1]["step"] if first else None
    error = ({"last_held": {"step": first[0]["step"], **point_error(first[0], target)},
              "first_unheld": {"step": first[1]["step"], **point_error(first[1], target)}} if first else None)
    tail = history[-STABLE:]
    spread = max((dist(coordinates(a), coordinates(b)) for a, b in combinations(tail, 2)), default=0.)
    errors = [point_error(r, target) for r in tail]
    evaluable = len(tail) == STABLE and detach is not None and tail[0]["step"] >= detach
    complete = evaluable and spread <= SETTLED_TOL and all(
        all(r["goals"].values()) and not any(r["held"].values()) for r in tail)
    release_target = bool(opening and opening["within_tolerance"] and error and detach == controlled
                          and error["last_held"]["within_tolerance"] and error["first_unheld"]["within_tolerance"])
    maintained = len(tail) == STABLE and all(e["within_tolerance"] for e in errors)
    reversals = [{"step": b["step"], "object": obj} for a, b in zip(history, history[1:])
                 for obj in PLACED if a["goals"][obj] and not b["goals"][obj]]
    max_move = {obj: max(dist(coordinates(history[0], obj), coordinates(r, obj)) for r in history) for obj in PLACED}
    return {"first_held_to_unheld_step": detach, "first_held_to_unheld_after_open_command_step": controlled,
            "target_error_at_first_detach": error, "tail_max_target_error": {
                "max_xy_error_m": max(e["xy_error_m"] for e in errors),
                "max_abs_z_error_m": max(e["abs_z_error_m"] for e in errors)},
            "tail_steps": [r["step"] for r in tail], "tail_soup_max_pairwise_displacement_m": spread,
            "physical_completion": complete, "specified_target_achieved_at_controlled_release": release_target,
            "specified_target_maintained_in_tail": maintained,
            "strict_success": complete and release_target and maintained and lost is None and status == "complete"
                              and settle_count == PHASE_CAPS["settle"] and not reversals,
            "completed_object_goal_reversals": reversals, "completed_object_max_displacement_m": max_move}, evaluable


def check_region(region, sample):
    pos, rot, size = (np.asarray(region[k]) for k in ("position_m", "rotation", "size"))
    extent = np.abs(rot @ size)
    lower, upper = pos - extent, pos + extent
    lower[2] -= .01
    require(region["state_type"] == "SiteObjectState" and region["name"] == "basket_1_contain_region",
            "Wrong native basket site")
    for obj, data in region["objects"].items():
        point = np.asarray(coordinates(sample, obj))
        inside = bool(np.all((lower < point) & (point < upper)))
        require(data["body_center_m"] == point.tolist() and data["total_size"] == extent.tolist()
                and data["lower"] == lower.tolist() and data["upper"] == upper.tolist()
                and data["lower_margin_m"] == (point - lower).tolist() and data["upper_margin_m"] == (upper - point).tolist()
                and data["library_check_contact"] is True
                and data["library_check_contain"] is data["official_in_predicate"] is data["strictly_inside"]
                    is sample["goals"][obj] is inside, "Saved site margins disagree with native predicate")
    require(set(region["objects"]) == set(GOALS), "Incomplete region predicate coverage")


def checked_case(path, report, source, slot):
    folder = path.parent.resolve()
    def lines(filename, field):
        return read_lines(checked_path(folder, filename, report[field]))
    prefix = lines("prefix-actions.jsonl", "original_replay_sha256")
    grasp = lines("grasp-audit.jsonl", "grasp_capture_sha256")
    controlled = lines("controlled-steps.jsonl", "controlled_steps_sha256")
    require(report["schema"] == "v2-soup-placement-feasibility-case/1" and report["slot"] == slot
            and report["layout_seed"] == LAYOUT and report["offset_xy_m"] == list(OFFSETS[slot])
            and prefix == source["actions"] and report["original_ipc_rows_exact"] == len(prefix),
            "Changed assigned slot or original action prefix")
    fork = report["fork_physics"]
    history = [fork, *controlled]
    end = history[-1]["step"]
    at = {r["step"]: r for r in history}
    require([r["step"] for r in history] == list(range(FORK, end + 1)) and end <= 710
            and report["controlled_steps_executed"] == len(controlled)
            and report["final_physics"] == history[-1] and stamp(report["fork"]) == stamp(prefix[-1])
            and fork["state_sha256"] == prefix[-1]["state_sha256"]
            and fork["goals"] == prefix[-1]["audit"]["goals"] == source["contact_fork"]["goals"]
            and fork["held"] == source["grasp"][-1]["held_by_target"], "Fork, budget or step continuity differs")
    for obj in GOALS:
        a, b = fork["body_poses"][obj], source["contact_fork"]["body_poses"][obj]
        require(a["position_m"] == b["position_m"] and a["rotation"] == b["rotation_matrix"],
                "Fork pose differs from pinned contact replay")
    first_official = next((r for r in controlled if all(r["goals"].values())), None)
    first_step = first_official["step"] if first_official else None
    expected_frames = sorted({FORK, end, *[s for s in at if s % 5 == 0], *([first_step] if first_official else [])})
    frames = report["frames"]
    require([r["step"] for r in frames] == expected_frames and frames[0] == report["fork"]
            and frames[-1] == report["final"] and grasp[:len(source["grasp"])] == source["grasp"],
            "Missing archived capture or changed original grasp sequence")
    after_fork = []
    for frame in frames:
        step = frame["step"]
        sample = at[step]
        require(frame["file"] == f"frames/{step}.npz" and frame["state_sha256"] == sample["state_sha256"]
                and all(frame[k] == prefix[-1][k] for k in ("raw_sha256", "bddl_sha256")),
                "Unpaired frame/native state")
        observation(checked_path(folder, frame["file"], frame["sha256"]), stamp(frame))
        check_region(frame["region"], sample)
        if step > FORK:
            after_fork.append({"step": step, "state_sha256": sample["state_sha256"],
                               "goals": sample["goals"], "held_by_target": sample["held"]})
    require(grasp[len(source["grasp"]):] == after_fork, "Postfork frame grasp sidecar differs")
    preview = report["final_preview"]
    require(preview["file"] == "final.png", "Unexpected preview path")
    checked_path(folder, preview["file"], preview["sha256"])
    for r in history:
        require(all(set(r[k]) == set(GOALS) and all(type(v) is bool for v in r[k].values()) for k in ("goals", "held")),
                "Malformed native goal/grasp states")
    require(all(type(r["environment_done"]) is bool and
                r["past_first_official_goal"] is bool(first_step is not None and r["step"] > first_step) for r in controlled),
            "Wrong native/diagnostic terminal separation")
    official_summary = None
    if first_official:
        frame = next(r for r in frames if r["step"] == first_step)
        official_summary = {"step": first_step, "held": first_official["held"], "environment_done": first_official["environment_done"],
                            "fingerprint": stamp(frame), "phase": first_official["phase"], "frame": frame["file"]}
    require(official_summary == report["first_official_all_goals"]
            and report["diagnostic_post_official_control_count"] == (0 if first_step is None else end - first_step),
            "First native success marker or diagnostic continuation count differs")
    region = frames[0]["region"]
    x, y, z = region["position_m"]
    dx, dy = OFFSETS[slot]
    target = [x + dx, y + dy, z]
    stop = report["stop_reason"]
    opening, lost, settle_count, status = check_phases(history, report["phase_results"], target, stop)
    metrics, evaluable = placement_metrics(history, target, opening, lost, status, settle_count)
    require(report["body_target_m"] == target and report["target_error_before_open"] == opening
            and report["target_error_at_end"] == point_error(history[-1], target)
            and report["shared_site_minus_soup_translation_m"] == [
                a - b for a, b in zip(fork["site_xyz_m"], coordinates(fork), strict=True)]
            and report["shared_site_rotation"] == fork["site_rotation"]
            and report["diagnostic_status"] == status and report["descent_grasp_lost_step"] == lost
            and all(report[k] == v for k, v in metrics.items()), "Reported target, stabilization or completion differs")
    summary = {k: report[k] for k in ("slot", "offset_xy_m", "body_target_m", "controlled_steps_executed",
                "target_error_before_open", "target_error_at_end", "phase_results", "diagnostic_status", "stop_reason",
                "descent_grasp_lost_step", "diagnostic_post_official_control_count", "first_official_all_goals")}
    return {**summary, **metrics, "stability_evaluable_at_terminal": evaluable,
            "final_step": end, "final_goals": history[-1]["goals"], "final_held": history[-1]["held"],
            "final_soup_center_m": coordinates(history[-1]), "archived_frames": len(frames),
            **{k: report[k] for k in ("original_replay_sha256", "grasp_capture_sha256", "controlled_steps_sha256")}}, frames[0]


def check_report_format_repair(root, cases):
    """Tie the earlier uncontrolled-report failures to the repaired action logs."""
    old_root = root.resolve()
    rows = []
    for card, slot in ((2, 0), (3, 1)):
        manifest = read(old_root / f"soup-placement-{card}.json")
        plan = read(checked_path(old_root, f"soup-placement-plan-{card}.json", manifest["plan_sha256"]))
        require(manifest["complete"] is False and manifest["card"] == plan["card"] == card
                and manifest["cases"] == [] and plan["runner_sha256"] == ORIGINAL_RUNNER_SHA
                and plan["planned"][0]["slot"] == slot, "Not the earlier report-format interruption")
        report = cases[slot]
        digests = {"controlled-steps.jsonl": report["controlled_steps_sha256"],
                   "prefix-actions.jsonl": report["original_replay_sha256"],
                   "grasp-audit.jsonl": report["grasp_capture_sha256"]}
        folder = old_root / f"slot-{slot}"
        for name, digest in digests.items():
            checked_path(folder, name, digest)
        rows.append({"slot": slot, "card": card, "interrupted_plan_sha256": manifest["plan_sha256"],
                     "control_and_physics_sha256_identical": digests})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source-run", "source-audit", "source-report", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--interrupted-run", type=Path, help="Original outputs from the report-format interruption")
    a = p.parse_args()
    root = a.run.resolve()
    require(not a.out.exists(), "Diagnostic summary already exists")
    source = source_case(a.source_run.resolve(), a.source_audit.resolve(), a.source_report.resolve())
    runner_sha = sha(HERE / "soup_placement_feasibility_v2.py")
    require(runner_sha == REPAIRED_RUNNER_SHA and sha(HERE / "soup_placement_feasibility.py") == ORIGINAL_RUNNER_SHA
            and sha(HERE / "actual_physics_simulator.py") == PHYSICS_SHA, "Wrong runner or physical sampler")
    result, cards, forks = [], [], []
    for card, slots in CARD_SLOTS.items():
        index_path = root / f"soup-placement-{card}.json"
        index = read(index_path)
        plan = read(checked_path(root, f"soup-placement-plan-{card}.json", index["plan_sha256"]))
        require(index["schema"] == "v2-soup-placement-feasibility-manifest/1" and index["complete"] is True
                and index["card"] == plan["card"] == card and tuple(r["slot"] for r in index["cases"]) == slots
                and plan["schema"] == "v2-soup-placement-feasibility-plan/1" and plan["source_layout"] == LAYOUT
                and plan["source_arm"] == "adaptive1" and plan["source_audit_sha256"] == AUDIT_SHA
                and plan["source_contact_report_sha256"] == SOURCE_REPORT_SHA and plan["runner_sha256"] == runner_sha
                and plan["physics_wrapper_sha256"] == PHYSICS_SHA and plan["fork"] == stamp(source["actions"][-1])
                and plan["source_episode_sha256"] == source["entry"]["result_sha256"]
                and plan["source_actions_sha256"] == source["entry"]["actions_sha256"]
                and plan["source_grasp_sha256"] == source["entry"]["grasp_audit_sha256"]
                and plan["source_bddl_sha256"] == source["initial"]["bddl_sha256"]
                and plan["phase_caps"] == PHASE_CAPS and plan["maximum_diagnostic_step"] == FORK + sum(PHASE_CAPS.values()) == 710
                and plan["native_total_budget"] == BUDGET and plan["stable_tail_steps"] == STABLE
                and plan["stable_position_tolerance_m"] == SETTLED_TOL
                and plan["opening_xy_tolerance_m"] == XY_TOL and plan["opening_z_tolerance_m"] == Z_TOL
                and plan["planned"] == [{"slot": s, "offset_xy_m": list(OFFSETS[s])} for s in slots],
                "Changed source, plan or five-case card assignment")
        cards.append({"card": card, "plan_sha256": index["plan_sha256"], "manifest_sha256": sha(index_path)})
        for item in index["cases"]:
            slot = item["slot"]
            require(item["report"] == f"slot-{slot}/report.json", "Unexpected case report path")
            path = checked_path(root, item["report"], item["report_sha256"])
            case, fork = checked_case(path, read(path), source, slot)
            require(all(case[k] == item[k] for k in ("physical_completion", "strict_success", "stop_reason",
                                                    "diagnostic_status", "controlled_steps_executed")),
                    "Manifest outcome differs from independent score")
            result.append({**item, **case, "card": card})
            forks.append(fork)
    require(len(result) == 5 and all(fork == forks[0] for fork in forks), "Five cases do not share the original fork")
    contact = read(a.source_report)
    contact_rows = read_lines(checked_path(a.source_report.resolve().parent, "contacts.jsonl", contact["contacts_sha256"]))
    old450, = [r for r in contact_rows if r["step"] == 450]
    require(old450["state_sha256"] == old450["state_sha256_after_read"], "Source step 450 is not an audited contact capture")
    contact_meta = {k: old450[k] for k in ("step", "state_sha256", "goals")}
    contact_meta["body_poses"] = {name: old450["body_poses"][name] for name in ("basket_1", SOUP, *PLACED)}
    by_slot = sorted(result, key=lambda r: r["slot"])
    repair = check_report_format_repair(a.interrupted_run, by_slot) if a.interrupted_run else []
    write(a.out, {"schema": "v2-soup-placement-feasibility-audit/1", "scope": "Five prescribed targets on the same seen source grasp",
                  "source_layout": LAYOUT, "source_arm": "adaptive1", "source_audit_sha256": AUDIT_SHA,
                  "source_contact_report_sha256": SOURCE_REPORT_SHA, "source_contacts_sha256": contact["contacts_sha256"],
                  "source_episode_sha256": source["entry"]["result_sha256"], "source_actions_sha256": source["entry"]["actions_sha256"],
                  "source_grasp_sha256": source["entry"]["grasp_audit_sha256"], "runner_sha256": runner_sha,
                  "audit_sha256": sha(Path(__file__)), "cards": cards, "common_fork": forks[0],
                  "report_format_repair_parity": repair,
                  "source_contact_450": contact_meta, "cases": by_slot, "independent_pilot_release": False,
                  "physical_completion_slots": [r["slot"] for r in by_slot if r["physical_completion"]],
                  "strict_success_slots": [r["slot"] for r in by_slot if r["strict_success"]],
                  "censored_slots": [r["slot"] for r in by_slot if r["diagnostic_status"] == "censored_native_terminal"]})
    print("SOUP_PLACEMENT_AUDIT", [(r["slot"], r["physical_completion"], r["strict_success"]) for r in by_slot], flush=True)


if __name__ == "__main__":
    main()
