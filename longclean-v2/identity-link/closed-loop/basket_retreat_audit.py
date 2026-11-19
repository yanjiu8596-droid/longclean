"""Recompute the paired recovery scores from archived native controls and audits."""

import argparse
from math import dist
from pathlib import Path

from basket_retreat_pair import LAYOUTS, ORDERS, SCENE_BODIES, PLACED, PROBE_SHA, MAX_SHIFT, MIN_CLEAR_Z, HOME_XY_ERROR
from basket_stall_probe import replay_rows, source_arm
from common import BUDGET, GOALS, SOUP, read, sha, write
from dense_online_diagnostic import checked, jsonl, require
from target_follow_audit import observation, stamp


RUNNER = "33960485c95107bd7fc5f1cbf874f219ba3b2f9f4b7f10204fbc998d3d68c855"


def check_arm(root, item, case, source, mode):
    layout, fork_step = case["layout_seed"], case["cutoff"]
    path = checked(root, item["result"], item["result_sha256"])
    arm = read(path)
    require(arm["layout_seed"] == layout and arm["arm"] == mode and
            arm["policy_seed"] == case["policy_seed"], "Assigned arm or policy seed differs")
    rows = jsonl(checked(path.parent, "actions.jsonl", arm["actions_sha256"]))
    physical = jsonl(checked(path.parent, "grasp-audit.jsonl", arm["grasp_sha256"]))
    original = replay_rows(source, fork_step)
    require(rows[:len(original)] == original and arm["original_prefix_intervals"] == len(original)
            and physical[:len(original) + 1] == [source["physical"][s] for s in (0, *[r["step"] for r in original])],
            "Replayed prefix differs from source")
    end = 0
    physics_by_step = {}
    for row, sample in zip(rows, physical[1:], strict=True):
        require(row["start_step"] == end and row["step"] == end + len(row["actions"]) <= BUDGET
                and sample["step"] == row["step"] and sample["goals"] == row["audit"]["goals"]
                and sample["state_sha256"] == row["state_sha256"], "Native sidecar does not match execution")
        end = row["step"]
        physics_by_step[end] = sample
    by_step = {r["step"]: r for r in rows}
    for field in ("fork", "resume", "terminal"):
        saved = arm[field]
        fingerprint = saved["fingerprint"]
        require(stamp(by_step[fingerprint["step"]]) == fingerprint, "Archive snapshot differs from native state")
        observation(checked(path.parent, saved["file"], saved["sha256"]), fingerprint)
    motion = arm["motion"]
    motion_end = arm["resume"]["fingerprint"]["step"]
    intervention = [r for r in rows if fork_step < r["step"] <= motion_end]
    shifts = {obj: max((dist(by_step[fork_step]["audit"]["positions"][obj], r["audit"]["positions"][obj])
                        for r in intervention), default=0.) for obj in SCENE_BODIES}
    require(motion["executed_steps"] == len(intervention) == motion_end - fork_step and
            motion["object_max_displacement_m"] == shifts, "Intervention control count or displacement differs")
    for row in intervention:
        action, = row["actions"]
        require(action[3:] == [0., 0., 0., -1.] and (mode == "retreat" or action[:3] == [0., 0., 0.])
                and row["target"] is None, "Intervention used a different action method")
    resumed = arm["resumed_at"]
    remaining = [r for r in rows if r["step"] > motion_end]
    if resumed is not None:
        require(resumed == motion_end == fork_step + 80 and motion["failure"] is None and
                max(shifts.values()) <= MAX_SHIFT and all(not any(physics_by_step[r["step"]]["held_by_target"].values())
                    and physics_by_step[r["step"]]["goals"] == {obj: obj != SOUP for obj in GOALS} for r in intervention),
                "The physical intervention gate was bypassed")
        if mode == "retreat":
            require(motion["samples"][19]["eef_m"][2] >= MIN_CLEAR_Z and motion["final_eef_m"][2] >= MIN_CLEAR_Z
                    and motion["home_xy_error_m"] <= HOME_XY_ERROR, "Retreat failed to reach the observation position")
        require(bool(remaining) and all(r["target"] == SOUP for r in remaining), "Changed resumed policy target")
    else:
        require(not remaining and motion["failure"] is not None, "Unadmitted arm continued policy inference")
    require([a for c in arm["policy_chunks"] for a in c["executed"]] == [a for r in remaining for a in r["actions"]],
            "Resumed action trace differs from the model log")
    hold = next((r["step"] for r in remaining if physics_by_step[r["step"]]["held_by_target"][SOUP]), None)
    reversals = [event for r in rows if r["step"] > fork_step for event in r["transitions"]
                 if event["object"] in PLACED and not event["in_basket"]]
    final = physical[-1]
    success = all(final["goals"].values())
    unheld = success and not any(final["held_by_target"].values())
    improve_grasp = hold is not None and hold <= resumed + 130
    require(hold == arm["first_sampled_soup_grasp"] and improve_grasp is arm["grasp_within_130"] and
            reversals == arm["completed_goal_reversals"] and final == arm["terminal_grasp_and_goals"] and
            success is arm["official_success"] and unheld is arm["official_success_all_unheld"], "Resumed result score differs")
    keys = ("resumed_at", "first_sampled_soup_grasp", "minimum_eef_to_soup_center_m", "terminal_reason",
            "official_success", "official_success_all_unheld", "completed_goal_reversals")
    return {**{k: arm[k] for k in keys}, "terminal_step": end, "grasp_within_130": improve_grasp,
            "intervention_failure": motion["failure"], "final_eef_m": motion["final_eef_m"],
            "max_object_displacement_m": max(shifts.values()), "final_goals": final["goals"],
            "result_sha256": item["result_sha256"], "actions_sha256": arm["actions_sha256"],
            "grasp_sha256": arm["grasp_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    require(not args.out.exists(), "Audit output already exists")
    summary = read(root / "summary.json")
    plan = read(checked(root, "plan.json", summary["plan_sha256"]))
    require(plan["probe_sha256"] == PROBE_SHA and plan["runner_sha256"] == summary["producer_sha256"] == RUNNER
            and [c["layout_seed"] for c in plan["cases"]] == list(LAYOUTS) and plan["recovery_steps"] == 80
            and [c["order"] for c in plan["cases"]] == list(map(list, ORDERS)), "Unexpected paired plan")
    results = []
    for pair, case in zip(summary["pairs"], plan["cases"], strict=True):
        layout = case["layout_seed"]
        require(layout == pair["layout_seed"] and case["cutoff"] == 360, "Changed recovery checkpoint")
        source = source_arm(args.source_root.resolve(), layout)
        arms = {mode: check_arm(root, pair["arms"][mode], case, source, mode) for mode in ("hold", "retreat")}
        results.append({"layout_seed": layout, "arms": arms})
    write(args.out, {"schema": "v2-basket-retreat-audit/1", "pairs": results, "summary_sha256": sha(root / "summary.json"),
                     "audit_sha256": sha(Path(__file__)), "runner_sha256": RUNNER, "probe_sha256": PROBE_SHA})


if __name__ == "__main__":
    main()
