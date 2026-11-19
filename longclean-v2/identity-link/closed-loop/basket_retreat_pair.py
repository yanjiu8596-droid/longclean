"""Paired retreat and equal-duration open hold from a sealed stall state."""

import argparse
from math import dist
from pathlib import Path

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import BUDGET, CHEESE, CHUNK, GOALS, SAUCE, SOUP, read, sha, write
from basket_stall_probe import archive, replay_row, replay_rows, sensor_position, source_arm, verify_sources
from dense_online import physical_sample, reason_at_terminal
from dense_online_diagnostic import checked, jsonl
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
PROBE_SHA = "df9860e40c70f4f73148acdbc8e17163764458484b0c1138e3dd4cedf75da6fc"
PROBE_CODE_SHA = "149f17ade335e6205b5c0a7bcf3d3f0e8567b2fa96790f1dfddcc44f90743448"
LAYOUTS = (931303, 931307)
ORDERS = (("hold", "retreat"), ("retreat", "hold"))
LIFT, TRANSLATE, FIRST_GRASP_WINDOW = 20, 60, 130
PLACED = (CHEESE, SAUCE)
SCENE_BODIES = (*GOALS, "basket_1")
CONTROLLER_GRIP = [0.] * 6 + [-1.]
MAX_SHIFT, MIN_CLEAR_Z, HOME_XY_ERROR = .005, .74, .025


def load_probe(root, bddl):
    require(sha(HERE / "basket_stall_probe.py") == PROBE_CODE_SHA, "Changed frozen trigger source")
    probe = read(checked(root, "summary.json", PROBE_SHA))
    require(probe["schema"] == "v2-basket-stall-probe/1" and probe["source_sha256"] == PROBE_CODE_SHA
            and probe["study_bddl_sha256"] == sha(bddl) and probe["develop_retreat_pair"] is True
            and probe["admitted_layouts"] == list(LAYOUTS), "Unqualified or changed trigger cohort")
    cases = []
    for layout in LAYOUTS:
        case, = [c for c in probe["cases"] if c["layout_seed"] == layout]
        report = read(checked(root, case["report"], case["report_sha256"]))
        require(report["first_trigger"] == case["first_trigger"] and report["initial"] == case["initial"]
                and report["evaluation_only"]["truth_admits_first_trigger_window"] is True,
                "Case trigger report changed")
        trigger = case["first_trigger"]
        directory = root / str(layout)
        for saved in (case["initial"], trigger["observation"]):
            observation(checked(directory, saved["file"], saved["sha256"]), saved["fingerprint"])
        cases.append(case)
    return cases


def osc_delta(point, waypoint):
    from numpy import float32

    delta = [max(-.7, min(.7, 1.4 * (goal - now) / .05)) for now, goal in zip(point, waypoint, strict=True)]
    return [float(float32(v)) for v in delta] + [0., 0., 0., -1.]


def intervene(api, worker, stream, arm, start, home):
    height = max(.76, start[2] + .08)
    up = [*start[:2], height]
    home_high = [*home[:2], height]
    before = worker.current["audit"]["positions"]
    furthest = {obj: 0. for obj in SCENE_BODIES}
    samples = []
    failure = None
    point = start
    for i in range(LIFT + TRANSLATE):
        if arm == "retreat" and i == LIFT and point[2] < MIN_CLEAR_Z:
            failure = "vertical_clearance_not_reached"
            break
        phase = "hold" if arm == "hold" else "lift" if i < LIFT else "translate"
        actions = CONTROLLER_GRIP if arm == "hold" else osc_delta(point, up if phase == "lift" else home_high)
        worker.step([actions], None, f"basket recovery {phase}")
        frame = api.Frame.read(worker.npz)
        point = sensor_position(frame)
        native = worker.current
        truth = physical_sample(stream, native)
        for obj in SCENE_BODIES:
            displacement = dist(before[obj], native["audit"]["positions"][obj])
            furthest[obj] = max(furthest[obj], displacement)
        samples.append({"step": native["step"], "phase": phase, "eef_m": point, "gap_m": frame.gap})
        if (any(truth["held_by_target"].values()) or truth["goals"] != {obj: obj != SOUP for obj in GOALS}
                or max(furthest.values()) > MAX_SHIFT or worker.terminal()):
            failure = "intervention_physics_gate"
            break
    xy_error = dist(point[:2], home[:2])
    if failure is None and arm == "retreat" and (xy_error > HOME_XY_ERROR or point[2] < MIN_CLEAR_Z):
        failure = "home_view_not_reached"
    return {"planned_steps": LIFT + TRANSLATE, "executed_steps": len(samples), "motion_targets_m": [up, home_high],
            "initial_eef_m": start, "final_eef_m": point, "home_xy_error_m": xy_error,
            "object_max_displacement_m": furthest, "failure": failure, "samples": samples}


def resume(api, worker, stream, policy, seed):
    policy.reset(seed)
    chunks, first_grasp = [], None
    nearest = None
    truth = None
    prompt = api.instruction(SOUP)
    while not worker.terminal():
        start = worker.current["step"]
        issued = policy.act(worker, prompt, limit=CHUNK)
        require(len(issued) == min(CHUNK, BUDGET - start), f"Invalid policy block length at {start}")
        realized = []
        for i in range(0, len(issued), 5):
            realized.extend(worker.step(issued[i:i + 5], SOUP, prompt))
            truth = physical_sample(stream, worker.current)
            native = worker.current["audit"]
            approach = dist(native["site_xyz"], native["positions"][SOUP])
            nearest = approach if nearest is None else min(nearest, approach)
            if truth["held_by_target"][SOUP] and first_grasp is None:
                first_grasp = truth["step"]
            if worker.terminal():
                break
        chunks.append({"start_step": start, "end_step": worker.current["step"], "prompt": prompt,
                       "issued": issued, "executed": realized})
    require(truth is not None, "No policy endpoint after intervention")
    return chunks, truth, first_grasp, nearest


def run_arm(api, worker_type, policy, bddl, case, source, mode, folder):
    cut = case["first_trigger"]["step"]
    layout = case["layout_seed"]
    seed = source["result"]["policy_seed"]
    original = replay_rows(source, cut)
    with worker_type(bddl, folder, layout) as worker, (folder / "grasp-audit.jsonl").open(encoding="utf-8") as grasp:
        initial = archive(api, worker, folder, "initial.npz")
        require(initial["fingerprint"] == case["initial"]["fingerprint"], "Unpaired reset")
        home = sensor_position(api.Frame.read(worker.npz))
        require(home == case["initial_eef_m"], "Initial sensor reference changed")
        require(physical_sample(grasp, worker.current) == source["physical"][0], "Changed initial physical audit")
        for row in original:
            replay_row(worker, row)
            require(physical_sample(grasp, worker.current) == source["physical"][row["step"]], "Changed prefix physical audit")
        fork = archive(api, worker, folder, "fork.npz")
        require(fork["fingerprint"] == case["first_trigger"]["observation"]["fingerprint"]
                and seed == case["source_policy_seed"], "Unpaired stall decision state or policy seed")
        start = sensor_position(api.Frame.read(worker.npz))
        motion = intervene(api, worker, grasp, mode, start, home)
        resume_state = archive(api, worker, folder, "resume.npz")
        resumed_at = worker.current["step"] if motion["failure"] is None else None
        chunks, first_grasp, nearest = [], None, None
        if resumed_at is not None:
            require(resumed_at == cut + LIFT + TRANSLATE and resumed_at % CHUNK == 0, "Unequal resumed action budget")
            chunks, _, first_grasp, nearest = resume(api, worker, grasp, policy, seed)
        terminal = archive(api, worker, folder, "terminal.npz")
        reason = motion["failure"] or reason_at_terminal(worker.current)
        require(not grasp.readline(), "Extra physical sample after terminal")
    native = jsonl(folder / "actions.jsonl")
    truth = jsonl(folder / "grasp-audit.jsonl")
    require(native[:len(original)] == original and [r["step"] for r in truth] == [0, *[r["step"] for r in native]],
            "Published prefix or physical sample sequence changed")
    suffix = [r for r in native if r["start_step"] >= cut]
    resumed = [r for r in suffix if resumed_at is not None and r["start_step"] >= resumed_at]
    require([a for chunk in chunks for a in chunk["executed"]] == [a for r in resumed for a in r["actions"]],
            "Policy controls differ from realized action trace")
    reversals = [event for row in suffix for event in row["transitions"]
                 if event["object"] in PLACED and not event["in_basket"]]
    final = truth[-1]
    official = all(final["goals"].values())
    result = {"schema": "v2-basket-retreat-arm/1", "layout_seed": layout, "arm": mode, "policy_seed": seed,
              "fork": fork, "resume": resume_state, "resumed_at": resumed_at, "terminal": terminal,
              "terminal_reason": reason, "terminal_grasp_and_goals": final, "cutoff": cut,
              "original_prefix_intervals": len(original), "motion": motion, "policy_chunks": chunks,
              "first_sampled_soup_grasp": first_grasp, "minimum_eef_to_soup_center_m": nearest,
              "grasp_within_130": bool(first_grasp is not None and first_grasp - resumed_at <= FIRST_GRASP_WINDOW),
              "official_success": official, "official_success_all_unheld": official and not any(final["held_by_target"].values()),
              "completed_goal_reversals": reversals, "actions_sha256": sha(folder / "actions.jsonl"),
              "grasp_sha256": sha(folder / "grasp-audit.jsonl")}
    write(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=GPU, required=True)
    for name in ("study", "probe", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    verify_sources()
    cases = load_probe(args.probe.resolve(), bddl)
    import run as api
    from actual_physics_worker import worker_class

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "v2-basket-retreat-plan/1", "probe_sha256": PROBE_SHA, "runner_sha256": sha(Path(__file__)),
            "card": args.card, "budget": BUDGET, "recovery_steps": LIFT + TRANSLATE,
            "cases": [{"layout_seed": case["layout_seed"], "cutoff": case["first_trigger"]["step"],
                       "policy_seed": case["source_policy_seed"], "order": list(order)}
                      for case, order in zip(cases, ORDERS, strict=True)]}
    write(out / "plan.json", plan)
    worker_type = worker_class(api)
    policy = api.Pi05(GPU[args.card])
    summaries = []
    for case, order in zip(cases, ORDERS, strict=True):
        layout = case["layout_seed"]
        source = source_arm(args.source_root.resolve(), layout)
        compared = {}
        for mode in order:
            folder = out / str(layout) / mode
            result = run_arm(api, worker_type, policy, bddl, case, source, mode, folder)
            fields = ("fork", "resumed_at", "terminal", "terminal_reason", "policy_seed", "first_sampled_soup_grasp",
                      "minimum_eef_to_soup_center_m", "grasp_within_130", "official_success", "official_success_all_unheld",
                      "completed_goal_reversals", "terminal_grasp_and_goals", "actions_sha256", "grasp_sha256")
            compared[mode] = {key: result[key] for key in fields}
            compared[mode].update(motion={k: v for k, v in result["motion"].items() if k != "samples"},
                                  result=(folder / "result.json").relative_to(out).as_posix(),
                                  result_sha256=sha(folder / "result.json"), policy_calls=len(result["policy_chunks"]))
            print("BASKET_RETREAT", layout, mode, result["terminal_reason"], result["first_sampled_soup_grasp"],
                  result["official_success_all_unheld"], flush=True)
        a, b = compared["hold"], compared["retreat"]
        require(a["fork"]["fingerprint"] == b["fork"]["fingerprint"] and a["policy_seed"] == b["policy_seed"],
                "Comparison fork or stochastic seed changed")
        if a["resumed_at"] is not None and b["resumed_at"] is not None:
            require(a["resumed_at"] == b["resumed_at"], "Comparison remaining budget changed")
        summaries.append({"layout_seed": layout, "arms": compared})
    write(out / "summary.json", {"schema": "v2-basket-retreat-pair/1", "plan_sha256": sha(out / "plan.json"),
                                 "producer_sha256": sha(Path(__file__)), "pairs": summaries,
                                 "scope": "Seen development comparison with two sealed stall states."})


if __name__ == "__main__":
    main()
