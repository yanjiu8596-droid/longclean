"""Execute the frozen stage 4 policy pairs; only the soup writer differs."""
import argparse
import os
from pathlib import Path
from shutil import copyfile
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]
from common import BUDGET, CHEESE, CHUNK, GPU, SAUCE, SOUP, read, sha, write
from stage4_fixture import ARMS, code_lock, require, validate_gpu, validate_study


STRIDE = 5


def audited(event, current):
    """Truth joins the emitted event only after the observer has returned."""
    obj = event["object"]
    goal = current["audit"]["goals"][obj]
    held = current["audit"]["held_soup"]
    return {**event, "step": current["step"], "matches_audit": goal,
            "held_soup": held, "confirmed_in_audit": goal and (obj != SOUP or not held)}


def replay_one(api, progress_type, policy, locate, bddl, fixture, dest, case, arm):
    if (dest / "result.json").is_file():
        saved = read(dest / "result.json")
        require(saved["arm"] == arm and saved["layout_seed"] == case["layout_seed"]
                and saved["prefix_sha256"] == case["prefix_sha256"]
                and saved["actions_sha256"] == sha(dest / "actions.jsonl")
                and saved["fork_sha256"] == sha(dest / "fork.npz"), f"Invalid existing episode: {dest}")
        return saved
    if dest.exists():
        dest.rename(dest.with_name(dest.name + f".partial-{os.getpid()}"))
    prefix = read(fixture / "prefix.json")
    design = read(fixture / "fixture.json")
    tracker = progress_type(locate, write_soup=(arm == "candidate"))
    state_events, soup_events, model_chunks, schedule_events = [], [], [], []
    completed_reassigned, order = [], []
    first_movement, regrasp_steps = None, 0
    with api.Worker(bddl, dest, case["layout_seed"], video=True) as worker:
        initial = api.fingerprint(worker.current)
        require(initial == prefix["initial"] and initial["raw_sha256"] == case["raw_sha256"]
                and initial["state_sha256"] == case["initial_state_sha256"], "Incorrect paired initialization")
        api.replay_prefix(worker, prefix, tracker)
        pre = api.fingerprint(worker.current)
        require(pre == prefix["end"] and tracker.complete(CHEESE), "Common cheese prefix changed")
        if case["moved"]:
            require(case["offset"] == design["offset"], "Frozen perturbation changed")
            boundary = worker.call({"op": "move", "offset": case["offset"]})
            require("rejected" not in boundary and api.fingerprint(boundary) == design["perturbed_checkpoint"],
                    "Physical perturbation did not match the verified checkpoint")
        else:
            boundary = worker.current
        fork = api.fingerprint(boundary)
        require(fork["step"] == pre["step"] and fork["step"] % CHUNK == 0, "Illegal branch point")
        copyfile(worker.npz, dest / "fork.npz")
        tracker.start_soup(api.Frame.read(worker.npz), fork["step"])
        policy.reset(case["policy_seed"])
        before = design["soup_before"]
        schedule = api.Schedule(tracker, True, before["xyz_m"] if before else None)
        while not worker.terminal():
            step = worker.current["step"]
            if (step <= pre["step"] + CHUNK and not schedule.reordered
                    and not tracker.complete(SOUP) and not tracker.complete(SAUCE)):
                soup = locate(worker.npz)
                info = schedule.change(soup["xyz_m"] if soup else None, api.sauce_location(api.Frame.read(worker.npz)))
                if "after" in info:
                    schedule_events.append({"step": step, **info})
                if first_movement is None:
                    first_movement = {"step": step, **info}
            obj = schedule.next(step)
            if obj is None:
                reason = "observed_complete"
                break
            task = api.instruction(obj)
            if not order or order[-1]["target"] != obj:
                order.append({"step": step, "target": obj})
            if worker.goal(obj):
                completed_reassigned.append({"step": step, "target": obj})
            actions = policy.act(worker, task)
            require(len(actions) == min(CHUNK, BUDGET - step), f"Variable policy chunk before step {step}")
            actual, sensor_steps = [], []
            for index in range(0, len(actions), STRIDE):
                # Exactly one frozen inference per chunk; no scheduling here.
                half = worker.step(actions[index:index + STRIDE], obj, task)
                actual.extend(half)
                snapshot = worker.current
                regrasp_steps += snapshot["completed_regrasp_steps"]
                sensor_steps.append(snapshot["step"])
                frame = api.Frame.read(worker.npz)
                for event in tracker.observe_soup(frame, half, snapshot["step"]):
                    require(event["step"] == snapshot["step"], "Observer event detached from the sensor sample")
                    row = audited(event, snapshot)
                    soup_events.append(row)
                    if row["event"] == "in_basket":
                        name = f"soup-commit-{snapshot['step']}.npz"
                        copyfile(worker.npz, dest / name)
                        row["evidence_file"] = name
                    if row["applied"]:
                        state_events.append(row)
                if worker.terminal():
                    break
            model_chunks.append({"start_step": step, "end_step": worker.current["step"], "target": obj,
                                 "requested": len(actions), "executed": len(actual), "sensor_steps": sensor_steps})
            legacy = tracker.observe(frame, obj, actual)
            if legacy:
                state_events.append(audited({**legacy, "source": "legacy", "writer": "legacy", "applied": True},
                                            worker.current))
        else:
            reason = "goal" if all(worker.current["audit"]["goals"].values()) else "budget"
        final = worker.current
        fingerprint = api.fingerprint(final)
    record = {"arm": arm, "layout_seed": case["layout_seed"], "policy_seed": case["policy_seed"],
              "method": "replan", "moved": case["moved"], "offset": case["offset"],
              "success": all(final["audit"]["goals"].values()), "terminal_goals": final["audit"]["goals"],
              "terminal_audit": final["audit"], "terminal_checkpoint": fingerprint,
              "steps": final["step"], "after_fork_steps": final["step"] - pre["step"], "terminal_reason": reason,
              "prefix_sha256": case["prefix_sha256"], "fixture_sha256": case["fixture_sha256"],
              "initial": initial, "base_checkpoint": pre, "trial_checkpoint": fork,
              "initial_motion_observation": first_movement, "schedule_changes": schedule_events,
              "target_runs": order, "state_events": state_events, "soup_events": soup_events,
              "policy_chunks": model_chunks, "observed_terminal": tracker.status(),
              "completed_reassigned_chunks": completed_reassigned, "completed_regrasp_steps": regrasp_steps,
              "actions_sha256": sha(dest / "actions.jsonl"), "fork_sha256": sha(dest / "fork.npz"),
              "gpu_placement": policy.placement}
    write(dest / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "pilot"))
    parser.add_argument("--card", required=True, choices=GPU, type=int)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    study, output = args.study.resolve(), args.output.resolve()
    validate_gpu(args.card)
    bddl, old = validate_study(study)
    plan, frozen = read(output / "plan.json"), read(output / "frozen.json")
    require(frozen["plan_sha256"] == sha(output / "plan.json") and frozen["code"] == code_lock(old)
            and frozen["cases"] == plan["cases"] and plan["phase"] == args.phase
            and plan["study4_sha256"] == sha(study / "frozen.json")
            and (plan["policy_chunk"], plan["observer_stride"], plan["budget"]) == (CHUNK, STRIDE, BUDGET),
            "The closed-loop freeze does not match the run")
    for card in GPU:
        require(sha(output / f"screen-{card}.json") == plan["screen_sha256"][str(card)], "Fixture qualification changed")
    selected = [r for r in frozen["cases"] if r["card"] == args.card]
    for case in selected:
        fixture = output / "fixtures" / str(case["layout_seed"])
        require(set(case["order"]) == set(ARMS) and len(case["order"]) == len(ARMS)
                and sha(fixture / "fixture.json") == case["fixture_sha256"]
                and sha(fixture / "prefix.json") == case["prefix_sha256"], "Changed paired fixture")
    import run as api
    from progress import ClosedLoopProgress

    policy, locate = api.Pi05(GPU[args.card]), api.load_locator()
    marker = output / f"run-{args.card}.json"
    progress = {"schema": "v2-closed-loop-execution/1", "phase": args.phase, "card": args.card,
                "freeze_sha256": sha(output / "frozen.json"), "planned": [s["tag"] for s in selected],
                "episodes": [], "complete": False}
    write(marker, progress)
    for case in selected:
        fixture = output / "fixtures" / str(case["layout_seed"])
        for arm in case["order"]:
            dest = output / "episodes" / case["tag"] / arm
            record = replay_one(api, ClosedLoopProgress, policy, locate, bddl, fixture, dest, case, arm)
            progress["episodes"].append({"tag": case["tag"], "arm": arm, "result_sha256": sha(dest / "result.json")})
            write(marker, progress)
            print("EPISODE_VERIFIED", case["tag"], arm, record["success"], record["steps"],
                  record["terminal_reason"], flush=True)
    progress["complete"] = True
    write(marker, progress)


if __name__ == "__main__":
    main()
