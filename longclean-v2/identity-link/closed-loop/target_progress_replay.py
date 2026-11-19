"""Replay exactly two sealed no-prefix soup trajectories; compare progress writers."""
import argparse
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, code_lock, read, require, sha, validate_gpu, validate_study, write
from common import ATTEMPT, BUDGET, CHUNK, GOALS, SAUCE, SOUP
from legacy_parity import trace
from progress import ClosedLoopProgress, SOUP_WRITER, STRIDE
from run_closed_loop import audited
from target_follow_audit import observation


AUDIT_SHA = "0d00abc23a0264bb783084332b41ea91e538d76e6e992dad8f836d08592c2be3"
SEALED = {2: (931200, 46), 3: (931201, 47)}
ARMS = ("baseline", "candidate")
SOURCE_PROMPT = "put the alphabet soup can in the basket"
WINDOW = 360


def checked(path, digest):
    require(path.is_file() and sha(path) == digest, f"Changed sealed artifact: {path}")
    return path


def sealed_source(root, card, bddl, old):
    audit = read(checked(root / "audit.json", AUDIT_SHA))
    require(audit["schema"] == "v2-stage4-target-follow-audit/1" and not audit["pilot_release"]
            and audit["recommendation"] == {"prefix": False, "prompt_id": "standard"},
            "Source is not the selected development comparison")
    here = Path(__file__).resolve().parent
    smoke = read(checked(root.parent / "stage4-smoke" / "frozen.json", audit["smoke_freeze_sha256"]))
    require(code_lock(old) == smoke["code"], "Sealed observer or evaluator source changed")
    checked(here / "target_follow.py", audit["runner_sha256"])
    checked(here / "target_follow_audit.py", audit["audit_sha256"])
    for filename, digest in audit["frozen_native_source_sha256"].items():
        checked(here / filename, digest)
    stamp, = [row for row in audit["manifests"] if row["card"] == card]
    manifest = read(checked(root / f"target-follow-{card}.json", stamp["manifest_sha256"]))
    plan = read(checked(root / f"target-follow-plan-{card}.json", stamp["plan_sha256"]))
    require(manifest["complete"] and manifest["plan_sha256"] == stamp["plan_sha256"]
            and plan["budget"] == WINDOW and plan["bddl_sha256"] == sha(bddl)
            and plan["policy_chunk"] == CHUNK and plan["observer_stride"] == STRIDE,
            "Sealed factorial plan does not match replay")
    layout, seed = SEALED[card]
    fields = {"card": card, "layout_seed": layout, "policy_seed": seed, "fresh": True,
              "prefix": False, "prompt_id": "standard", "prompt": SOURCE_PROMPT}
    row, = [r for r in audit["records"] if all(r[k] == v for k, v in fields.items())]
    item, = [r for r in manifest["records"] if all(r[k] == v for k, v in fields.items())]
    result_name = f"{layout}/no-prefix/standard/result.json"
    require(row["result"] == item["result"] == result_name and row["status"] == item["status"] == "complete"
            and row["result_sha256"] == item["result_sha256"] and row["category"] == "soup_first"
            and row["soup_pending_sauce_boundary"] is not None,
            "Sealed audit and source manifest disagree on the assigned record")
    record_file = checked(root / result_name, row["result_sha256"])
    result = read(record_file)
    require(all(result[k] == v for k, v in fields.items())
            and result["initial"] == result["fork"] == row["fork"] == row["initial"]
            and row["fork"]["step"] == 0 and result["motion"] is False
            and result["suffix_steps"] == row["suffix_steps"] == row["terminal"]["step"] == WINDOW
            and result["terminal_checkpoint"] == row["terminal"]
            and result["terminal_goals"] == row["terminal_goals"]
            and result["terminal_reason"] == row["terminal_reason"] == "budget"
            and result["actions"] == "actions.jsonl" and result["actions_sha256"] == row["actions_sha256"]
            and result["fork_npz_sha256"] == row["fork_npz_sha256"],
            "Source record metadata differs from the sealed audit")
    actions = checked(record_file.parent / "actions.jsonl", row["actions_sha256"])
    observation(checked(record_file.parent / "fork.npz", row["fork_npz_sha256"]), row["fork"])
    return row, actions, stamp


def boundary(schedule, state, official_success):
    step = state["step"]
    before = list(schedule.order)
    prior, until = schedule.target, schedule.until
    target = schedule.next(step)
    return {"step": step, "target": target, "order": list(schedule.order), "order_before": before,
            "prior_target": prior, "until_before": until, "until": schedule.until,
            "official_success": official_success}


def shadow_projection(event):
    return {k: v for k, v in event.items() if k != "applied"}


def compare(boundaries, arms, official):
    baseline, candidate = (arms[arm] for arm in ARMS)
    pairs = [{"step": b["step"], "baseline": b, "candidate": c}
             for b, c in zip(boundaries["baseline"], boundaries["candidate"], strict=True)]
    valid = []
    for event in candidate:
        if event["writer"] != SOUP_WRITER or not event["applied"] or not event["confirmed_in_audit"]:
            continue
        before = [p for p in pairs if p["step"] < event["step"]]
        by = next((p for p in pairs if p["step"] >= event["step"]), None)
        at_write = official[event["step"]]
        after = official[by["step"]] if by else None
        if (before and by and by["step"] < WINDOW and not at_write["goals"][SAUCE]
                and after["goals"][SOUP] and not after["held_soup"] and not after["goals"][SAUCE]
                and SOUP in before[-1]["candidate"]["order"] and SOUP not in by["candidate"]["order"]):
            valid.append({"event": event, "boundary_before": before[-1]["candidate"],
                          "boundary_after": by["candidate"], "baseline_boundary": by["baseline"],
                          "audit_at_write": at_write, "audit_at_boundary": after,
                          "different_eligibility": SOUP in by["baseline"]["order"],
                          "different_target": by["baseline"]["target"] != by["candidate"]["target"]})
    # Compare only boundaries with another recorded policy block to influence.
    operational = [p for p in pairs if p["step"] < WINDOW]
    eligibility = next((p for p in operational
                        if (SOUP in p["baseline"]["order"]) != (SOUP in p["candidate"]["order"])), None)
    target = next((p for p in operational if p["baseline"]["target"] != p["candidate"]["target"]), None)
    checkpoint = pairs[-1]
    require(checkpoint["step"] == WINDOW, "Missing source checkpoint schedule")
    checkpoint_differs = checkpoint["baseline"]["target"] != checkpoint["candidate"]["target"]
    removals = {arm: next((b["step"] for b in boundaries[arm] if SOUP not in b["order"]), None)
                for arm in ARMS}
    wrong = sum(e["applied"] and not e["confirmed_in_audit"] for e in baseline + candidate)
    false_shadow = sum(e["writer"] == SOUP_WRITER and e["event"] == "in_basket"
                       and not e["confirmed_in_audit"] for e in baseline)
    return {"valid_writes": valid, "valid_write": bool(valid), "wrong_writes": wrong,
            "wrong_candidate_observations": false_shadow, "first_eligibility_difference": eligibility,
            "first_target_difference": target or (checkpoint if checkpoint_differs else None),
            "first_target_difference_within_source": target,
            "checkpoint_target_difference": checkpoint if checkpoint_differs else None,
            "earliest_soup_removal": removals}


def replay(api, bddl, row, actions, dest):
    locate = api.load_locator()
    trackers = {arm: ClosedLoopProgress(locate, write_soup=(arm == "candidate")) for arm in ARMS}
    # There is no perturbation or reference displacement in this selected run.
    schedules = {arm: api.Schedule(trackers[arm], True, None) for arm in ARMS}
    events, decisions = ({arm: [] for arm in ARMS} for _ in range(2))
    official, combined = {}, []
    n = 0
    with api.Worker(bddl, dest, row["layout_seed"]) as worker:
        fork = api.fingerprint(worker.current)
        require(fork == row["fork"] and not worker.terminal()
                and worker.current["audit"]["goals"] == dict.fromkeys(GOALS, False)
                and not worker.current["audit"]["held_soup"], "Initial simulator fork or audit diverged")
        copyfile(worker.npz, dest / "fork.npz")
        initial_frame = api.Frame.read(worker.npz)
        official[0] = worker.current["audit"]
        for arm in ARMS:
            trackers[arm].start_soup(initial_frame, 0)
            decisions[arm].append(boundary(schedules[arm], worker.current, False))
        for expected in trace(actions):
            start, step = expected["start_step"], expected["step"]
            require(not worker.terminal() and step <= WINDOW and len(expected["actions"]) == STRIDE
                    and expected["target"] == SOUP and expected["instruction"] == SOURCE_PROMPT,
                    f"Noncanonical source block at {start}")
            commands = expected["actions"]
            # Never feed either hypothetical scheduler choice into the fixed replay.
            actual = worker.step(commands, expected["target"], expected["instruction"])
            current = worker.current
            require({"start_step": start, "target": SOUP, "instruction": SOURCE_PROMPT,
                     "actions": actual, **current} == expected,
                    f"Replay diverged at {step} (realized controls, fingerprint, audit or native transitions)")
            n += 1
            official[step] = current["audit"]
            combined.extend(actual)
            frame = api.Frame.read(worker.npz)
            half_events = {arm: [audited(e, current) for e in trackers[arm].observe_soup(frame, actual, step)]
                           for arm in ARMS}
            require([shadow_projection(e) for e in half_events["baseline"]]
                    == [shadow_projection(e) for e in half_events["candidate"]],
                    f"Shadow observers did not agree at {step}")
            for arm in ARMS:
                for e in half_events[arm]:
                    if e["event"] == "in_basket":
                        name = f"soup-commit-{step}.npz"
                        if not (dest / name).exists():
                            copyfile(worker.npz, dest / name)
                        e.update(evidence_file=name, evidence_sha256=sha(dest / name))
                    events[arm].append(e)
            if step % CHUNK == 0:
                for arm in ARMS:
                    e = trackers[arm].observe(frame, expected["target"], combined)
                    if e:
                        name = f"soup-commit-{step}.npz"
                        if not (dest / name).exists():
                            copyfile(worker.npz, dest / name)
                        events[arm].append(audited({**e, "writer": "legacy", "applied": True,
                                                    "evidence_file": name,
                                                    "evidence_sha256": sha(dest / name)}, current))
                    decisions[arm].append(boundary(schedules[arm], current, all(current["audit"]["goals"].values())))
                combined.clear()
        terminal = api.fingerprint(worker.current)
        terminal_audit = worker.current["audit"]
        done = worker.current["done"]
        require(not combined and terminal == row["terminal"] and terminal["step"] == WINDOW
                and n == WINDOW // STRIDE and terminal_audit["goals"] == row["terminal_goals"]
                and not worker.terminal() and done is False,
                "Incomplete replay or altered terminal")
    reproduced = dest / "actions.jsonl"
    require(list(trace(reproduced)) == list(trace(actions)), "Saved replay trace differs from validated source")
    return {"fork": fork, "terminal": terminal, "terminal_audit": terminal_audit, "action_steps": WINDOW,
            "source_done": done, "source_stop_reason": "diagnostic_cutoff",
            "original_episode_budget": BUDGET, "unexecuted_episode_steps": BUDGET - WINDOW,
            "replayed_samples": n, "actions": reproduced.name, "actions_sha256": sha(reproduced),
            "fork_npz_sha256": sha(dest / "fork.npz"), "fork_audit": official[0],
            "arms": {arm: {"events": events[arm], "schedule_boundaries": decisions[arm],
                           "terminal_progress": trackers[arm].status()} for arm in ARMS},
            **compare(decisions, events, official)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=GPU, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    require(source != out and not out.is_relative_to(source) and not source.is_relative_to(out)
            and out.name != "stage4-pilot", "Use a development output outside the sealed source and pilot")
    validate_gpu(args.card)
    bddl, old = validate_study(args.study.resolve())
    row, actions, source_manifest = sealed_source(source, args.card, bddl, old)
    layout, seed = SEALED[args.card]
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / f"target-progress-plan-{args.card}.json"
    marker = out / f"target-progress-{args.card}.json"
    dest = out / str(layout)
    require(not plan_path.exists() and not marker.exists() and not dest.exists(),
            "Refusing to repeat or replace this assigned replay")
    plan = {"schema": "v2-stage4-target-progress-plan/1", "card": args.card, "layout_seed": layout,
            "policy_seed": seed, "prefix": False, "motion": False, "scope": "development_only",
            "source_audit_sha256": AUDIT_SHA, "source_result_sha256": row["result_sha256"],
            "source_actions_sha256": row["actions_sha256"], "source_manifest": source_manifest,
            "source_sha256": sha(Path(__file__)), "bddl_sha256": sha(bddl), "steps": WINDOW,
            "observer_stride": STRIDE, "policy_chunk": CHUNK, "schedule_attempt": ATTEMPT,
            "source_target": SOUP, "source_prompt": SOURCE_PROMPT, "hypothetical_schedules": True}
    write(plan_path, plan)
    import run as api

    require(api.instruction(SOUP) == SOURCE_PROMPT, "Frozen soup instruction differs")
    report = {**plan, "schema": "v2-stage4-target-progress/1", "source": row, "record_dir": dest.name,
              "plan_sha256": sha(plan_path), **replay(api, bddl, row, actions, dest)}
    write(marker, report)
    print("TARGET_PROGRESS", layout, "valid=", report["valid_write"], "wrong=", report["wrong_writes"],
          "false_shadow=", report["wrong_candidate_observations"], "removed=", report["earliest_soup_removal"],
          "target_diff=", report["first_target_difference"], flush=True)


if __name__ == "__main__":
    main()
