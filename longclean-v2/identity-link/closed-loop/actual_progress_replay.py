"""Same-action development diagnostic of the frozen actual-object placement writer."""
import argparse
import importlib.util
from pathlib import Path
from shutil import copyfile
import sys

from stage4_fixture import GPU, code_lock, read, require, sha, validate_gpu, validate_study, write
from common import ATTEMPT, BUDGET, CHEESE, CHUNK, GOALS, PREFIX_TASK, SAUCE, SOUP
from legacy_parity import trace
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
PLACEMENT = HERE.parents[1] / "placement-next"
AUDIT_SHA = "0d00abc23a0264bb783084332b41ea91e538d76e6e992dad8f836d08592c2be3"
OBSERVER_SHA = "81d6b3b34f1e968093f3396a99d76d60eea3905e77b58f222dbc920134624dd2"
COMPARE_SHA = "7bc4215e57310facb19a2f3e2ac3a548ebeb904a1d2549f385f034d2171d499e"
PHYSICS_WORKER_SHA = "be5e073d13e30ad0fb0e0faaf9c4897404d4c80eddf8086ac65fd45a9db5cbff"
PHYSICS_SIMULATOR_SHA = "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8"
CASES = {2: ((931000, 46, False), (931200, 46, True)), 3: ((931005, 47, False), (931201, 47, True))}
ARMS = ("baseline", "candidate")
PROMPT = "put the alphabet soup can in the basket"
WINDOW = 360
SUBSTRIDE = 5


def checked(path, digest):
    require(path.is_file() and sha(path) == digest, f"Changed sealed artifact: {path}")
    return path


def load_exact(alias, path, digest):
    spec = importlib.util.spec_from_file_location(alias, checked(path, digest))
    require(spec is not None and spec.loader is not None, f"Cannot load sealed module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def source_card(root, card, bddl, old):
    audit = read(checked(root / "audit.json", AUDIT_SHA))
    require(audit["schema"] == "v2-stage4-target-follow-audit/1" and audit["pilot_release"] is False,
            "Expected the frozen development audit")
    smoke = read(checked(root.parent / "stage4-smoke" / "frozen.json", audit["smoke_freeze_sha256"]))
    require(code_lock(old) == smoke["code"], "Frozen candidate or evaluator source changed")
    checked(HERE / "target_follow.py", audit["runner_sha256"])
    checked(HERE / "target_follow_audit.py", audit["audit_sha256"])
    stamp, = [r for r in audit["manifests"] if r["card"] == card]
    manifest = read(checked(root / f"target-follow-{card}.json", stamp["manifest_sha256"]))
    plan = read(checked(root / f"target-follow-plan-{card}.json", stamp["plan_sha256"]))
    require(manifest["complete"] and manifest["plan_sha256"] == stamp["plan_sha256"]
            and plan["budget"] == WINDOW and plan["bddl_sha256"] == sha(bddl)
            and plan["policy_chunk"] == CHUNK and plan["observer_stride"] == SUBSTRIDE,
            "Wrong source factorial plan")
    result = []
    for layout, seed, fresh in CASES[card]:
        fields = {"card": card, "layout_seed": layout, "policy_seed": seed, "fresh": fresh,
                  "prefix": True, "prompt_id": "standard", "prompt": PROMPT}
        row, = [r for r in audit["records"] if all(r[k] == v for k, v in fields.items())]
        item, = [r for r in manifest["records"] if all(r[k] == v for k, v in fields.items())]
        setup, = [r for r in manifest["prefixes"] if r["layout_seed"] == layout]
        name = f"{layout}/cheese-prefix/standard/result.json"
        require(row["result"] == item["result"] == name and row["status"] == item["status"] == "complete"
                and row["result_sha256"] == item["result_sha256"] and row["category"] == "sauce_first"
                and row["terminal_reason"] == "budget" and row["suffix_steps"] == WINDOW,
                f"Unexpected frozen record: {layout}")
        current = checked(root / name, row["result_sha256"])
        saved = read(current)
        prefix_name = f"{layout}/prefix-setup/prefix.json"
        require(all(saved[k] == v for k, v in fields.items()) and saved["motion"] is False
                and setup["available"] is True and setup["prefix"] == saved["prefix_record"] == prefix_name
                and saved["prefix_sha256"] == setup["sha256"] and saved["actions"] == "actions.jsonl"
                and saved["actions_sha256"] == row["actions_sha256"]
                and saved["fork_npz_sha256"] == row["fork_npz_sha256"]
                and saved["initial"] == setup["initial"] == row["initial"]
                and saved["fork"] == setup["end"] == row["fork"]
                and saved["terminal_checkpoint"] == row["terminal"]
                and saved["terminal_goals"] == row["terminal_goals"],
                f"Source record or prefix metadata changed: {layout}")
        prefix = read(checked(root / prefix_name, setup["sha256"]))
        require(prefix["initial"] == row["initial"] and prefix["end"] == row["fork"]
                and setup["observed_cheese"] == prefix["observed_cheese"] is True
                and row["terminal"]["step"] == row["fork"]["step"] + WINDOW,
                f"No confirmed common cheese prefix: {layout}")
        observation(checked(current.parent / "fork.npz", row["fork_npz_sha256"]), row["fork"])
        actions = checked(current.parent / "actions.jsonl", row["actions_sha256"])
        result.append((row, prefix, actions, setup["sha256"]))
    return stamp, result


class ActualProgress:
    """Expose original prefix cheese plus the frozen observer's new placements."""
    def __init__(self, prefix, observer):
        self.prefix = prefix
        self.observer = observer

    def complete(self, obj):
        return self.prefix.complete(obj) if obj == CHEESE else obj in self.observer.complete

    def status(self):
        return {obj: "in_basket" if self.complete(obj) else "pending" for obj in GOALS}


def boundary(schedule, current):
    step = current["step"]
    before, prior, until = list(schedule.order), schedule.target, schedule.until
    target = schedule.next(step)
    return {"step": step, "target": target, "order": list(schedule.order), "order_before": before,
            "prior_target": prior, "until_before": until, "until": schedule.until,
            "official_success": all(current["audit"]["goals"].values())}


def event_row(event, current, assigned, writer, worker):
    obj = event["object"]
    applied = event["event"] == "in_basket"
    result = {**event, "step": current["step"], "object": obj, "assigned": assigned, "writer": writer,
              "applied": applied, "matches_audit": current["audit"]["goals"][obj],
              "held_soup": current["audit"]["held_soup"]}
    if applied:
        p = worker.path / f"commit-{current['step']}.npz"
        if not p.exists():
            copyfile(worker.npz, p)
        result.update(evidence_file=p.name, evidence_sha256=sha(p))
    return result


def join_grasp_audit(path, source, initial, event_rows):
    """Adjudicate after replay; neither any observer nor scheduler reads the sidecar."""
    import json

    with path.open(encoding="utf-8") as stream:
        truth = list(map(json.loads, stream))
    by_step = {}
    for native, physics in zip([initial, *source], truth, strict=True):
        step = native["step"]
        held = physics["held_by_target"]
        require(physics["step"] == step and physics["state_sha256"] == native["state_sha256"]
                and physics["goals"] == native["audit"]["goals"] and set(held) == set(GOALS)
                and all(type(v) is bool for v in held.values())
                and held[SOUP] == native["audit"]["held_soup"] and step not in by_step,
                f"Independent grasp sidecar differs from native source at {step}")
        by_step[step] = held
    for event in event_rows:
        step, obj = event["step"], event["object"]
        event.update(held_by_target=by_step[step][obj], released_in_audit=not by_step[step][obj],
                     previous_sampled_held_step=max((s for s, held in by_step.items() if s < step and held[obj]),
                                                   default=None))


def differences(decisions, events, timeline, end):
    pairs = [{"step": a["step"], "baseline": a, "candidate": b}
             for a, b in zip(decisions["baseline"], decisions["candidate"], strict=True)]
    official = {r["step"]: r["goals"] for r in timeline}
    target = next((p for p in pairs if p["step"] < end and p["baseline"]["target"] != p["candidate"]["target"]), None)
    eligibility = next((p for p in pairs if p["step"] < end
                        and set(p["baseline"]["order"]) != set(p["candidate"]["order"])), None)
    tail = pairs[-1]
    checkpoint = tail if tail["baseline"]["target"] != tail["candidate"]["target"] else None
    commits = []
    for e in events["candidate"]:
        if not (e["object"] == SAUCE and e["applied"] and e["matches_audit"]):
            continue
        step = e["step"]
        at = next(p for p in pairs if p["step"] == step)
        soup_entered = any(t["object"] == SOUP and t["in_basket"] and t["step"] <= step
                           for r in timeline for t in r["transitions"])
        if (step < end and not soup_entered and not official[step][SOUP]
                and SAUCE in at["baseline"]["order"] and SAUCE not in at["candidate"]["order"]):
            commits.append({"event": e, "boundary": at, "soup_pending": True,
                            "sauce_release_independently_verified": e["released_in_audit"]})
    return {"extra_sauce_before_soup": commits,
            "safe_extra_sauce_before_soup": [c for c in commits if c["sauce_release_independently_verified"]],
            "first_eligibility_difference": eligibility,
            "first_target_difference": target or checkpoint, "first_target_difference_within_source": target,
            "checkpoint_target_difference": checkpoint,
            "wrong_writes": {arm: sum(e["applied"] and not e["matches_audit"] for e in events[arm]) for arm in ARMS},
            "held_writes": {arm: sum(e["applied"] and e["held_by_target"] for e in events[arm]) for arm in ARMS},
            "unsafe_writes": {arm: sum(e["applied"] and (not e["matches_audit"] or e["held_by_target"])
                                       for e in events[arm]) for arm in ARMS}}


def replay(api, worker_type, observer_module, compare, bddl, row, prefix, actions, dest):
    locate = api.load_locator()
    old = api.Progress(locate)
    candidate_prefix = api.Progress(locate)
    candidate = ActualProgress(candidate_prefix, observer_module.PlacementObserver())
    trackers = {"baseline": old, "candidate": candidate}
    decisions, events = ({a: [] for a in ARMS} for _ in range(2))
    prefix_events, timeline = [], []
    accumulation, transitions = [], []
    previous = None
    source_rows = list(trace(actions))
    start, cut = row["fork"]["step"], row["terminal"]["step"]
    prefix_i, suffix_anchors = 0, 0
    schedules = {}
    with worker_type(bddl, dest, row["layout_seed"]) as worker:
        require(api.fingerprint(worker.current) == row["initial"] and not worker.terminal(),
                "Initial native state differs from the audited source")
        initial = worker.current
        for source in source_rows:
            s, step = source["start_step"], source["step"]
            in_prefix = s < start
            assigned = CHEESE if in_prefix else SOUP
            prompt = PREFIX_TASK if in_prefix else PROMPT
            require(not worker.terminal() and step <= cut
                    and source["target"] == assigned and source["instruction"] == prompt,
                    f"Wrong assigned target or trace bound at {s}")
            actual = worker.step(source["actions"], source["target"], source["instruction"])
            current = worker.current
            require({"start_step": s, "target": assigned, "instruction": prompt, "actions": actual, **current}
                    == source, f"Native replay differs at {step} (controls, state, observations, audit or transitions)")
            if in_prefix:
                chunk = prefix["chunks"][prefix_i]
                prefix_i += 1
                require(actual == chunk["actions"] and api.fingerprint(current) == chunk["end"],
                        f"Frozen cheese prefix differs at {step}")
                frame = api.Frame.read(worker.npz)
                e = old.observe(frame, assigned, actual)
                require(e == candidate_prefix.observe(frame, assigned, actual)
                        and old.complete(CHEESE) == candidate_prefix.complete(CHEESE) == chunk["cheese_observed"]
                        and current["audit"]["goals"][CHEESE] == chunk["cheese_truth"],
                        f"Prefix observer differs at {step}")
                if e:
                    aud = event_row(e, current, assigned, "legacy", worker)
                    require(aud["matches_audit"], "False official-predicate write in the preexisting cheese prefix")
                    prefix_events.append(aud)
                if step == start:
                    require(prefix_i == len(prefix["chunks"]) and old.complete(CHEESE)
                            and api.fingerprint(current) == row["fork"]
                            and current["audit"]["goals"] == {o: o == CHEESE for o in GOALS},
                            "Invalid common fork after cheese")
                    fork_audit = current["audit"]
                    copyfile(worker.npz, dest / "fork.npz")
                    candidate.observer.clear_view = frame
                    schedules = {arm: api.Schedule(trackers[arm], True, None) for arm in ARMS}
                    for arm in ARMS:
                        decisions[arm].append(boundary(schedules[arm], current))
                continue
            require(schedules and len(actual) == SUBSTRIDE, f"Wrong source half stride at {step}")
            accumulation.extend(actual)
            transitions.extend(current["transitions"])
            suffix_anchors += 1
            if (step - start) % CHUNK:
                continue
            require(len(accumulation) == CHUNK, f"Missing 10-control block at {step}")
            frame = api.Frame.read(worker.npz)
            prior = old.observe(frame, assigned, accumulation)
            if prior:
                events["baseline"].append(event_row(prior, current, assigned, "legacy", worker))
            for e in candidate.observer.observe(frame, accumulation, step):
                events["candidate"].append(event_row(e, current, assigned, "placement_next", worker))
            goals = current["audit"]["goals"]
            # Same release-probe definition and observation cadence as the placement-next comparison.
            probe = (previous is not None and previous.gap < .069 and frame.gap >= .067
                     and any(a[-1] < -.5 for a in accumulation) and observer_module.near_basket(frame.geometry["eef"]))
            timeline.append({"step": step, "assigned": assigned, "goals": goals, "all_goals": all(goals.values()),
                             "transitions": list(transitions), "release_probe": probe})
            previous = frame
            for arm in ARMS:
                decisions[arm].append(boundary(schedules[arm], current))
            accumulation.clear()
            transitions.clear()
        end = api.fingerprint(worker.current)
        terminal_audit = worker.current["audit"]
        done = worker.current["done"]
        require(not accumulation and not transitions and end == row["terminal"] and end["step"] == cut
                and suffix_anchors == WINDOW // SUBSTRIDE and len(timeline) == WINDOW // CHUNK
                and terminal_audit["goals"] == row["terminal_goals"] and done is False and not worker.terminal(),
                "Incomplete or terminal replay")
    reproduced = dest / "actions.jsonl"
    require(list(trace(reproduced)) == source_rows, "Saved combined trace differs from audited raw source")
    grasp_audit = dest / "grasp-audit.jsonl"
    join_grasp_audit(grasp_audit, source_rows, initial, prefix_events + events["baseline"] + events["candidate"])
    return {"fork": row["fork"], "fork_audit": fork_audit, "fork_npz_sha256": sha(dest / "fork.npz"),
            "terminal": end, "terminal_audit": terminal_audit, "source_done": done,
            "source_stop_reason": "diagnostic_cutoff", "original_episode_budget": BUDGET,
            "unexecuted_episode_steps": BUDGET - cut, "prefix_action_steps": start, "action_steps": WINDOW,
            "total_action_steps": cut, "replayed_samples": len(source_rows), "suffix_anchors": suffix_anchors,
            "actions": reproduced.name, "actions_sha256": sha(reproduced), "prefix_events": prefix_events,
            "grasp_audit": grasp_audit.name, "grasp_audit_sha256": sha(grasp_audit),
            "arms": {arm: {"events": events[arm], "schedule_boundaries": decisions[arm],
                           "terminal_progress": trackers[arm].status(),
                           "placement_score": compare.score(timeline, [e for e in events[arm] if e["applied"]],
                                                            observer_module.TARGETS)}
                     for arm in ARMS},
            "negative_releases": compare.negative_releases(timeline, observer_module.TARGETS),
            **differences(decisions, events, timeline, cut)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=GPU, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    require(source != out and not out.is_relative_to(source) and not source.is_relative_to(out)
            and out.name != "stage4-pilot", "Keep development outputs out of the sealed source and pilot")
    validate_gpu(args.card)
    bddl, old = validate_study(args.study.resolve())
    manifest_stamp, assigned = source_card(source, args.card, bddl, old)
    observer = load_exact("frozen_actual_observer", PLACEMENT / "observer.py", OBSERVER_SHA)
    compare = load_exact("frozen_actual_compare", PLACEMENT / "compare.py", COMPARE_SHA)
    physics = load_exact("actual_progress_physics_worker", HERE / "actual_physics_worker.py", PHYSICS_WORKER_SHA)
    checked(HERE / "actual_physics_simulator.py", PHYSICS_SIMULATOR_SHA)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / f"actual-progress-plan-{args.card}.json"
    marker = out / f"actual-progress-{args.card}.json"
    require(not plan_path.exists() and not marker.exists()
            and all(not (out / str(row["layout_seed"])).exists() for row, *_ in assigned),
            "Refusing to repeat or replace a sealed development attempt")
    plan = {"schema": "v2-stage4-actual-progress-plan/1", "card": args.card, "scope": "development_only",
            "source_audit_sha256": AUDIT_SHA, "source_manifest": manifest_stamp, "bddl_sha256": sha(bddl),
            "source_sha256": sha(Path(__file__)), "observer_sha256": OBSERVER_SHA, "compare_sha256": COMPARE_SHA,
            "prefix": True, "motion": False, "policy_chunk": CHUNK, "source_stride": SUBSTRIDE,
            "observer_stride": CHUNK, "schedule_attempt": ATTEMPT, "steps": WINDOW,
            "source_target": SOUP, "source_prompt": PROMPT, "hypothetical_schedules": True,
            "sauce_release_audit_available": True, "physics_worker_sha256": PHYSICS_WORKER_SHA,
            "physics_simulator_sha256": PHYSICS_SIMULATOR_SHA,
            "cases": [{"layout_seed": row["layout_seed"], "policy_seed": row["policy_seed"],
                       "source_result_sha256": row["result_sha256"], "source_actions_sha256": row["actions_sha256"],
                       "prefix_sha256": prefix_sha} for row, _, _, prefix_sha in assigned]}
    write(plan_path, plan)
    manifest = {"schema": "v2-stage4-actual-progress-manifest/1", "card": args.card,
                "plan_sha256": sha(plan_path), "source_audit_sha256": AUDIT_SHA, "complete": False, "records": []}
    write(marker, manifest)
    import run as api

    require(api.instruction(SOUP) == PROMPT, "The frozen soup wording changed")
    worker_type = physics.worker_class(api)
    for row, prefix, actions, prefix_sha in assigned:
        layout = row["layout_seed"]
        dest = out / str(layout)
        result = {**{k: v for k, v in plan.items() if k != "cases"}, "schema": "v2-stage4-actual-progress/1",
                  "layout_seed": layout, "policy_seed": row["policy_seed"], "record_dir": dest.name,
                  "prefix_sha256": prefix_sha, "source_actions_sha256": row["actions_sha256"],
                  "source": row, "plan_sha256": manifest["plan_sha256"],
                  **replay(api, worker_type, observer, compare, bddl, row, prefix, actions, dest)}
        report = dest / "report.json"
        write(report, result)
        manifest["records"].append({"layout_seed": layout, "report": report.relative_to(out).as_posix(),
                                    "report_sha256": sha(report)})
        write(marker, manifest)
        print("ACTUAL_PROGRESS", layout, "correct_extra_sauce=", len(result["extra_sauce_before_soup"]),
              "false_predicates=", result["wrong_writes"], flush=True)
    manifest["complete"] = True
    write(marker, manifest)


if __name__ == "__main__":
    main()
