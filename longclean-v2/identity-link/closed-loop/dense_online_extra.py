"""Three-arm development comparison of actual-object tracking and release sampling."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from stage4_fixture import GPU, validate_gpu, validate_study
from common import BUDGET, CHEESE, CHUNK, GOALS, PREFIX_TASK, SOUP, read, sha, write


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("baseline", "actual10", "adaptive1")
LAYOUTS = tuple(range(931304, 931308))
ADAPTIVE_SHA = "839db7d67af2d67eb9fe2c2dd6f86cc2a3723f034fe3b5a0e97b58152d4a1c27"
PLACEMENT_SHA = "81d6b3b34f1e968093f3396a99d76d60eea3905e77b58f222dbc920134624dd2"
CONTROLS_PLAN_SHA = "e8014407a15daa4c2a2cdccb3482bdfad7bc3a4f5ce42f10fbd05dc2f3559e17"
SOURCE_SHA = {
    "dense_release_observer.py": ADAPTIVE_SHA,
    "actual_progress_replay.py": "f0c1000074559a2eb48859cd3ff44c22d123b0650d7dd962a9fe44ba05909bb2",
    "actual_physics_worker.py": "be5e073d13e30ad0fb0e0faaf9c4897404d4c80eddf8086ac65fd45a9db5cbff",
    "actual_physics_simulator.py": "8d5883ac2e45bfe7b85daabded1928caea80bc5fbbe19b7a564e5cda878557d8",
    "../../placement-next/observer.py": PLACEMENT_SHA,
}
ORDINARY_SLICE = 5
SAFETY_MARKER = "adaptive-safety-stop.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def planned(card):
    rows = []
    for index, layout in enumerate(LAYOUTS):
        if layout % 2 != card - 2:
            continue
        offset = index % len(ARMS)
        rows.append({"layout_seed": layout, "policy_seed": 48 + layout % 2, "card": card,
                     "order": list(ARMS[offset:] + ARMS[:offset])})
    return rows


def pinned(path, digest):
    require(path.is_file() and sha(path) == digest, f"Required evidence changed: {path}")
    return read(path)


def validate_evidence(regression, regression_sha, controls, controls_sha):
    for name, digest in SOURCE_SHA.items():
        require(sha(HERE / name) == digest, f"Required observer or evaluator source changed: {name}")
    r = pinned(regression, regression_sha)
    require(r["schema"] == "v2-dense-release-regression/1" and r["wrapper_sha256"] == ADAPTIVE_SHA
            and r["observer_sha256"] == PLACEMENT_SHA and r["expected_archived_signatures_match"] is True
            and r["incorrect_or_held_writes_in_archived_prefixes"] == 0
            and r["source_sha256"] == sha(HERE / "dense_release_regression.py"), "Required frozen regression did not pass")
    require(sorted(c["layout_seed"] for c in r["cases"]) == [931000, 931005, 931200, 931201],
            "Regression cases changed")
    commits = [e for c in r["cases"] for e in c["candidate_events"] if e["event"] == "in_basket"]
    require(len(commits) == 3 and all(e["predicate_true"] is True and e["held_by_target"] is False for e in commits),
            "Expected all three archived candidate writes to be true and unheld")
    c = pinned(controls, controls_sha)
    require(c["schema"] == "v2-sauce-controls-audit/1" and c["candidate_sha256"] == ADAPTIVE_SHA
            and c["plan_sha256"] == CONTROLS_PLAN_SHA and sorted(c["layouts_checked"]) == list(range(929336, 929352))
            and c["wrong_or_held_candidate_commits"] == 0 and c["development_online_ready"] is True,
            "The full archived sauce-control safety gate did not pass")
    return {"regression_sha256": regression_sha, "regression_source_sha256": r["source_sha256"],
            "controls_audit_sha256": controls_sha, "controls_plan_sha256": CONTROLS_PLAN_SHA}


def physical_sample(stream, state):
    line = stream.readline()
    require(bool(line), f"Missing evaluator-only grasp sample at {state['step']}")
    sample = json.loads(line)
    grasp = sample["held_by_target"]
    require(sample["step"] == state["step"] and sample["state_sha256"] == state["state_sha256"]
            and sample["goals"] == state["audit"]["goals"] and set(grasp) == set(GOALS)
            and all(type(v) is bool for v in grasp.values()) and grasp[SOUP] is state["audit"]["held_soup"],
            f"Evaluator-only grasp sample disagrees with native state at {state['step']}")
    return sample


def event_records(events, state, truth, assigned, arm, dest, seen_held):
    records = []
    for event in events:
        obj = event["object"]
        commit = event["event"] == "in_basket"
        held = truth["held_by_target"][obj]
        goal = truth["goals"][obj]
        row = {**event, "step": state["step"], "assigned": assigned, "writer": arm, "applied": commit,
               "matches_audit": goal, "held_by_target": held, "previous_sampled_held_step": seen_held.get(obj),
               "physically_confirmed": goal and not held, "noninstruction_target": obj != assigned}
        if commit:
            evidence = dest / f"commit-{state['step']}.npz"
            if not evidence.exists():
                copyfile(dest / "observation.npz", evidence)
            row.update(evidence_file=evidence.name, evidence_sha256=sha(evidence))
        records.append(row)
    seen_held.update({obj: state["step"] for obj, held in truth["held_by_target"].items() if held})
    return records


def saved_prefix(api, worker, stream, prefix, tracker, dest, seen_held):
    """Replay the frozen prefix and retain exact placement events for audit."""
    require(api.fingerprint(worker.current) == prefix["initial"], "Shared layout initial fingerprint differs")
    start_truth = physical_sample(stream, worker.current)
    require(not any(start_truth["goals"].values()), "A goal was true at the initial state")
    events = []
    truth = start_truth
    for chunk in prefix["chunks"]:
        actual = worker.step(chunk["actions"], CHEESE, PREFIX_TASK)
        state = worker.current
        require(actual == chunk["actions"] and api.fingerprint(state) == chunk["end"], "Shared cheese prefix diverged")
        event = tracker.observe(api.Frame.read(worker.npz), CHEESE, actual)
        require(tracker.complete(CHEESE) is chunk["cheese_observed"], "Shared prefix observer changed")
        truth = physical_sample(stream, state)
        require(truth["goals"][CHEESE] is chunk["cheese_truth"] and (not event or truth["goals"][CHEESE]),
                "Shared prefix goal or confirmation differs from the frozen trace")
        events.extend(event_records([event] if event else [], state, truth, CHEESE, "prefix_legacy", dest, seen_held))
    require(api.fingerprint(worker.current) == prefix["end"] and tracker.complete(CHEESE)
            and truth["goals"] == {obj: obj == CHEESE for obj in GOALS} and not worker.terminal(),
            "Shared fork is not the prespecified cheese-only state")
    return events, truth


def safety_stop(marker, case, events):
    """The evaluator may stop unsafe testing without supplying truth to a policy."""
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"layout_seed": case["layout_seed"], "card": case["card"], "events": events}, stream,
                      indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        pass


def reason_at_terminal(state):
    if all(state["audit"]["goals"].values()):
        return "goal"
    return "environment_done" if state["done"] else "budget"


def run_arm(api, worker_type, tracker_type, frozen, adaptive_type, policy, locate, bddl, case, arm, folder, stop):
    """Run one branch. No change to an issued block is made on a progress event."""
    from actual_progress_replay import boundary

    begin = perf_counter()
    legacy = api.Progress(locate)
    adaptive = adaptive_type() if arm == "adaptive1" else None
    passive = frozen.PlacementObserver() if arm == "actual10" else adaptive
    tracker = legacy if arm == "baseline" else tracker_type(legacy, passive)
    prefix = read(folder / "prefix.json")
    dest = folder.parent.parent / "episodes" / str(case["layout_seed"]) / arm
    decisions, chunks, events, repeats = [], [], [], []
    seen_held = {}
    native_regrasps = 0
    reason, owned_gpu = None, None
    with worker_type(bddl, dest, case["layout_seed"]) as worker, (dest / "grasp-audit.jsonl").open(encoding="utf-8") as grasp:
        prefix_events, truth = saved_prefix(api, worker, grasp, prefix, legacy, dest, seen_held)
        fork = api.fingerprint(worker.current)
        copyfile(worker.npz, dest / "fork.npz")
        observation = api.Frame.read(worker.npz)
        if adaptive is not None:
            adaptive.start(observation, fork["step"])
        elif passive is not None:
            passive.clear_view = observation
        policy.reset(case["policy_seed"])
        schedule = api.Schedule(tracker, allow_reorder=False, before=None)
        while not worker.terminal():
            if adaptive is not None and stop.exists():
                reason = "adaptive_safety_stop_peer"
                break
            step = worker.current["step"]
            require(step % CHUNK == 0, f"Unexpected policy decision endpoint: {step}")
            decision = boundary(schedule, worker.current)
            decisions.append({**decision, "goals_before": truth["goals"], "held_before": truth["held_by_target"]})
            obj = decision["target"]
            if obj is None:
                reason = "observed_complete_incomplete"
                break
            if truth["goals"][obj]:
                repeats.append({"step": step, "target": obj, "held_at_selection": truth["held_by_target"][obj]})
            prompt = api.instruction(obj)
            issued = policy.act(worker, prompt, limit=CHUNK)
            require(len(issued) == CHUNK, f"Unequal policy chunk at {step}")
            owned_gpu = policy.placement
            actual, samples = [], []
            while len(actual) < len(issued):
                if adaptive is not None and stop.exists():
                    reason = "adaptive_safety_stop_peer"
                    break
                dense = bool(adaptive is not None and adaptive.needs_dense)
                width = min(1 if dense else ORDINARY_SLICE, len(issued) - len(actual))
                first = len(actual)
                used = worker.step(issued[first:first + width], obj, prompt)
                require(used == issued[first:first + len(used)] and 0 < len(used) <= width, "Realized OSC controls changed")
                actual.extend(used)
                current = worker.current
                endpoint = current["step"]
                native_regrasps += current["completed_regrasp_steps"]
                samples.append({"step": endpoint, "steps_executed": len(used), "dense_requested": dense})
                frame = api.Frame.read(worker.npz)
                returned = []
                if adaptive is not None:
                    returned = adaptive.observe(frame, used, endpoint)
                elif endpoint % CHUNK == 0:
                    if passive is not None:
                        returned = passive.observe(frame, actual, endpoint)
                    else:
                        event = legacy.observe(frame, obj, actual)
                        returned = [event] if event else []
                truth = physical_sample(grasp, current)
                captured = event_records(returned, current, truth, obj, arm, dest, seen_held)
                events.extend(captured)
                unsafe = [e for e in captured if e["applied"] and not e["physically_confirmed"]]
                if adaptive is not None and unsafe:
                    safety_stop(stop, case, unsafe)
                    reason = "adaptive_safety_stop"
                    break
                if worker.terminal():
                    break
            chunks.append({"start_step": step, "end_step": worker.current["step"], "target": obj, "prompt": prompt,
                           "issued_actions": issued, "executed_actions": actual, "samples": samples})
            if reason is not None:
                break
        terminal = worker.current
        require(not grasp.readline(), "The simulator wrote an unexpected extra grasp sample")
        reason = reason or reason_at_terminal(terminal)
        official = all(terminal["audit"]["goals"].values())
        observed = tracker.status()
    unqualified = reason.startswith("adaptive_safety_stop") or (reason == "observed_complete_incomplete")
    result = {"schema": "v2-dense-online-arm/1", **{k: case[k] for k in ("layout_seed", "policy_seed", "card")},
              "arm": arm, "prefix_sha256": sha(folder / "prefix.json"), "motion": False,
              "budget": BUDGET, "initial": prefix["initial"], "fork": fork, "terminal": api.fingerprint(terminal),
              "terminal_reason": reason, "official_success": official, "qualified_success": official and not unqualified,
              "terminal_native_audit": terminal["audit"], "terminal_held_by_target": truth["held_by_target"],
              "terminal_progress": observed, "prefix_events": prefix_events, "state_events": events,
              "incorrect_predicate_commits": [e for e in events if e["applied"] and not e["matches_audit"]],
              "still_held_commits": [e for e in events if e["applied"] and e["held_by_target"]],
              "schedule_decisions": decisions, "policy_chunks": chunks, "completed_reassigned_chunks": repeats,
              "completed_regrasp_steps": native_regrasps, "after_fork_steps": terminal["step"] - fork["step"],
              "policy_calls": len(chunks), "single_control_observations_requested": sum(
                  s["dense_requested"] for c in chunks for s in c["samples"]),
              "actions_sha256": sha(dest / "actions.jsonl"), "grasp_audit_sha256": sha(dest / "grasp-audit.jsonl"),
              "fork_npz_sha256": sha(dest / "fork.npz"), "gpu_placement": owned_gpu,
              "elapsed_seconds": round(perf_counter() - begin, 6)}
    path = dest / "result.json"
    write(path, result)
    return path, result


def prepare_prefix(api, policy, bddl, case, folder):
    folder.mkdir(parents=True, exist_ok=False)
    layout = case["layout_seed"]
    prefix = api.make_prefix(policy, bddl, folder, layout)
    require(prefix == read(folder / "prefix.json") and prefix["initial"]["step"] == 0, f"Changed generated prefix: {layout}")
    prefix_end = prefix["end"]["step"]
    actions = folder / "prefix" / "actions.jsonl"
    tail = None
    with actions.open(encoding="utf-8") as stream:
        for line in stream:
            tail = json.loads(line)
    goals = tail["audit"]["goals"] if tail is not None else None
    qualified = ("failure" not in prefix and prefix["observed_cheese"] is True and bool(prefix["chunks"])
                 and prefix_end % CHUNK == 0 and prefix_end < BUDGET
                 and prefix["chunks"][-1]["end"] == prefix["end"] and prefix["chunks"][-1]["cheese_truth"] is True
                 and tail is not None and api.fingerprint(tail) == prefix["end"] and tail["done"] is False
                 and goals == {obj: obj == CHEESE for obj in GOALS})
    record = {"layout_seed": layout, "qualified": qualified,
              "reason": None if qualified else prefix.get("failure", "common_cheese_prefix_unavailable"),
              "prefix_seed": 60, "prefix_sha256": sha(folder / "prefix.json"), "initial": prefix["initial"],
              "fork": prefix["end"], "terminal_official_goals": goals,
              "prefix_checkpoint_npz_sha256": sha(folder / "checkpoint.npz"), "prefix_actions_sha256": sha(actions)}
    write(folder / "fixture.json", record)
    return record


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--card", required=True, type=int, choices=GPU)
    cli.add_argument("--study", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--regression", required=True, type=Path)
    cli.add_argument("--regression-sha", required=True)
    cli.add_argument("--controls", required=True, type=Path)
    cli.add_argument("--controls-sha", required=True)
    cli.add_argument("--preflight", action="store_true", help="Validate source and gate evidence without opening a worker")
    a = cli.parse_args()
    out, study = a.output.resolve(), a.study.resolve()
    require(out.name not in ("stage4-pilot", "stage4-smoke", "stage4-release-dense", "stage4-sauce-controls")
            and not any(p.resolve().is_relative_to(out) for p in (a.regression, a.controls)),
            "Use a separate development output directory")
    bddl, old = validate_study(study)
    gates = validate_evidence(a.regression, a.regression_sha, a.controls, a.controls_sha)
    if a.preflight:
        print("DENSE_ONLINE_PREFLIGHT", a.card, planned(a.card), gates, flush=True)
        return
    validate_gpu(a.card)
    cases = planned(a.card)
    assigned = [str(c["layout_seed"]) for c in cases]
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"dense-online-plan-{a.card}.json"
    mark = out / f"dense-online-{a.card}.json"
    stop = out / SAFETY_MARKER
    require(not plan_file.exists() and not mark.exists()
            and all(not (out / part / layout).exists() for part in ("fixtures", "episodes") for layout in assigned),
            "An assigned development attempt already exists")
    code = {**SOURCE_SHA, "dense_online_extra.py": sha(Path(__file__)),
            "dense_release_regression.py": gates["regression_source_sha256"],
            **{f"../../{k}": v for k, v in old["code"].items()}}
    plan = {"schema": "v2-dense-online-plan/1", "scope": "new_development_layouts", "card": a.card,
            "cases": cases, "motion": False, "budget": BUDGET, "policy_chunk": CHUNK,
            "normal_slice": ORDINARY_SLICE, "adaptive_pending_slice": 1,
            "bddl_sha256": sha(bddl), "study4_sha256": sha(study / "frozen.json"),
            "gate_evidence": gates, "code": code, "release_original_pilot": False}
    write(plan_file, plan)
    manifest = {"schema": "v2-dense-online-manifest/1", "card": a.card, "plan_sha256": sha(plan_file),
                "complete": False, "prefixes": [], "episodes": [], "adaptive_wrong_commit": False,
                "release_original_pilot": False}
    write(mark, manifest)

    import run as api
    from actual_physics_worker import worker_class
    from actual_progress_replay import ActualProgress
    from dense_release_observer import DenseReleaseObserver, frozen_module

    worker_type = worker_class(api)
    policy, locate = api.Pi05(GPU[a.card]), api.load_locator()
    seen_raw = set()
    for case in cases:
        layout = str(case["layout_seed"])
        folder = out / "fixtures" / layout
        fixture = prepare_prefix(api, policy, bddl, case, folder)
        raw = fixture["initial"]["raw_sha256"]
        require(raw not in seen_raw, "Two assigned layouts share the same native initialization")
        seen_raw.add(raw)
        manifest["prefixes"].append({**fixture, "fixture": (folder / "fixture.json").relative_to(out).as_posix(),
                                     "fixture_sha256": sha(folder / "fixture.json")})
        write(mark, manifest)
        print("DENSE_ONLINE_PREFIX", layout, fixture["qualified"], fixture["reason"], flush=True)
        for arm in case["order"]:
            item = {"layout_seed": case["layout_seed"], "policy_seed": case["policy_seed"], "arm": arm}
            if not fixture["qualified"] or (arm == "adaptive1" and stop.exists()):
                status = "prefix_unavailable" if not fixture["qualified"] else "adaptive_safety_stop_skipped"
                item.update(status=status, result=None, result_sha256=None)
            else:
                path, record = run_arm(api, worker_type, ActualProgress, frozen_module(), DenseReleaseObserver,
                                       policy, locate, bddl, case, arm, folder, stop)
                item.update(status=record["terminal_reason"], official_success=record["official_success"],
                            qualified_success=record["qualified_success"], result=path.relative_to(out).as_posix(),
                            result_sha256=sha(path))
                manifest["adaptive_wrong_commit"] |= record["terminal_reason"] == "adaptive_safety_stop"
            manifest["episodes"].append(item)
            write(mark, manifest)
            print("DENSE_ONLINE_EPISODE", layout, arm, item["status"], item.get("qualified_success"), flush=True)
    manifest["complete"] = True
    manifest["adaptive_safety_stop_present"] = stop.exists()
    write(mark, manifest)


if __name__ == "__main__":
    main()
