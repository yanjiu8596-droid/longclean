"""Check a sensor-based stall rule against sealed native action traces."""

import argparse
from itertools import combinations
from math import dist, isfinite
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import CHEESE, CHUNK, GOALS, SAUCE, SOUP, read, sha, write
from dense_online import SOURCE_SHA
from dense_online_diagnostic import checked, jsonl, load_arm
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
SOURCES = {
    "stage4-dense-online": "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4",
    "stage4-dense-online-extra": "4b01c1d4ae89b729a529937427b331ba1c74722a9601d7b948a261d5f3f92e8d",
}
LAYOUTS = (931301, 931303, 931305, 931307)
CONTROL = 931301
CUTOFF, WINDOW, COUNT = 400, 130, 3
LOWER, UPPER = (-.105, .15, .62), (.11, .37, .85)
GAP, MAX_SPREAD = .075, .015


def source_arm(source_root, layout):
    tag = "stage4-dense-online" if layout in (931301, 931303) else "stage4-dense-online-extra"
    root = source_root / tag
    audit_digest = SOURCES[tag]
    audit = read(checked(root, "audit.json", audit_digest))
    require(audit["schema"] == "v2-dense-online-audit/1" and audit["valid_native_terminal_records"] == 12
            and audit["adaptive_safety_stop"] is None, "Incomplete or unexpected archived experiment")
    entry, = [r for r in audit["records"] if r["layout_seed"] == layout and r["arm"] == "adaptive1"]
    arm = load_arm(root, entry)
    result = arm["result"]
    require(result["terminal_reason"] == "budget" and CUTOFF in arm["endpoint"], "Missing native source at cutoff")
    prefix = read(checked(root, f"fixtures/{layout}/prefix.json", result["prefix_sha256"]))
    require(prefix["end"] == result["fork"] and prefix["observed_cheese"] is True, "Changed observed cheese prefix")
    arm.update(prefix=prefix, source_audit_sha256=audit_digest, source_tag=tag)
    return arm


def verify_sources():
    for name, digest in SOURCE_SHA.items():
        require(sha(HERE / name) == digest, f"Changed frozen module: {name}")


def replay_row(worker, original):
    require(worker.current["step"] == original["start_step"], "Unpaired native action interval")
    used = worker.step(original["actions"], original["target"], original["instruction"])
    require(used == original["actions"] and
            {**worker.current, **{k: original[k] for k in ("start_step", "target", "instruction", "actions")}} == original,
            f"Native state, observation or transition changed at {original['step']}")
    return used


def replay_rows(arm, cutoff):
    require(cutoff in arm["endpoint"], "Cutoff is not an archived native endpoint")
    return [r for r in arm["native"] if r["step"] <= cutoff]


def archive(api, worker, folder, name):
    path = folder / name
    copyfile(worker.npz, path)
    fingerprint = api.fingerprint(worker.current)
    observation(path, fingerprint)
    return {"file": name, "sha256": sha(path), "fingerprint": fingerprint}


def sensor_position(frame):
    position = [float(v) for v in frame.geometry["eef"]]
    require(len(position) == 3 and all(map(isfinite, position)) and isfinite(frame.gap), "Invalid proprioception")
    return position


def sensor_sample(step, frame, target, progress, adaptive, previous):
    position = sensor_position(frame)
    coarse = adaptive._coarse
    vote = adaptive._frozen.grasp_vote(frame)
    checks = {
        "remaining_target_is_soup": target == SOUP and not progress.complete(SOUP),
        "cheese_and_sauce_observed_complete": progress.complete(CHEESE) and progress.complete(SAUCE),
        "open_gap": frame.gap >= GAP,
        "above_basket": all(lo <= v <= hi for lo, v, hi in zip(LOWER, position, UPPER, strict=True)),
        "no_pending_release": coarse.pending is None,
        "no_transport_votes": all(obj is None for _, obj, _ in coarse.votes),
        "no_local_can_vote": vote is None,
    }
    sample = {"step": step, "eef_m": position, "gap_m": frame.gap, "target": target,
              "check": checks, "local_can_vote": vote}
    window = [*previous[-(COUNT - 1):], sample]
    consecutive = [x["step"] for x in window] == list(range(step - (COUNT - 1) * CHUNK, step + 1, CHUNK))
    spread = max((dist(a["eef_m"], b["eef_m"]) for a, b in combinations(window, 2)), default=0.)
    base_checks = [key for key in checks if key != "no_local_can_vote"]
    base = consecutive and spread <= MAX_SPREAD and all(x["check"][key] for x in window for key in base_checks)
    sample.update(window_steps=[x["step"] for x in window], max_window_spread_m=spread,
                  base_trigger=base, trigger=base and all(x["check"]["no_local_can_vote"] for x in window))
    return sample


def replay(api, worker_type, bddl, layout, out, arm):
    from actual_progress_replay import ActualProgress
    from dense_release_observer import DenseReleaseObserver

    source_result, prefix = arm["result"], arm["prefix"]
    source_rows = replay_rows(arm, CUTOFF)
    prefix_end = source_result["fork"]["step"]
    legacy = api.Progress(api.load_locator())
    adaptive = DenseReleaseObserver()
    progress = ActualProgress(legacy, adaptive)
    samples, events = [], []
    sauce_step = None
    with worker_type(bddl, out, layout) as worker:
        require(api.fingerprint(worker.current) == source_result["initial"], "Initial state changed")
        start = archive(api, worker, out, "initial.npz")
        home = sensor_position(api.Frame.read(worker.npz))
        for original in source_rows:
            step = original["step"]
            used = replay_row(worker, original)
            frame = api.Frame.read(worker.npz)
            if step <= prefix_end:
                chunk, = [c for c in prefix["chunks"] if c["end"]["step"] == step]
                legacy.observe(frame, CHEESE, used)
                require(legacy.complete(CHEESE) is chunk["cheese_observed"], "Frozen cheese observer changed")
                if step != prefix_end:
                    continue
                require(progress.complete(CHEESE), "No observed cheese confirmation")
                adaptive.start(frame, step)
                schedule = api.Schedule(progress, allow_reorder=False, before=None)
            else:
                for event in adaptive.observe(frame, used, step):
                    events.append({**event, "step": step})
                    if event["object"] == SAUCE and event["event"] == "in_basket" and sauce_step is None:
                        sauce_step = step
            if step % CHUNK:
                continue
            target = schedule.next(step)
            saved = arm["decisions"][step]
            require(target == saved["target"] and list(schedule.order) == saved["order"]
                    and schedule.until == saved["until"], f"Frozen scheduler changed at {step}")
            if sauce_step is not None and sauce_step <= step <= sauce_step + WINDOW:
                row = sensor_sample(step, frame, target, progress, adaptive, samples)
                row["observation"] = archive(api, worker, out, f"boundary-{step}.npz")
                samples.append(row)
        terminal = api.fingerprint(worker.current)
    expected_events = [e for e in source_result["state_events"] if e["step"] <= CUTOFF]
    require(len(events) == len(expected_events) and all(
        all(expected.get(k) == value for k, value in event.items()) for event, expected in zip(events, expected_events)),
        "Frozen observation event stream changed")
    require(jsonl(out / "actions.jsonl") == source_rows, "Saved action rows differ from archived evidence")
    physical = jsonl(out / "grasp-audit.jsonl")
    require(physical == [arm["physical"][s] for s in (0, *[r["step"] for r in source_rows])], "Grasp sidecar changed")
    require(sauce_step is not None and sauce_step + WINDOW <= CUTOFF, "Release window was not fully observed")
    first = next((x for x in samples if x["trigger"]), None)
    held = {x["step"]: x for x in physical}
    truth = None if first is None else [held[s] for s in first["window_steps"]]
    first_window_truth_ok = bool(truth and all(not any(r["held_by_target"].values())
                                and r["goals"] == {obj: obj != SOUP for obj in GOALS} for r in truth))
    return {"schema": "v2-basket-stall-case/1", "layout_seed": layout, "source": arm["source_tag"],
            "source_audit_sha256": arm["source_audit_sha256"], "source_result_sha256": arm["entry"]["result_sha256"],
            "source_policy_seed": source_result["policy_seed"], "source_actions_sha256": source_result["actions_sha256"],
            "first_observed_sauce_commit": sauce_step, "initial": start, "initial_eef_m": home,
            "replay_end": terminal, "exact_replay_rows": len(source_rows),
            "replayed_actions_sha256": sha(out / "actions.jsonl"), "replayed_grasp_sha256": sha(out / "grasp-audit.jsonl"),
            "observed_state_events": events, "samples": samples, "first_trigger": first,
            "evaluation_only": {"first_trigger_window_physics": truth,
                                "truth_admits_first_trigger_window": first_window_truth_ok,
                                "original_first_sampled_soup_grasp_after_sauce": next(
                                    (s for s, x in arm["physical"].items() if s > sauce_step and x["held_by_target"][SOUP]), None)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=GPU, required=True)
    for name in ("study", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    verify_sources()
    import run as api
    from actual_physics_worker import worker_class

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    worker_type = worker_class(api)
    results = []
    for layout in LAYOUTS:
        arm = source_arm(args.source_root.resolve(), layout)
        case = replay(api, worker_type, bddl, layout, out / str(layout), arm)
        path = out / str(layout) / "report.json"
        write(path, case)
        compact = {key: case[key] for key in ("layout_seed", "source", "source_audit_sha256", "source_result_sha256",
                   "source_policy_seed", "source_actions_sha256", "first_observed_sauce_commit", "initial_eef_m",
                   "initial", "first_trigger", "evaluation_only", "exact_replay_rows", "replayed_actions_sha256",
                   "replayed_grasp_sha256")}
        compact.update(report=path.relative_to(out).as_posix(), report_sha256=sha(path),
                       base_trigger_steps=[s["step"] for s in case["samples"] if s["base_trigger"]],
                       all_trigger_steps=[s["step"] for s in case["samples"] if s["trigger"]])
        results.append(compact)
        print("BASKET_STALL", layout, compact["all_trigger_steps"], case["evaluation_only"], flush=True)
    admitted = [r["layout_seed"] for r in results if r["first_trigger"] is not None
                and r["evaluation_only"]["truth_admits_first_trigger_window"]]
    gate = CONTROL not in admitted and all(r["first_trigger"] is None or
           r["evaluation_only"]["truth_admits_first_trigger_window"] for r in results) and len(admitted) >= 2
    write(out / "summary.json", {"schema": "v2-basket-stall-probe/1", "source_sha256": sha(Path(__file__)),
          "card": args.card, "study_bddl_sha256": sha(bddl), "layouts": list(LAYOUTS), "record_cutoff": CUTOFF,
          "rule": {"window_steps": WINDOW, "consecutive_boundaries": COUNT, "cadence": CHUNK, "gap_m": GAP,
                   "eef_lower_m": LOWER, "eef_upper_m": UPPER, "max_spread_m": MAX_SPREAD},
          "cases": results, "admitted_layouts": admitted, "develop_retreat_pair": gate,
          "scope": "Seen exact-action replay. No new policy inference or control."})


if __name__ == "__main__":
    main()
