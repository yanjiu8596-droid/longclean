"""Exercise stage-4 progress writing on sealed smoke forks with fixed OSC actions."""
import argparse
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, read, require, sha, validate_gpu, validate_study, write
from common import CHEESE, CHUNK, SAUCE, SOUP
from legacy_parity import cases, trace
from progress import ClosedLoopProgress, STRIDE
from run_closed_loop import audited


ARMS = ("baseline", "candidate")
FOLLOW_STEPS = 40
HOLD = [0, 0, 0, 0, 0, 0, -1]


def begin(api, worker, prefix, fixture, case, progress, target):
    api.replay_prefix(worker, prefix, progress)
    require(api.fingerprint(worker.current) == prefix["end"] and progress.complete(CHEESE),
            "The shared smoke prefix changed")
    if case["moved"]:
        state = worker.call({"op": "move", "offset": case["offset"]})
        require("rejected" not in state and api.fingerprint(state) == fixture["perturbed_checkpoint"],
                "Smoke motion changed")
    else:
        require(api.fingerprint(worker.current) == prefix["end"], "Sham changed at branch")
    fork = api.fingerprint(worker.current)
    require(fork["step"] % CHUNK == 0 and worker.goal(CHEESE) and not worker.goal(SAUCE)
            and not worker.goal(SOUP), "Expected cheese-only shared fork")
    copyfile(worker.npz, target / "fork.npz")
    return fork


def generate(api, bddl, case, prefix, fixture, target, locate):
    """Capture one physically executed source trace, separately from both observers."""
    with api.Worker(bddl, target, case["layout_seed"]) as worker:
        fork = begin(api, worker, prefix, fixture, case, api.Progress(locate), target)
        api.scripted_soup(worker)
        for _ in range(FOLLOW_STEPS):
            if worker.terminal():
                break
            worker.step([HOLD], SOUP, "diagnostic release hold")
        end = api.fingerprint(worker.current)
    rows = [r for r in trace(target / "actions.jsonl") if r["start_step"] >= fork["step"]]
    expected = {r["step"]: r for r in rows}
    actions = [a for r in rows for a in r["actions"]]
    require(end["step"] == fork["step"] + len(actions) and actions,
            "Incomplete fixed OSC source trace")
    source = {"fork": fork, "terminal": end, "actions_sha256": sha(target / "actions.jsonl"),
              "fork_npz_sha256": sha(target / "fork.npz"), "action_steps": len(actions)}
    write(target / "source.json", source)
    return fork, actions, expected, source


def replay_arm(api, bddl, case, prefix, fixture, target, locate, arm, fork, commands, expected):
    progress = ClosedLoopProgress(locate, write_soup=(arm == "candidate"))
    events, boundaries, evidence = [], [], []
    tag = "diagnostic soup transport; fixed actions"
    with api.Worker(bddl, target, case["layout_seed"]) as worker:
        local_fork = begin(api, worker, prefix, fixture, case, progress, target)
        require(local_fork == fork, "Observer arms must start at the fixed-motion source fork")
        progress.start_soup(api.Frame.read(worker.npz), fork["step"])
        old_location = fixture["soup_before"]
        schedule = api.Schedule(progress, True, old_location["xyz_m"] if old_location else None)
        located = locate(worker.npz)
        schedule.change(located["xyz_m"] if located else None, api.sauce_location(api.Frame.read(worker.npz)))
        boundaries.append({"step": fork["step"], "target": schedule.next(fork["step"]),
                           "order": list(schedule.order), "official_success": False})
        for start in range(0, len(commands), CHUNK):
            combined = []
            for i in range(start, min(start + CHUNK, len(commands)), STRIDE):
                called = commands[i:min(i + STRIDE, len(commands))]
                actual = worker.step(called, SOUP, tag)
                require(actual == called, "Replay changed the fixed commands")
                combined += actual
                current = worker.current
                key = current["step"]
                require(api.fingerprint(current) == api.fingerprint(expected[key])
                        and current["audit"] == expected[key]["audit"], f"Mechanism replay diverged at {key}")
                frame = api.Frame.read(worker.npz)
                for event in progress.observe_soup(frame, actual, key):
                    sample = audited(event, current)
                    if sample["event"] == "in_basket":
                        name = f"soup-commit-{key}.npz"
                        copyfile(worker.npz, target / name)
                        sample.update(evidence_file=name, evidence_sha256=sha(target / name))
                    events.append(sample)
                if worker.terminal():
                    break
            legacy = progress.observe(frame, SOUP, combined)
            if legacy:
                events.append(audited({**legacy, "writer": "legacy", "applied": True}, worker.current))
            end = worker.current["step"]
            if end % CHUNK == 0:
                nxt = schedule.next(end)
                boundaries.append({"step": end, "target": nxt, "order": list(schedule.order),
                                   "official_success": all(worker.current["audit"]["goals"].values())})
            if worker.terminal():
                break
        terminal = api.fingerprint(worker.current)
    require(terminal == api.fingerprint(expected[max(expected)]), "Incomplete observer replay")
    for event in events:
        if event["writer"] != "stage3d_plane" or not event["applied"]:
            continue
        before = [b for b in boundaries if b["step"] < event["step"]]
        after = [b for b in boundaries if b["step"] >= event["step"]]
        if before and after and SOUP in before[-1]["order"] and SOUP not in after[0]["order"]:
            evidence.append({"event": event, "boundary_before": before[-1], "boundary_after": after[0]})
    result = {"arm": arm, "fork": local_fork, "terminal": terminal, "events": events,
              "schedule_boundaries": boundaries, "eligibility_changes": evidence,
              "actions_sha256": sha(target / "actions.jsonl"), "fork_npz_sha256": sha(target / "fork.npz")}
    write(target / "arm.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=GPU, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    assigned = [c for c in cases(args.output) if c["card"] == args.card]
    require(len(assigned) == 1, "Expected one sealed smoke fork per card")
    case, = assigned
    fixture_dir = args.output / "fixtures" / str(case["layout_seed"])
    require(sha(fixture_dir / "prefix.json") == case["prefix_sha256"]
            and sha(fixture_dir / "fixture.json") == case["fixture_sha256"], "Changed smoke fixture")
    prefix, fixture = read(fixture_dir / "prefix.json"), read(fixture_dir / "fixture.json")
    destination = args.output / "progress-exercise" / case["tag"]
    marker = args.output / f"progress-exercise-{args.card}.json"
    require(not marker.exists(), "Refusing to overwrite a mechanism smoke report")
    destination.mkdir(parents=True, exist_ok=False)
    import run as api

    locate = api.load_locator()
    fork, actions, expected, source = generate(api, bddl, case, prefix, fixture, destination / "source", locate)
    arms = {arm: replay_arm(api, bddl, case, prefix, fixture, destination / arm, locate,
                            arm, fork, actions, expected) for arm in ARMS}
    valid = [x for x in arms["candidate"]["eligibility_changes"]
             if x["event"]["confirmed_in_audit"] and not x["boundary_after"]["official_success"]]
    wrong = sum(e["applied"] and not e["confirmed_in_audit"]
                for r in arms.values() for e in r["events"])
    report = {"schema": "v2-stage4-progress-exercise/1", "kind": "fixed_motion_mechanism_smoke",
              "card": args.card, "layout_seed": case["layout_seed"], "moved": case["moved"],
              "freeze_sha256": sha(args.output / "frozen.json"), "source_sha256": sha(Path(__file__)),
              "record_dir": destination.relative_to(args.output).as_posix(), "source": source,
              "arms": arms, "valid_write": bool(valid), "wrong_writes": wrong, "valid_writes": valid}
    write(marker, report)
    print("PROGRESS_EXERCISE", case["layout_seed"], case["moved"], bool(valid), wrong, flush=True)


if __name__ == "__main__":
    main()
