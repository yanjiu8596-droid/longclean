"""Score passive observers against simulator audit on exact, common action traces."""
import argparse
import json
from pathlib import Path
import sys


DEV = (
    ("924122-s34-moved", "replan", 350),
    ("924122-s35-moved", "replan", 260),
    ("924130-s34-sham", "replan", 500),
    ("924109-s34-moved", "replan", 386),
    ("924109-s35-moved", "replan", 382),
    ("924101-s34-moved", "replan", 280),
    ("924122-s34-moved", "fixed", 300),
    ("924105-s35-moved", "fixed", 290),
    ("924113-s34-moved", "fixed", 290),
)


def score(timeline, events, targets):
    placements = []
    by_step = {row["step"]: row for row in timeline}
    for target in targets:
        # Look for starts of predicate intervals that persist across the next 20
        # steps, or reach the official terminal frame of a successful episode.
        transitions = [(ev["step"], ev["in_basket"], row["assigned"])
                       for row in timeline for ev in row["transitions"] if ev["object"] == target]
        for i, (step, value, assigned) in enumerate(transitions):
            end = transitions[i + 1][0] if i + 1 < len(transitions) else timeline[-1]["step"] + 1
            if not value or (end - step <= 20 and
                             not (i + 1 == len(transitions) and timeline[-1]["all_goals"])):
                continue
            eligible = [row["step"] for row in timeline if step <= row["step"] <= step + 30 and row["step"] < end]
            if not eligible:
                continue
            hits = [ev["step"] for ev in events if ev["object"] == target and ev["step"] in eligible]
            placements.append({"object": target, "first_truth_step": step, "assigned": assigned,
                               "outside_assigned": assigned is not None and assigned != target,
                               "eligible_steps": eligible, "hit_step": min(hits) if hits else None,
                               "terminal_censored": len(eligible) == 1 and eligible[0] == timeline[-1]["step"]})
            break
    errors = []
    for ev in events:
        if not by_step[ev["step"]]["goals"][ev["object"]]:
            future = [t["step"] for row in timeline for t in row["transitions"]
                      if t["object"] == ev["object"] and t["in_basket"] and t["step"] > ev["step"]]
            errors.append({**ev, "next_truth_step": future[0] if future else None})
    return {"assertions": events, "false_at_assertion": errors, "placements": placements}


def negative_releases(timeline, targets):
    result = []
    for previous, row in zip(timeline, timeline[1:]):
        step = row["step"]
        if not row["release_probe"] or timeline[-1]["step"] < step + 30:
            continue
        still_due = [target for target in targets if not previous["goals"][target]]
        # A release probe is a measured opening near the basket. Include empty
        # releases and unsuccessful drops, without consulting either observer.
        if still_due and not any(t["object"] in still_due and t["in_basket"]
                                 and previous["step"] < t["step"] <= step + 30
                                 for block in timeline for t in block["transitions"]):
            result.append({"step": step, "still_due": still_due})
    return result


def replay(study, out, fixture, trace, limit, probe_hold):
    from common import read, sha
    from observer import PlacementObserver, TARGETS, near_basket
    from run import Worker, fingerprint, replay_prefix
    from vision import Frame, Progress, load_locator

    h = read(trace / "result.json")
    prefix = read(fixture / "prefix.json")
    targets = TARGETS
    bddl = study / read(study / "plan.json")["bddl"]
    assert h["prefix_sha256"] == sha(fixture / "prefix.json")
    older, candidate = Progress(load_locator()), PlacementObserver()
    previous, old_events, new_events, decision_samples, timeline = None, [], [], [], []
    with Worker(bddl, out, h["layout_seed"]) as worker:
        replay_prefix(worker, prefix, older)
        if h["moved"]:
            offset = read(fixture / "fixture.json")["offset"]
            worker.call({"op": "move", "offset": offset})
        assert fingerprint(worker.current) == h["trial_checkpoint"]
        start = worker.current["step"]
        candidate.clear_view = Frame.read(worker.npz)

        def observe(row, actions):
            nonlocal previous
            frame = Frame.read(worker.npz)
            step, target = row["step"], row["target"]
            # Neither observer receives row['audit'] or predicate transitions.
            prior = older.observe(frame, target, actions)
            seen = candidate.observe(frame, actions, step)
            if prior:
                old_events.append({"step": step, "matches_audit": row["audit"]["goals"][prior["object"]], **prior})
            new_events.extend({"step": step, **ev} for ev in seen if ev["event"] == "in_basket")
            decision_samples.extend({"step": step, **ev} for ev in seen)
            goals = row["audit"]["goals"]
            release = (previous is not None and previous.gap < .069 and frame.gap >= .067
                       and any(a[-1] < -.5 for a in actions) and near_basket(frame.geometry["eef"]))
            timeline.append({"step": step, "assigned": target, "goals": goals, "all_goals": all(goals.values()),
                             "transitions": row["transitions"], "release_probe": release})
            previous = frame

        with (trace / "actions.jsonl").open() as f:
            for row in map(json.loads, f):
                if row["start_step"] < start:
                    continue
                if limit is not None and row["step"] > limit:
                    break
                actions = worker.step(row["actions"], row["target"], row["instruction"])
                assert (fingerprint(worker.current) == fingerprint(row) and
                        worker.current["audit"] == row["audit"] and
                        worker.current["transitions"] == row["transitions"]), (trace, row["step"])
                observe(row, actions)
        assert old_events == [e for e in h["state_events"] if e["step"] <= worker.current["step"]], trace
        exact = len(timeline)
        if probe_hold:
            assert worker.current["step"] == 350 and h["terminal_reason"] == "observed_complete"
            assert not worker.terminal()
            actions = worker.step([[0, 0, 0, 0, 0, 0, -1]] * 10, None, "diagnostic hold")
            observe({**worker.current, "target": None}, actions)

    return {"layout": h["layout_seed"], "method": h["method"], "moved": h["moved"],
            "steps_replayed": exact, "probe_hold": probe_hold, "end": timeline[-1],
            "baseline": score(timeline, old_events, targets), "candidate": score(timeline, new_events, targets),
            "negative_releases": negative_releases(timeline, targets), "candidate_decisions": decision_samples}


def aggregate(rows):
    sums = {}
    for method in ("baseline", "candidate"):
        placements = [p for row in rows for p in row[method]["placements"]]
        evaluated = [p for p in placements if not p["terminal_censored"]]
        off = [p for p in evaluated if p["outside_assigned"]]
        errors = [e for row in rows for e in row[method]["false_at_assertion"]]
        sums[method] = {
            "eligible_positive": len(evaluated), "hits_within_30": sum(p["hit_step"] is not None for p in evaluated),
            "terminal_censored": sum(p["terminal_censored"] for p in placements),
            "hits_at_terminal_censored": sum(p["hit_step"] is not None for p in placements if p["terminal_censored"]),
            "off_target_positive": len(off), "off_target_hits": sum(p["hit_step"] is not None for p in off),
            "false_at_assertion": len(errors), "no_truth_in_tail": sum(e["next_truth_step"] is None for e in errors),
            "delays": [p["hit_step"] - p["first_truth_step"] for p in evaluated if p["hit_step"] is not None],
            "positive_by_object": {obj: sum(p["object"] == obj for p in evaluated)
                                   for obj in ("alphabet_soup_1", "tomato_sauce_1")},
            "hit_by_object": {obj: sum(p["object"] == obj and p["hit_step"] is not None for p in evaluated)
                              for obj in ("alphabet_soup_1", "tomato_sauce_1")},
        }
    return sums


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("develop", "validate"))
    p.add_argument("study", type=Path)
    p.add_argument("fresh", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    study, fresh, out = (path.resolve() for path in (a.study, a.fresh, a.out))
    sys.path.insert(0, str(study.parent))
    from common import read, sha, write

    frozen = read(study / "frozen.json")
    assert all(sha(study.parent / name) == digest for name, digest in frozen["code"].items())
    assert sha(study / read(study / "plan.json")["bddl"]) == read(study / "plan.json")["bddl_sha256"]
    this = Path(__file__).resolve().parent
    hashes = {filename: sha(this / filename) for filename in ("collect.py", "compare.py", "observer.py")}
    out.mkdir()
    rows, intake = [], []
    if a.phase == "develop":
        cases = ((f"{tag}-{arm}", study / "fixtures" / tag.split("-")[0], study / "pilot" / tag / arm,
                  limit, tag == "924122-s34-moved" and arm == "replan") for tag, arm, limit in DEV)
    else:
        freeze = read(this / "candidate-freeze.json")
        assert hashes == freeze["source"]
        by_initial = set()
        old_seeds = {case["layout_seed"] for case in frozen["cases"]}
        old_seeds.update(int(path.name) for path in (study / "fixtures").iterdir() if path.name.isdecimal())
        old_initial = {read(study / "fixtures" / str(seed) / "prefix.json")["initial"]["state_sha256"]
                       for seed in old_seeds if (study / "fixtures" / str(seed) / "prefix.json").is_file()}
        cases = []
        for seed in range(925100, 925116):
            fixture = fresh / str(seed)
            assert seed not in old_seeds
            d = read(fixture / "fixture.json")
            initial = d["initial"]["state_sha256"]
            is_new = initial not in by_initial and initial not in old_initial
            by_initial.add(initial)
            intake.append({**d, "distinct_hash": is_new})
            if d["collected"] and is_new:
                cases.append((str(seed), fixture, fixture / "trace", None, False))
    for name, fixture, trace, limit, hold in cases:
        row = replay(study, out / name, fixture, trace, limit, hold)
        write(out / name / "comparison.json", row)
        rows.append(row)
        print("verified", name, row["steps_replayed"], flush=True)
    metrics = aggregate(rows)
    layouts = len({row["layout"] for row in rows})
    neg_layouts = len({row["layout"] for row in rows if row["negative_releases"]})
    c, b = metrics["candidate"], metrics["baseline"]
    adequate = (layouts >= 8 and neg_layouts >= 4 and c["off_target_positive"] >= 4
                and min(c["positive_by_object"].values()) >= 3)
    passed = (adequate and c["false_at_assertion"] == 0 and c["off_target_hits"] > b["off_target_hits"]
              and c["hits_within_30"] > b["hits_within_30"]
              and all(c["hit_by_object"][obj] >= max(1, b_hits) for obj, b_hits in b["hit_by_object"].items()))
    write(out / "summary.json", {"phase": a.phase, "source": hashes, "layouts_compared": layouts,
                                  "traces_compared": len(rows),
                                  "negative_release_layouts": neg_layouts,
                                  "coverage_gate": adequate if a.phase == "validate" else None,
                                  "passes_candidate_gate": passed if a.phase == "validate" else None,
                                  "metrics": metrics, "intake": intake, "traces": rows})


if __name__ == "__main__":
    main()
