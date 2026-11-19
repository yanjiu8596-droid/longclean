"""Score a frozen landing observer on new natural positives and physical negatives.

Usage: python -B stage3d_gate.py FREEZE.json OUT.json [--preflight]

FREEZE.json (archive paths are relative to the freeze; code paths to this script)::

    {
      "schema": "identity-stage3d-freeze-v1",
      "source_sha256": {"tracker.py": "...", "observer_module.py": "..."},
      "candidate": {"factory": "observer_module:make_observer", "cadence": "coarse"},
      "collections": [
        {"kind": "natural", "manifest": "natural/manifest-2.json",
         "sha256": "...", "collector_sha256": "...", "layouts": [930100]},
        {"kind": "physical", "manifest": "outside/manifest-2.json",
         "sha256": "...", "collector_sha256": "...", "layouts": [929320],
         "classes": ["outside"]}
      ],
      "excluded_layouts": [], "excluded_state_sha256": [], "excluded_raw_sha256": []
    }

Freeze the observer and collection plans before opening the holdout and seal each
complete manifest before scoring. A factory returns a fresh object with
`observe(frame, realized_actions, step)`. Only RGB-D, calibration, eef/gripper,
the actual action interval and step enter it. A "dense" factory gets every step
and may implement an adaptive cadence internally; all inputs must be dense.
Do not infer dense zero-error performance from the stride-5 archive.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import CAMERAS, CHEESE, COMMIT, SAUCE, SOUP, array_hash, read, sha, write

CLASSES = ("outside", "rim", "other_can", "identity_unknown")
KNOWN_DEVELOPMENT = {*range(929090, 929124), 929190, 929191, *range(929200, 929208),
                     *range(929300, 929304), *range(930000, 930008)}
CAMERA_FIELDS = ("", "_depth", "_intrinsic", "_extrinsic")
REQUIRED_ARRAYS = {name + suffix for name in CAMERAS for suffix in CAMERA_FIELDS} | {
    "eef_pos", "eef_quat", "gripper_qpos"
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def exact_file(path, expected):
    require(digest(expected) and sha(path) == expected, f"SHA-256 mismatch: {path}")


def pin_inputs(freeze_path):
    """Validate the plan and held-out metadata before opening videos or scoring."""
    freeze = read(freeze_path)
    require(freeze["schema"] == "identity-stage3d-freeze-v1", "Unsupported freeze schema")
    candidate = freeze["candidate"]
    require(candidate["cadence"] in ("dense", "coarse"), "Invalid candidate cadence")
    require(re.fullmatch(r"[A-Za-z_]\w*:[A-Za-z_]\w*", candidate["factory"]),
            "Factory must have module:function syntax")
    module_name = candidate["factory"].split(":")[0]
    source = freeze["source_sha256"]
    require({"tracker.py", module_name + ".py"} <= set(source), "Unpinned baseline or candidate")
    for name, wanted in source.items():
        exact_file(HERE / name, wanted)
    seed_seen = KNOWN_DEVELOPMENT | set(freeze.get("excluded_layouts", ()))
    states = set(freeze.get("excluded_state_sha256", ()))
    raw = set(freeze.get("excluded_raw_sha256", ()))
    require(all(digest(h) for h in states | raw), "Invalid excluded start digest")
    collections, episodes, manifests_seen = [], [], set()
    for declared in freeze["collections"]:
        kind = declared["kind"]
        require(kind in ("natural", "physical"), f"Unknown collection kind: {kind}")
        path = (freeze_path.parent / declared["manifest"]).resolve()
        require(path not in manifests_seen, f"Duplicate manifest: {path}")
        manifests_seen.add(path)
        exact_file(path, declared["sha256"])
        m = read(path)
        require(m["phase"] == "blind" and m["complete"] is True, f"Not a complete holdout: {path}")
        require(m["commit"] == COMMIT, f"Unexpected benchmark revision: {path}")
        require(digest(m["bddl_sha256"]) and digest(m["collector_sha256"]) and
                m["collector_sha256"] == declared["collector_sha256"], f"Unexpected collector or task: {path}")
        stride = m["frame_stride"]
        require(stride in (1, 5), f"Unsupported frame stride {stride}: {path}")
        require(candidate["cadence"] != "dense" or stride == 1,
                f"Dense qualification needs full stride-1 input; recollect or exactly replay {path}")
        planned = m["planned"]
        ids = [job["layout"] for job in planned]
        require(ids and ids == declared["layouts"], f"Unexpected planned layouts or order: {path}")
        require(len(ids) == len(m["records"]), f"Incomplete records: {path}")
        cohort_bddl = declared.get("bddl_sha256", m["bddl_sha256"])
        require(cohort_bddl == m["bddl_sha256"], f"Unexpected task digest: {path}")
        if kind == "natural":
            require(m["cohort"] == "natural", f"Scripted controls cannot qualify as natural: {path}")
        else:
            classes = declared["classes"]
            require(m["followup_steps"] == 30 and bool(classes) and
                    set(classes) <= set(CLASSES) and set(classes) == {job["case"] for job in planned},
                    f"Unexpected physical class/followup: {path}")
        for job, record in zip(planned, m["records"]):
            layout = job["layout"]
            require(type(layout) is int and layout not in seed_seen, f"Repeated or development seed: {layout}")
            seed_seen.add(layout)
            rel = f"{layout}/fixture.json"
            actual = record if kind == "natural" else record["path"]
            require(actual == rel, f"Unexpected layout record: {actual}")
            directory = path.parent / str(layout)
            f = read(directory / "fixture.json")
            require(f["layout"] == layout and f["initial"]["step"] == 0 and
                    f["initial"]["bddl_sha256"] == cohort_bddl == f["end"]["bddl_sha256"],
                    f"Misaligned fixture identity or task: {directory}")
            state, origin = f["initial"]["state_sha256"], f["initial"]["raw_sha256"]
            require(digest(state) and digest(origin) and state not in states and origin not in raw,
                    f"Repeated or invalid initial state: {directory}")
            states.add(state)
            raw.add(origin)
            if kind == "physical":
                require(job["case"] == record["case"] == f["case"] and layout == record["layout"] and
                        state == record["initial_sha256"], f"Mismatched physical fixture: {directory}")
                quality = f["release_quality"]
                require(all(quality.get(k) == v for k, v in record.items() if k in quality) and
                        {"checks", "eligible", "measured_open_step"} <= set(record),
                        f"Release checks do not match sealed manifest: {directory}")
            frames, end = f["frame_steps"], f["end"]["step"]
            require(type(end) is int and end >= 0 and frames == sorted(set([*range(0, end + 1, stride), end])),
                    f"Missing or unordered acquisition frames: {directory}")
            episodes.append((kind, stride, directory, f, record))
        collections.append({"kind": kind, "manifest": declared["manifest"], "sha256": declared["sha256"],
                            "bddl_sha256": cohort_bddl, "frame_stride": stride, "layouts": ids})
    require(episodes and any(kind == "natural" for kind, *_ in episodes), "No natural holdout")
    require(len({m["bddl_sha256"] for m in collections}) == 1,
            "Natural positives and controls must use the same official task definition")
    return freeze, collections, episodes


def replay_rows(directory, fixture):
    """Yield archived endpoints, enforcing contiguous realized action intervals."""
    fdir = directory / "frames"
    yield 0, fdir / "0.npz", [], fixture["initial"], fixture["initial_audit"], []
    last = 0
    seen = [0]
    expected = fixture["frame_steps"]
    final = {**fixture["initial"], "audit": fixture["initial_audit"]}
    with (directory / "trace" / "actions.jsonl").open(encoding="utf-8") as src:
        for row in map(json.loads, src):
            step, actions = row["step"], row["actions"]
            require(row["start_step"] == last and step == last + len(actions) and bool(actions) and
                    len(seen) < len(expected) and step == expected[len(seen)],
                    f"Invalid action boundaries: {directory}:{step}")
            require(all(len(a) == 7 and all(type(x) in (int, float) and math.isfinite(x) for x in a)
                        for a in actions), f"Malformed realized actions: {directory}:{step}")
            require(row["bddl_sha256"] == fixture["initial"]["bddl_sha256"] and
                    row["raw_sha256"] == fixture["initial"]["raw_sha256"] and digest(row["state_sha256"]),
                    f"Inconsistent trace state fingerprint: {directory}:{step}")
            transitions = row.get("transitions", [])
            require(all(last < t["step"] <= step and type(t["in_basket"]) is bool for t in transitions),
                    f"Out-of-interval truth transitions: {directory}:{step}")
            seen.append(step)
            last = step
            final = row
            yield step, fdir / f"{step}.npz", actions, row, row["audit"], transitions
    require(seen == expected and all(final[k] == v for k, v in fixture["end"].items()) and
            final["audit"] == fixture["end_audit"], f"Incomplete episode end: {directory}")


def load_frame(path, expected, np, Frame):
    with np.load(path, allow_pickle=False) as obs:
        require(set(obs.files) == set(expected) == REQUIRED_ARRAYS, f"Missing or extra sensor arrays: {path}")
        require(all(array_hash(obs[k]) == value for k, value in expected.items()),
                f"Observation SHA-256 mismatch: {path}")
    # The file path identifies the fixture/class and is not a sensor input.
    return replace(Frame.read(path), source=None)


def truth_positive(trail):
    # Same stable-window rule as stage 3b: correct and unheld for 20 steps
    # from the first sampled qualifying time; require 30 steps of observation.
    censored = False
    for i, row in enumerate(trail):
        start = row["step"]
        if not row["truth"]:
            continue
        end = start + 20
        if any((r["step"] <= end and not r["truth"]) or any(
                start < t["step"] <= end and t["object"] == SOUP and not t["in_basket"]
                for t in r["transitions"]) for r in trail[i:]):
            continue
        if trail[-1]["step"] >= start + 30:
            return start, False
        censored = True
    return None, censored


def physical_eligibility(directory, fixture, record, trail, coarse):
    """Re-derive the standard collector checks from saved sensor/truth streams."""
    label = fixture["case"]
    quality = fixture["release_quality"]
    audit_path = directory / "physics-audit.json"
    if "physics_audit_sha256" in record:
        exact_file(audit_path, record["physics_audit_sha256"])
    physics = read(audit_path)
    require(physics["release_quality"] == quality, f"Physical truth metadata disagrees: {directory}")
    steps, carried = physics["steps"], physics["controlled_object"]
    allowed = {"outside": (SOUP,), "rim": (SOUP,), "other_can": (SAUCE,), "identity_unknown": (SAUCE, CHEESE)}
    require(carried in allowed[label] and [r["step"] for r in steps] == list(range(trail[-1]["step"] + 1)),
            f"Wrong carried object or incomplete physics audit: {directory}")
    for row in trail:
        step = row["step"]
        require(row["goals"] == steps[step]["goals"] and row["held"] == steps[step]["held"][SOUP],
                f"Conflicting physics audit: {directory}:{step}")
    commanded = quality["open_command_after_step"]
    require(commanded == physics["stages"].get("open") and
            (commanded is None or type(commanded) is int and 0 <= commanded <= steps[-1]["step"]),
            f"Opening command mismatch: {directory}")
    measured = next((b["step"] for a, b in zip(coarse, coarse[1:]) if commanded is not None and
                     commanded < b["step"] <= commanded + 15 and .031 < a["gap"] < .072 and
                     b["gap"] >= .067 and b["opens"] and b["near"]), None)
    require(measured == quality["measured_open_step"], f"Opening observation mismatch: {directory}")
    after = steps[measured:measured + 31] if measured is not None else []
    tail = after[-1] if after else None
    full = len(after) == 31
    stable_rim = bool(full and all(a["rim_contact"] and not a["held"][SOUP] for a in after[-5:]))
    actual = {
        "held_at_release_command": bool(commanded is not None and steps[commanded]["held"][carried]),
        "measured_near_open": measured is not None,
        "complete_30step_window": full,
        "soup_false_through_window": full and all(not a["goals"][SOUP] for a in after),
        "released_by_end": bool(tail and not tail["held"][carried]),
        "carried_at_least_5cm": bool(tail and sum((x - y) ** 2 for x, y in
            zip(tail["positions"][carried], steps[0]["positions"][carried])) >= .05 ** 2),
    }
    if label in ("outside", "rim", "identity_unknown"):
        actual["other_can_not_in"] = full and all(not a["goals"][SAUCE] for a in after)
    if label in ("other_can", "identity_unknown"):
        actual["soup_never_held"] = all(not a["held"][SOUP] for a in steps)
    if label == "outside" or (label == "identity_unknown" and carried == SAUCE):
        dx, dy = [tail["positions"][carried][i] - tail["positions"]["basket_1"][i]
                  for i in (0, 1)] if tail else (0, 0)
        actual["outside_xy" if label == "outside" else "sauce_outside_xy"] = abs(dx) > .070 or abs(dy) > .085
    if label == "outside":
        actual["rim_did_not_catch"] = not stable_rim
    checks = quality["checks"]
    if label == "rim":
        actual["unheld_rim_contact_last_5"] = stable_rim
        if "robot_clear_last_5" in checks or "tail_displacement_at_most_5mm" in checks:
            segment = after[26:31]
            xyz = [row["positions"][SOUP] for row in segment]
            spread = max((math.dist(p, q) for p in xyz for q in xyz), default=None)
            actual["robot_clear_last_5"] = full and all(not row["robot_contact"] for row in segment)
            actual["tail_displacement_at_most_5mm"] = full and spread <= .005
            reported = quality["tail_displacement_m"]
            require((spread is None and reported is None) or
                    (spread is not None and reported is not None and abs(spread - reported) <= 1e-9),
                    f"Rim displacement disagrees with physics audit: {directory}")
    if label == "other_can":
        actual["sauce_in_final_20"] = full and all(a["goals"][SAUCE] and not a["held"][SAUCE] for a in after[-20:])
    if label == "identity_unknown" and carried == CHEESE:
        actual["cheese_in_final_20"] = full and all(a["goals"][CHEESE] and not a["held"][CHEESE] for a in after[-20:])
    require(all(type(v) is bool for v in checks.values()) and all(checks.get(k) == v for k, v in actual.items()) and
            quality["eligible"] is all(checks.values()), f"Physical checks disagree with evidence: {directory}")
    return quality["eligible"], measured


def event_commits(observer, frame, actions, step):
    result = []
    for e in observer.observe(frame, actions, step):
        if e["event"] == "in_basket" and e.get("object", SOUP) == SOUP:
            require(e.get("step", step) == step, f"Observer returned a backdated commitment at {step}")
            result.append({"step": step, "source": str(e.get("source", "unspecified"))})
    return result


def score_episode(episode, factory, mode, baseline_class, Frame, np):
    kind, stride, directory, fixture, record = episode
    observers = {"baseline": baseline_class(), "candidate": factory()}
    events = {name: [] for name in observers}
    seen = Counter()
    trail, coarse, batched = [], [], []
    for step, path, actions, raw, audit, transitions in replay_rows(directory, fixture):
        frame = load_frame(path, raw["observation_sha256"], np, Frame)
        batched.extend(actions)
        grid = step % 5 == 0 or step == fixture["end"]["step"]
        if grid:
            x, y, z = frame.geometry["eef"]
            coarse.append({"step": step, "gap": frame.gap, "opens": any(a[-1] == -1 for a in batched),
                           "near": -.105 < x < .11 and .15 < y < .37 and .50 < z < .83})
            events["baseline"] += event_commits(observers["baseline"], deepcopy(frame), deepcopy(batched), step)
            seen["baseline"] += 1
        if grid or mode == "dense":
            events["candidate"] += event_commits(observers["candidate"], frame, actions if mode == "dense" else batched,
                                                 step)
            seen["candidate"] += 1
        if grid:
            batched = []
        # Never pass truth, transitions, layout, target, task or file names to an observer.
        goals, held = audit["goals"], audit["held_soup"]
        require(type(held) is bool and all(type(goals[k]) is bool for k in (SOUP, SAUCE, CHEESE)),
                f"Invalid independent truth audit: {directory}:{step}")
        trail.append({"step": step, "goals": goals, "held": held, "truth": goals[SOUP] and not held,
                      "transitions": transitions})
    positive, censored = truth_positive(trail) if kind == "natural" else (None, False)
    eligible, release = physical_eligibility(directory, fixture, record, trail, coarse) if kind == "physical" else (False, None)
    truth = {row["step"]: row["truth"] for row in trail}
    scores = {}
    for name, commits in events.items():
        errors = [e for e in commits if not truth[e["step"]]]
        hits = [e["step"] for e in commits if e not in errors and positive is not None and
                positive <= e["step"] <= positive + 30]
        scores[name] = {"hit": min(hits, default=None), "false": errors, "commits": commits,
                        "observation_calls": seen[name]}
    return {"layout": fixture["layout"], "cohort": kind, "frame_stride": stride,
            "positive_step": positive, "terminal_censored": censored, "frames_verified": len(trail),
            "negative_class": fixture.get("case"), "negative_eligible": eligible,
            "negative_release_step": release, "scores": scores}


def summarize(results):
    positives = [r for r in results if r["positive_step"] is not None]
    neg = {label: sum(r["negative_eligible"] and r["negative_class"] == label for r in results) for label in CLASSES}
    attempted = {label: sum(r["negative_class"] == label for r in results) for label in CLASSES}
    methods = {}
    for name in ("baseline", "candidate"):
        delays = [r["scores"][name]["hit"] - r["positive_step"] for r in positives
                  if r["scores"][name]["hit"] is not None]
        methods[name] = {"30_step_hits": len(delays), "delay_steps": delays,
                         "false_full_episodes": sum(len(r["scores"][name]["false"]) for r in results),
                         "observation_calls": sum(r["scores"][name]["observation_calls"] for r in results)}
    coverage = len(positives) >= 8 and sum(neg.values()) >= 16 and all(v >= 4 for v in neg.values())
    candidate = methods["candidate"]
    return {"positive": len(positives), "positive_attempts": sum(r["cohort"] == "natural" for r in results),
            "terminal_censored": sum(r["terminal_censored"] for r in results),
            "negative_eligible": neg, "negative_attempted": attempted, "metrics": methods, "coverage": coverage,
            "gate": coverage and candidate["30_step_hits"] >= math.ceil(.75 * len(positives))
            and candidate["false_full_episodes"] == 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("freeze", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--preflight", action="store_true", help="Validate sealed metadata only; do not import observers")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frozen_sha = sha(args.freeze)
    freeze, collections, episodes = pin_inputs(args.freeze)
    planned_natural = sum(ep[0] == "natural" for ep in episodes)
    declared_eligible = {label: sum(ep[0] == "physical" and ep[3]["case"] == label and ep[4]["eligible"]
                                    for ep in episodes) for label in CLASSES}
    feasible = planned_natural >= 8 and sum(declared_eligible.values()) >= 16 and all(
        n >= 4 for n in declared_eligible.values())
    report = {"schema": "identity-stage3d-report-v1", "freeze_sha256": frozen_sha,
              "scorer_sha256": sha(Path(__file__)), "collections": collections, "candidate": freeze["candidate"],
              "preflight": {"natural_attempts": planned_natural, "declared_eligible": declared_eligible,
                            "ready_for_scoring": feasible}}
    if args.preflight:
        write(args.out, {**report, "preflight_only": True})
        return
    require(feasible, f"Coverage is incomplete; holdout left unopened: {report['preflight']}")
    module, func = freeze["candidate"]["factory"].split(":")
    imported = import_module(module)
    require(Path(imported.__file__).resolve() == HERE / (module + ".py"), "Factory module resolved outside frozen path")
    factory = getattr(imported, func)
    from tracker import IdentityObserver
    from vision import Frame
    import numpy as np

    results = [score_episode(episode, factory, freeze["candidate"]["cadence"], IdentityObserver, Frame, np)
               for episode in episodes]
    summary = summarize(results)
    require(sha(args.freeze) == frozen_sha, "Freeze was changed during scoring")
    for name, wanted in freeze["source_sha256"].items():
        exact_file(HERE / name, wanted)
    write(args.out, {**report, **summary, "results": results})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
