"""Same-observation comparison; simulator audit is consumed only after scoring."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "placement-next", ROOT / "soup-study", ROOT / "lid-geometry", HERE):
    sys.path.insert(0, str(path))
from common import SAUCE, SOUP, array_hash, read, sha, write
from observer import PlacementObserver
from screen import choose
from tracker import IdentityObserver
from vision import Frame
from wrist import WristSoup


def near_basket(frame):
    x, y, z = frame.geometry["eef"]
    return bool(-.105 < x < .11 and .15 < y < .37 and .50 < z < .83)


class GeometryOnly:
    """Release-timed geometry comparator, intentionally without identity."""
    def __init__(self):
        self.prev = None
        self.pending = None
        self.last = None
        self.done = False

    def observe(self, frame, actions, step):
        events = []
        if not self.done and self.prev is not None:
            opens = any(a[-1] < -.5 for a in actions)
            if (self.pending is None and opens and near_basket(frame) and
                    .031 < self.prev.gap < .072 and frame.gap >= .063):
                self.pending = (step, self.prev.geometry["eef"][:2])
            if self.pending is not None:
                started, anchor = self.pending
                if step > started + 40 or frame.gap < .058:
                    self.pending, self.last = None, None
                else:
                    _, choice = choose(frame, anchor)
                    if choice and self.last:
                        ts, old = self.last
                        if (5 <= step - ts <= 20 and
                                np.linalg.norm(np.array(old["center_m"]) - choice["center_m"]) <= .022
                                and abs(old["radius_m"] - choice["radius_m"]) <= .012):
                            self.done = True
                            events.append({"object": SOUP, "event": "in_basket", "center_m": choice["center_m"],
                                           "source": "geometry_only"})
                    self.last = (step, choice) if choice else None
        self.prev = frame
        return events


def observers():
    return {"rgbd": PlacementObserver(), "wrist": WristSoup(),
            "geometry": GeometryOnly(), "identity": IdentityObserver()}


def replay(rows):
    methods = observers()
    outputs = {name: [] for name in methods}
    snapshots = []
    for step, frame_path, actions, recorded in rows:
        audit = recorded.get("audit")
        hashes = recorded.get("observation_sha256")
        if hashes is not None:
            with np.load(frame_path, allow_pickle=False) as raw:
                assert all(array_hash(raw[n]) == digest for n, digest in hashes.items()), frame_path
        frame = Frame.read(frame_path)
        snapshots.append({"step": step, "opens": any(a[-1] < -.5 for a in actions),
                          "gap": frame.gap, "near": near_basket(frame), "audit": audit,
                          "transitions": recorded.get("transitions", [])})
        for name, obj in methods.items():
            result = obj.observe(frame, actions, step)
            outputs[name].extend({"step": step, **e} for e in result if e.get("object", SOUP) == SOUP)
    return outputs, snapshots


def stable_positive(rows):
    if not rows:
        return None, False
    censored = False
    for i, row in enumerate(rows):
        if not row["audit"]["goals"][SOUP] or row["audit"]["held_soup"]:
            continue
        end = row["step"] + 20
        following = [r for r in rows[i:] if r["step"] <= end]
        if any(not r["audit"]["goals"][SOUP] or r["audit"]["held_soup"]
               or any(t["object"] == SOUP and not t["in_basket"] and row["step"] < t["step"] <= end
                      for t in r["transitions"]) for r in following):
            continue
        if rows[-1]["step"] >= row["step"] + 30:
            return row["step"], False
        censored = True
    return None, censored


def negative_labels(rows):
    """Annotate up to one measured open-near-basket window per class and layout."""
    labels = {}
    for i in range(1, len(rows)):
        left, right = rows[i - 1:i + 1]
        step = right["step"]
        if not (.031 < left["gap"] < .072 and right["gap"] >= .067 and right["opens"] and right["near"]):
            continue
        if rows[-1]["step"] < step + 30:
            continue
        after = [r for r in rows[i:] if r["step"] <= step + 30]
        if left["audit"]["goals"][SOUP] or any(
            r["audit"]["goals"][SOUP] or any(t["object"] == SOUP and t["in_basket"] for t in r["transitions"])
            for r in after
        ):
            continue
        recent = rows[max(0, i - 3):i]
        if any(r["audit"]["goals"][SAUCE] for r in [left, *after]):
            kind = "other_can"
        elif any(r["audit"]["held_soup"] for r in recent):
            if after[-1]["audit"]["held_soup"]:
                continue
            pos = after[-1]["audit"]["positions"]
            x, y, z = np.subtract(pos[SOUP], pos["basket_1"])
            if abs(x) <= .085 and abs(y) <= .095 and z > .115:
                kind = "near_rim_high"
            elif abs(x) > .070 or abs(y) > .085:
                kind = "outside"
            else:
                continue
        else:
            # A measured release with no audited soup transport nearby is an
            # identity-insufficient control, not a claimed camera occlusion.
            kind = "identity_unknown"
        labels.setdefault(kind, {"release": step, "end": step + 30,
                                 "other_in_before": bool(left["audit"]["goals"][SAUCE])})
    return labels


def evaluate(rows):
    events, trail = replay(rows)
    target_start, censored = stable_positive(trail)
    negatives = negative_labels(trail)
    by_step = {r["step"]: r["audit"] for r in trail}
    scores = {}
    for name, all_events in events.items():
        assertions = [e for e in all_events if e["event"] == "in_basket"]
        errors = []
        for e in assertions:
            a = by_step[e["step"]]
            if not a["goals"][SOUP] or a["held_soup"]:
                errors.append(e)
        hits = [e["step"] for e in assertions if target_start is not None
                and target_start <= e["step"] <= target_start + 30 and e not in errors]
        scores[name] = {"hit": min(hits, default=None), "false": errors,
                        "commits": assertions, "event_counts": {
                            k: sum(e["event"] == k for e in all_events)
                            for k in sorted({e["event"] for e in all_events})}}
        if name == "identity":
            scores[name]["trace"] = all_events
    return {"positive_step": target_start, "terminal_censored": censored, "negative_windows": negatives,
            "frames_verified": len(trail), "scores": scores}


def cached_development():
    folder = ROOT / "soup-study" / "probe-output"
    original = {r["layout"]: r for r in read(folder / "probe.json")}
    results = []
    for seed in (925109, 925111):
        src = original[seed]
        samples = {d["step"]: d for d in src["samples"]}
        steps = [s["step"] for s in src["anchors"]]
        previous = min(samples) - 1
        rows = []
        for step in steps:
            state = samples[step]
            actions = [[0] * 6 + [samples[t]["action_grip"]] for t in range(previous + 1, step + 1)]
            previous = step
            rows.append((step, folder / str(seed) / "frames" / f"{step}.npz", actions, {"target": SOUP,
                         "audit": {"goals": {SOUP: state["audit_in"], SAUCE: False},
                                   "held_soup": state["audit_held"]}}))
        events, trail = replay(rows)
        by_step = {d["step"]: d["audit"] for d in trail}
        results.append({"layout": seed, "events": {
            name: [e | ({"true_at_step": by_step[e["step"]]["goals"][SOUP],
                          "held_at_step": by_step[e["step"]]["held_soup"]} if e["event"] == "in_basket" else {})
                   for e in all_events] for name, all_events in events.items()}})
    return results


def trace_rows(folder):
    """Read exact saved controller observations, in the recorded action order."""
    trace = folder / "trace"
    pixels = folder / "frames"
    if not (pixels / "0.npz").is_file():
        pixels = trace / "frames"
    fixture = read(folder / "fixture.json")
    yield 0, pixels / "0.npz", [], {**fixture["initial"], "audit": fixture["initial_audit"]}
    with (trace / "actions.jsonl").open() as stream:
        last = 0
        for row in map(json.loads, stream):
            assert row["start_step"] == last and row["step"] == last + len(row["actions"]), folder
            last = row["step"]
            yield last, pixels / f"{last}.npz", row["actions"], row
        assert last == fixture["end"]["step"], folder


def scripted_development():
    root = ROOT / "soup-study" / "challenge" / "dev"
    result = []
    for seed in (926090, 926091):
        pairs = read(root / str(seed) / "pair.json")
        for label in pairs:
            frames = root / str(seed) / label["condition"] / "frames"
            samples = {row["step"]: row for row in label["samples"]}
            steps = sorted(int(p.stem) for p in frames.glob("*.npz") if int(p.stem) % 10 == 0)
            rows = []
            previous = 0
            for step in steps:
                state = samples.get(step, {"in": False, "held": False})
                actions = [[0] * 6 + [samples[t]["grip"]] for t in range(previous + 1, step + 1)]
                previous = step
                rows.append((step, frames / f"{step}.npz", actions,
                             {"audit": {"goals": {SOUP: state["in"], SAUCE: False}, "held_soup": state["held"]}}))
            events, timeline = replay(rows)
            truth = {row["step"]: row["audit"] for row in timeline}
            result.append({"layout": seed, "condition": label["condition"], "final_predicate": label["final_predicate"],
                           "events": {name: [e | ({"true_at_step": truth[e["step"]]["goals"][SOUP],
                                                   "held_at_step": truth[e["step"]]["held_soup"]}
                                                  if e["event"] == "in_basket" else {}) for e in all_events]
                                      for name, all_events in events.items()}})
    return result


def aggregate(results):
    keys = ("outside", "near_rim_high", "other_can", "identity_unknown")
    negative_counts = {key: sum(key in r["negative_windows"] for r in results) for key in keys}
    positive = sum(r["positive_step"] is not None for r in results)
    metrics = {}
    for name in observers():
        method = [r["scores"][name] for r in results]
        hits = sum(r["hit"] is not None for r in method)
        errors = sum(len(r["false"]) for r in method)
        metrics[name] = {"30_step_hits": hits, "false": errors,
                         "delay_steps": [r["scores"][name]["hit"] - r["positive_step"] for r in results
                                         if r["scores"][name]["hit"] is not None]}
    negative_layouts = sum(bool(r["negative_windows"]) for r in results)
    coverage = positive >= 8 and negative_layouts >= 16 and all(n >= 4 for n in negative_counts.values())
    gate = coverage and metrics["identity"]["30_step_hits"] >= math.ceil(.75 * positive) and metrics["identity"]["false"] == 0
    return {"positive": positive, "terminal_censored": sum(r["terminal_censored"] for r in results),
            "negative_counts": negative_counts, "negative_layouts": negative_layouts,
            "metrics": metrics, "coverage": coverage, "gate": gate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("develop", "blind"))
    p.add_argument("collection", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    source = {name: sha(HERE / name) for name in ("eval.py", "tracker.py", "collect.py")}
    if args.phase == "develop":
        write(args.out, {"source": source, "known": cached_development(), "scripted": scripted_development()})
        return
    freeze = read(HERE / "freeze.json")
    assert source == freeze["source"]
    dev = HERE / "development.json"
    assert sha(dev) == freeze["development_sha256"]
    manifests = [read(args.collection / f"manifest-{card}.json") for card in (2, 3)]
    for i, manifest in enumerate(manifests, start=2):
        assert manifest["card"] == i and manifest["cohort"] == "natural" and manifest["phase"] == "blind"
        assert manifest["complete"] and manifest["collector_sha256"] == source["collect.py"]
        seeds = [job["layout"] for job in manifest["planned"]]
        assert seeds == freeze["layouts"][i - 2::2] and manifest["records"] == [f"{s}/fixture.json" for s in seeds]
    hashes = set()
    results = []
    for seed in freeze["layouts"]:
        folder = args.collection / str(seed)
        fixture = read(folder / "fixture.json")
        assert fixture["layout"] == seed and fixture["initial"]["state_sha256"] not in hashes
        hashes.add(fixture["initial"]["state_sha256"])
        result = evaluate(trace_rows(folder))
        results.append({"seed": seed, **result})
        print(seed, result["positive_step"], list(result["negative_windows"]),
              result["scores"]["identity"]["hit"], flush=True)
    assert len({r["seed"] for r in results}) == len(results)
    summary = aggregate(results)
    write(args.out, {"source": source, "manifests": manifests, **summary, "results": results})
    print({k: summary[k] for k in ("positive", "negative_counts", "metrics", "coverage", "gate")}, flush=True)


if __name__ == "__main__":
    main()
