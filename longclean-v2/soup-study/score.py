"""Score old and candidate on identical sampled frames; labels are read only afterward."""
from itertools import islice
from pathlib import Path
import sys

from probe import ROOT
from common import SOUP, read, sha, write
from observer import PlacementObserver
from vision import Frame
from wrist import WristSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import BLIND, DEV


def state_events(frame_dir, labels, sample_steps, grip_key):
    observers = {"base": PlacementObserver(), "wrist": WristSoup()}
    results = {key: [] for key in observers}
    start = min(labels) - 1
    for step in sample_steps:
        frame = Frame.read(frame_dir / f"{step}.npz")
        actions = [[0] * 6 + [labels[t][grip_key]] for t in range(start + 1, step + 1)]
        start = step
        for key, observer in observers.items():
            results[key].extend({"step": step, **event} for event in observer.observe(frame, actions, step)
                                if event["object"] == SOUP)
    return results


def reference_result(label, steps, events):
    release = label["release"]
    track = {d["step"]: d for d in label["samples"]}
    positive = next((t for t in sorted(track) if release is not None and t >= release
                     and track[t]["in"] and not track[t]["held"]
                     and all(track[s]["in"] and not track[s]["held"] for s in range(t, t + 21) if s in track)
                     and t + 20 <= label["num_steps"]), None)
    negative = (release is not None and label["num_steps"] >= release + 30
                and any(not track[t]["held"] for t in range(release, release + 31))
                and not any(d["in"] and not d["held"] for t, d in track.items() if t >= release))
    valid = (label["grasped"] and label["held_at_release"]
             and ((label["condition"] == "inside" and positive is not None and label["final_predicate"]
                   and any(positive + 10 <= s <= positive + 30 for s in steps))
                  or (label["condition"] == "outside" and negative)))
    methods = {}
    for key, series in events.items():
        commits = [e for e in series if e["event"] == "in_basket"]
        errors = [e for e in commits if not track[e["step"]]["in"] or track[e["step"]]["held"]]
        hits = [e["step"] for e in commits if positive is not None and positive <= e["step"] <= positive + 30
                and track[e["step"]]["in"] and not track[e["step"]]["held"]]
        methods[key] = {"false": errors, "hit_step": min(hits, default=None), "trace": series}
    return {key: label[key] for key in ("layout", "condition", "initial", "release", "final_predicate")} | {
        "valid": valid, "grasped": label["grasped"], "held_at_release": label["held_at_release"],
        "first_stable_released": positive, "negative": negative, "methods": methods,
    }


def report_known(out):
    traces = read(out / "probe.json")
    reports = []
    for trace in traces:
        seed = trace["layout"]
        labels = {d["step"]: d for d in trace["samples"]}
        steps = [a["step"] for a in trace["anchors"]]
        results = state_events(out / str(seed) / "frames", labels, steps, "action_grip")
        commit = {key: [e | {"truth_in": labels[e["step"]]["audit_in"],
                             "truth_held": labels[e["step"]]["audit_held"]}
                        for e in series if e["event"] == "in_basket"] for key, series in results.items()}
        reports.append({"layout": seed, "source_anchors": steps, "methods": commit, "decisions": results})
    return reports


def main():
    phase, root, out = sys.argv[1:]
    assert phase in ("dev", "independent")
    root, out = Path(root).resolve(), Path(out).resolve()
    src = ROOT / "soup-study"
    sources = {name: sha(src / name) for name in ("collect.py", "probe.py", "score.py", "wrist.py")}
    old_manifest = read(ROOT / "placement-next/candidate-freeze.json")
    assert sha(ROOT / "placement-next/observer.py") == old_manifest["source"]["observer.py"]
    if phase == "independent":
        frozen = read(src / "freeze.json")
        assert sources == frozen["source"]
        assert list(BLIND) == frozen["layouts"]
        assert sha(src / "dev-result.json") == frozen["dev_sha256"]
    seeds = DEV if phase == "dev" else BLIND
    paired = []
    initials = set()
    for seed in seeds:
        folder = root / phase / str(seed)
        pair = read(folder / "pair.json")
        assert pair[0]["initial"] == pair[1]["initial"] and pair[0]["initial"] not in initials
        with (folder / "inside/actions.jsonl").open() as inside, (folder / "outside/actions.jsonl").open() as outside:
            assert list(islice(inside, 146)) == list(islice(outside, 146)), seed
        initials.add(pair[0]["initial"])
        branches = {}
        for label in pair:
            condition = label["condition"]
            paths = folder / condition / "frames"
            # Keep the 10-step observation timing used in the earlier validation.
            steps = sorted(int(p.stem) for p in paths.glob("*.npz") if int(p.stem) % 10 == 0)
            labels = {d["step"]: d for d in label["samples"]}
            results = state_events(paths, labels, steps, "grip")
            branches[condition] = reference_result(label, steps, results)
        paired.append(branches)
    good = [pair for pair in paired if all(row["valid"] for row in pair.values())]
    counts = {key: {"hits": sum(p["inside"]["methods"][key]["hit_step"] is not None for p in good),
                    "false": sum(len(b["methods"][key]["false"]) for p in paired for b in p.values()),
                    "valid_negative_false": sum(len(p["outside"]["methods"][key]["false"]) for p in good)}
              for key in ("base", "wrist")}
    gain = sum(p["inside"]["methods"]["base"]["hit_step"] is None
               and p["inside"]["methods"]["wrist"]["hit_step"] is not None for p in good)
    gate = len(good) >= 5 and gain >= 3 and counts["wrist"]["false"] == 0
    output = {"phase": phase, "source": sources, "pairs": paired, "valid_pairs": len(good),
              "counts": counts, "gains": gain, "challenge_gate": gate if phase == "independent" else None}
    if phase == "dev":
        output["known_replays"] = report_known(src / "probe-output")
    write(out, output)
    print(phase, len(good), counts, "gain", gain, "gate", output["challenge_gate"], flush=True)


if __name__ == "__main__":
    main()
