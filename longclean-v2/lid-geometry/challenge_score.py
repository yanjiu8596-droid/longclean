"""Score the frozen geometry component on independent controlled RGB-D views."""
from collections import Counter
import argparse
from pathlib import Path

import numpy as np

from screen import HERE, ROOT, Frame, corresponding, episode, read, sha, write

EXPECTED = ("inside", "outside", "rim", "other_can", "soup_offscreen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    frozen = read(HERE / "challenge-freeze.json")
    assert all(sha(HERE / name) == digest for name, digest in frozen["source"].items())
    manifest = read(args.fixtures / "manifest.json")
    assert manifest["source_sha256"] == frozen["source"]["challenge/render.py"]
    assert [layout["seed"] for layout in manifest["layouts"]] == frozen["layouts"]
    assert len(set(row["initial_sha256"] for row in manifest["layouts"])) == len(frozen["layouts"])
    previous = ROOT / "active-look" / "blind-output"
    old_hashes = {p["arms"]["vertical"]["initial"]["state_sha256"] for c in (2, 3)
                  for p in read(previous / f"card-{c}.json")["pairs"]}
    assert not old_hashes & {r["initial_sha256"] for r in manifest["layouts"]}

    trials = {c: Counter() for c in EXPECTED}
    details = []
    for item in manifest["layouts"]:
        seed = item["seed"]
        for name in EXPECTED:
            trials[name]["attempted"] += 1
            if name not in item["cases"]:
                details.append({"seed": seed, "condition": name, "eligible": False,
                                "reason": item.get("reason", "missing case")})
                continue
            folder = args.fixtures / str(seed) / name
            truth = read(folder / "truth.json")
            assert truth["condition"] == name and truth["eligible"] == item["cases"][name]["eligible"]
            if not truth["eligible"]:
                details.append({"seed": seed, "condition": name, "eligible": False, "truth": truth})
                continue
            trials[name]["eligible"] += 1
            assert truth["soup_in"] == (name == "inside")
            steps = [f["step"] for f in truth["frames"]]
            result = episode(folder, steps, mask_read=True)
            first = next(iter(result["commits"]), None)
            if any(row["proposals"] for row in result["series"]):
                trials[name]["surface_detected"] += 1
            if first:
                trials[name]["geometry_triggers"] += 1
                if name == "inside" and first["soup_match"]:
                    trials[name]["correct_soup_trigger"] += 1
                if name != "inside" or not first["soup_match"]:
                    trials[name]["wrong_soup_trigger_if_identity_assumed"] += 1
                if name == "other_can":
                    step = first["step"]
                    choice = next(row["choice"] for row in result["series"] if row["step"] == step)
                    frame = Frame.read(folder / f"{step}.npz")
                    sauce = np.load(folder / f"{step}-sauce-mask.npy", allow_pickle=False)
                    first["matches_other_can"] = corresponding(frame, choice, sauce)
                    if first["matches_other_can"]:
                        trials[name]["trigger_matches_other_can"] += 1
            details.append({"seed": seed, "condition": name, "eligible": True,
                            "first_geometry_trigger": first, "truth": truth, **result})
    coverage = trials["inside"]["eligible"] >= 8 and all(trials[c]["eligible"] >= 4 for c in EXPECTED[1:])
    gate = (coverage and trials["inside"]["correct_soup_trigger"] >= 6
            and not any(trials[c]["wrong_soup_trigger_if_identity_assumed"] for c in EXPECTED))
    write(args.out, {"scope": "controlled view; no transport identity or closed-loop evidence",
                     "coverage": coverage, "geometry_alone_gate": gate,
                     "source": frozen["source"], "trials": trials, "cases": details})
    print({k: dict(v) for k, v in trials.items()}, "coverage", coverage, "geometry_alone_gate", gate, flush=True)


if __name__ == "__main__":
    main()
