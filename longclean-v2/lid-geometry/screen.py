"""Sensor-only lid placement evidence and archived-view screening."""
import argparse
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import read, sha, write
from vision import Frame, cloud
from lid import inspect_lids


def contained(lid):
    """Conservative basket interior for the fixed scene-01 camera task."""
    x, y, z = lid["center_m"]
    clearance = min(x + .055, .065 - x, y - .195, .335 - y)
    return bool(.470 < z < .575 and clearance >= .55 * lid["radius_m"])


def choose(frame, anchor):
    proposals = inspect_lids(frame)
    if frame.gap < .067:
        return proposals, None
    eligible = [p for p in proposals if contained(p)
                and np.linalg.norm(np.array(p["center_m"][:2]) - anchor) <= .080
                and frame.geometry["eef"][2] - p["center_m"][2] >= .065]
    return proposals, min(eligible, key=lambda p: np.linalg.norm(np.array(p["center_m"][:2]) - anchor), default=None)


def corresponding(frame, proposal, mask):
    """Evaluator only: measure whether the chosen region really belongs to soup."""
    if proposal is None or mask is None or not mask.any():
        return None
    points = cloud(frame, 1)[mask]
    dist = np.linalg.norm(points - np.array(proposal["center_m"]), axis=1)
    return bool((dist < .012).sum() >= 5)


def episode(folder, steps, mask_read=False, anchor=None):
    last = None
    series, commits = [], []
    steps = sorted(steps)
    for step in steps:
        frame_path = folder / f"{step}.npz"
        frame = Frame.read(frame_path)
        if anchor is None:
            # The archived runs lift and hold in place after opening. Recover the
            # release neighborhood from the first observation, not simulator truth.
            anchor = np.array(frame.geometry["eef"][:2])
        proposals, choice = choose(frame, anchor)

        # Score a sensor-only decision. This ground-truth information never
        # changes proposal selection or accumulation.
        mask_path = folder / f"{step}-mask.npy"
        mask = np.load(mask_path, allow_pickle=False) if mask_read else None
        matched = corresponding(frame, choice, mask)
        current = {"step": step, "grip_gap": frame.gap, "proposals": proposals, "choice": choice,
                   "soup_match": matched}
        series.append(current)
        if choice and last and last["choice"] and 5 <= step - last["step"] <= 20:
            prev = last["choice"]
            if (np.linalg.norm(np.array(choice["center_m"]) - prev["center_m"]) <= .022
                    and abs(choice["radius_m"] - prev["radius_m"]) <= .012):
                commits.append({"step": step, "previous_step": last["step"],
                                "soup_match": bool(matched and last["soup_match"]) if mask_read else None})
        last = current
    return {"series": series, "commits": commits}


def known():
    folder = ROOT / "soup-study/probe-output"
    original = {p["layout"]: p["samples"] for p in read(folder / "probe.json")}
    cases = []
    for seed, steps in [(925109, (430, 440, 450)), (925111, (390, 400, 410))]:
        result = episode(folder / str(seed) / "frames", steps)
        audit = {s["step"]: (s["audit_in"], s["audit_held"]) for s in original[seed]}
        cases.append({"layout": seed, "audit": [{"step": s, "in": audit[s][0], "held": audit[s][1]} for s in steps],
                      **result})
    return cases


def archive():
    active = ROOT / "active-look" / "blind-output"
    cases = []
    # These layouts have already been examined for color evidence and remain
    # a feasibility screen. An independent test needs new layouts.
    for gpu in (2, 3):
        report = read(active / f"card-{gpu}.json")
        for case in report["pairs"]:
            arm = case["arms"]["vertical"]
            if not arm["eligible"]:
                cases.append({"layout": case["layout"], "condition": case["condition"], "eligible": False})
                continue
            folder = active / f'{case["layout"]}-{case["condition"]}' / "vertical"
            assert not arm["goals_changed_or_soup_regrasped_at"], folder
            steps = [s["step"] for s in arm["samples"]]
            cases.append({"layout": case["layout"], "condition": case["condition"], "eligible": True,
                          **episode(folder, steps, mask_read=True)})
    return sorted(cases, key=lambda row: (row["layout"], row["condition"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    args = p.parse_args()
    cases = archive()
    inside = [c for c in cases if c["condition"] == "inside" and c["eligible"]]
    outside = [c for c in cases if c["condition"] == "outside" and c["eligible"]]
    hits = [c for c in inside if c["commits"] and c["commits"][0]["soup_match"]]
    wrong_positive = [c["layout"] for c in inside if c["commits"] and not c["commits"][0]["soup_match"]]
    false = [c["layout"] for c in outside if c["commits"]]
    fitted = [c["series"][0]["choice"]["radius_m"] for c in hits if c["series"][0]["choice"]]
    gate = len(inside) == 8 and len(outside) == 4 and len(hits) >= 6 and not (false or wrong_positive)
    report = {"source": {n: sha(HERE / n) for n in ("lid.py", "screen.py")},
              "inside": len(inside), "outside": len(outside), "inside_hits": len(hits),
              "inside_wrong_object": wrong_positive, "outside_false": false, "pass": gate,
              "median_first_radius_m": median(fitted) if fitted else None,
              "known_development": known(), "cases": cases}
    write(args.out, report)
    print(f'archive screen {len(hits)}/{len(inside)}, outside false {len(false)}/{len(outside)}, '
          f'wrong-object {wrong_positive}, gate {gate}', flush=True)


if __name__ == "__main__":
    main()
