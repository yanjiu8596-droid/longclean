"""Measure sampling aliasing without changing release or landing thresholds."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SOUP, array_hash, read, sha, write
from tracker import Pending, evidence
from vision import Frame


def checked_frame(path, expected):
    with np.load(path, allow_pickle=False) as raw:
        if not expected or set(raw) != set(expected):
            raise ValueError(f"Incomplete archived hashes: {path}")
        if any(array_hash(raw[name]) != digest for name, digest in expected.items()):
            raise ValueError(f"Observation hash mismatch: {path}")
    return Frame.read(path)


def replay(samples):
    """Run the frozen two-sample landing rule on a timestamped candidate stream."""
    prior = None
    for row in samples:
        if row["reset"]:
            break
        center = row["center"]
        if center is None:
            if row["reason"] == "ambiguous_landing":
                prior = None
            continue
        if prior is not None:
            delay = row["step"] - prior["step"]
            close = np.linalg.norm(np.subtract(center, prior["center"])) <= .025
            same = row["source"] == prior["source"] and close
            if same and 5 <= delay <= 20:
                return {"step": row["step"], "prior_step": prior["step"],
                        "source": row["source"], "true": row["true"]}
            if same and delay <= 20:
                continue
        prior = row
    return None


def audit_case(case, dense, natural, original, mask_enabled):
    seed = case["seed"]
    directory = dense / str(seed)
    if sha(directory / "dense.json") != case["dense_sha256"]:
        raise ValueError(f"Archive hash mismatch: {seed}")
    record = read(directory / "dense.json")
    release = record["release"]
    rows = {r["step"]: r for r in record["records"]}
    if sorted(rows) != list(range(record["window"][0], record["window"][1] + 1)):
        raise ValueError(f"Missing dense step: {seed}")
    origin = np.asarray(release["origin_xy_m"])
    stamp = release["step"]
    old = natural / str(seed)
    with (old / "trace" / "actions.jsonl").open(encoding="utf-8") as source:
        import json
        blocks = {row["step"]: row for row in map(json.loads, source)}
    fixture = read(old / "fixture.json")
    blocks[0] = fixture["initial"]
    before = tuple((t, checked_frame(old / "frames" / f"{t}.npz",
                                     blocks[t]["observation_sha256"]))
                   for t in release["baseline_steps"])
    released = checked_frame(directory / "frames" / f"{stamp}.npz", rows[stamp]["observation_sha256"])
    window = Pending(stamp, origin, released.geometry["eef"], before)
    archived = {e["step"]: e for e in original["scores"]["identity"]["trace"]
                if e["event"] == "identity_landing_sample"}
    samples = []
    for step, row in sorted(rows.items()):
        if step <= stamp:
            continue
        path = directory / "frames" / f"{step}.npz"
        frame = checked_frame(path, row["observation_sha256"])
        candidate, reason = None, None
        # Frozen inputs are batched for archived sample points, one realized action
        # otherwise. The release trigger and its original baseline stay fixed.
        opening = any(a[-1] < -.5 for a in row["actions"])
        grid_opening = (any(a[-1] < -.5 for a in blocks[step]["actions"])
                        if row["archived_anchor"] else opening)
        eligible = frame.gap >= .067 and step - stamp >= 3
        if eligible:
            candidate, _, reason = evidence(frame, window)
        tag, center = (None, None) if candidate is None else (candidate[0], candidate[1].tolist())
        if step in archived and (tag, reason) != (archived[step]["candidate"], archived[step]["reason"]):
            raise RuntimeError(f"{seed}:{step} frozen candidate/reason drift")
        truth = row["audit"]
        pixels = None
        if mask_enabled:
            soup_mask = np.load(directory / "frames" / f"{step}-mask.npy", allow_pickle=False)
            sauce_mask = np.load(directory / "frames" / f"{step}-sauce-mask.npy", allow_pickle=False)
            pixels = {"soup": int(np.count_nonzero(soup_mask)), "sauce": int(np.count_nonzero(sauce_mask))}
        samples.append({"step": step, "grid": row["archived_anchor"],
                        "reset": not opening and frame.gap < .061,
                        "reset_grid": not grid_opening and frame.gap < .061,
                        "source": tag, "center": center, "reason": reason, "pixels": pixels,
                        "true": bool(truth["goals"][SOUP] and not truth["held_soup"])})
    coarse = replay([{**s, "reset": s["reset_grid"]} for s in samples if s["grid"]])
    fine = replay(samples)
    if (coarse["step"] if coarse else None) != record["frozen_hit"]:
        raise RuntimeError(f"{seed}: archived landing replay did not match ({coarse}, {record['frozen_hit']})")
    with_truth = [s for s in samples if s["true"] and s["pixels"] is not None]
    return {"seed": seed, "positive_step": record["positive_step"], "release_step": stamp,
            "anchors": len(record["verified_anchor_steps"]), "coarse": coarse, "fine": fine,
            "grid_candidates": [s["step"] for s in samples if s["grid"] and s["source"]],
            "between_grid_candidates": [s["step"] for s in samples if not s["grid"] and s["source"]],
            "wrist_soup_pixels_median": (float(np.median([s["pixels"]["soup"] for s in with_truth]))
                                        if with_truth else None),
            "wrist_soup_pixels_peak": max((s["pixels"]["soup"] for s in with_truth), default=None),
            "samples": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("natural", type=Path)
    p.add_argument("original_report", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("dense", nargs="+", type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    originals = {r["seed"]: r for r in read(args.original_report)["results"]}
    cases, manifests = [], []
    for source in args.dense:
        manifest_path = source / "dense-summary.json"
        manifest = read(manifest_path)
        if not manifest["complete"] or manifest["report_sha256"] != sha(args.original_report):
            raise ValueError(f"Invalid dense replay manifest: {manifest_path}")
        manifests.append({"name": source.name, "sha256": sha(manifest_path), "card": manifest["card"]})
        for case in manifest["cases"]:
            cases.append(audit_case(case, source, args.natural, originals[case["seed"]],
                                    bool(manifest["registration_code_sha256"])))
    if len({c["seed"] for c in cases}) != len(cases):
        raise ValueError("Duplicate dense replay seed")
    count = Counter(("both" if c["coarse"] and c["fine"] else "coarse_only" if c["coarse"]
                     else "dense_only" if c["fine"] else "neither") for c in cases)
    print(dict(count), flush=True)
    write(args.out, {"manifests": manifests, "comparison": dict(count), "results": cases,
                     "code_sha256": sha(Path(__file__)), "original_sha256": sha(args.original_report)})


if __name__ == "__main__":
    main()
