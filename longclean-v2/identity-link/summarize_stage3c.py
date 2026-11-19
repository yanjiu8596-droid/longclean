"""Archive compact stage 3c results without opening sealed candidate scores."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("collections", nargs="+", type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    diagnosis = read(Path("landing-fits.json"))
    timing = read(Path("dense-timing.json"))
    counts, eligible, seen, cohorts = Counter(), Counter(), set(), []
    for root in args.collections:
        status = int(root.with_suffix(".exit").read_text())
        paths = sorted(root.glob("manifest-*.json"))
        if status or len(paths) != 1:
            raise RuntimeError(f"Incomplete collection {root}: {status}")
        path = paths[0]
        m = read(path)
        if not m["complete"] or [(r["layout"], r["case"]) for r in m["planned"]] != [
                (r["layout"], r["case"]) for r in m["records"]]:
            raise RuntimeError(f"Plan and records differ in {root}")
        for record in m["records"]:
            fixture = read(root / record["path"])
            if (fixture["initial"]["state_sha256"] != record["initial_sha256"]
                    or fixture["release_quality"]["eligible"] != record["eligible"]
                    or fixture["release_quality"]["checks"] != record["checks"]):
                raise ValueError(f"Manifest does not match fixture: {root} {fixture['layout']}")
            if m["phase"] == "blind":
                key = (record["layout"], record["initial_sha256"])
                if any(layout == key[0] or digest == key[1] for layout, digest in seen):
                    raise RuntimeError(f"Reused blind layout: {key}")
                seen.add(key)
                counts[record["case"]] += 1
                eligible[record["case"]] += int(record["eligible"])
        cohorts.append({"name": root.name, "manifest_sha256": sha(path), "exit": status, **m})
    dense = [{k: v for k, v in r.items() if k != "samples"} for r in timing["results"]]
    write(args.out, {"original_sha256": timing["original_sha256"],
                     "diagnosis_sha256": sha(Path("landing-fits.json")), "diagnosis": diagnosis["summary"],
                     "dense_timing_sha256": sha(Path("dense-timing.json")), "dense": dense,
                     "dense_comparison": timing["comparison"],
                     "blind_attempts": dict(counts), "blind_eligible": dict(eligible), "cohorts": cohorts})


if __name__ == "__main__":
    main()
