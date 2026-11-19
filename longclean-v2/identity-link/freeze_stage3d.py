"""Bind the predeclared stage 3d candidate to completed collection manifests."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--with-rim", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or args.plan.parent.resolve() != args.out.parent.resolve():
        raise ValueError("Keep the freeze beside its plan and use a new filename")
    plan = read(args.plan)
    if sha(HERE / "stage3d-ablation.json") != plan["development_sha256"]:
        raise ValueError("Candidate choice does not match development report")
    for name, digest in plan["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"Source changed since candidate selection: {name}")
    declarations = [*plan["collections"], *plan["contingent_rim"]] if args.with_rim else plan["collections"]
    collections = []
    for declared in declarations:
        path = args.plan.parent / declared["manifest"]
        manifest = read(path)
        if (not manifest["complete"] or [r["layout"] for r in manifest["planned"]] != declared["layouts"]
                or manifest["collector_sha256"] != declared["collector_sha256"]):
            raise ValueError(f"Unexpected experiment collection: {path}")
        collections.append({**declared, "sha256": sha(path)})
    previous = read(HERE / "independent-summary.json")
    previous_layouts = [r["seed"] for r in previous["results"]]
    previous_raw = [read(HERE / "natural-output" / str(seed) / "fixture.json")["initial"]["raw_sha256"]
                    for seed in previous_layouts]
    write(args.out, {"schema": "identity-stage3d-freeze-v1", "plan_sha256": sha(args.plan),
                     "source_sha256": plan["source_sha256"], "candidate": plan["candidate"],
                     "collections": collections, "excluded_layouts": previous_layouts,
                     "excluded_state_sha256": [r["initial_sha256"] for r in previous["results"]],
                     "excluded_raw_sha256": previous_raw})


if __name__ == "__main__":
    main()
