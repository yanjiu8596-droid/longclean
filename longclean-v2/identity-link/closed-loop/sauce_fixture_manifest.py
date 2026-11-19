"""Print a read-only plan for the 16 archived actual-sauce control attempts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


SAUCE = "tomato_sauce_1"
FREEZE_SHA = "4b72ee2bed52a346b4329c0577519843fe5891a33e8a276973bedbe415914a7c"
BDDL_SHA = "32f2219ef8e19d4ffe1c68ad17a2493dcf0b8601a6d143f2866bab06aff8a0ac"
GROUPS = {"counter-blind": ("other_can", "sauce_inside", range(929336, 929344)),
          "counter-unknown": ("identity_unknown", "sauce_outside", range(929344, 929352))}


def require(condition, why):
    if not condition:
        raise ValueError(why)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def attempt(root, collection, card, category, treatment, row):
    seed = row["layout"]
    scope = collection.parent / str(seed)
    fixture_path = scope / "fixture.json"
    require(row["path"] == f"{seed}/fixture.json" and row["case"] == category, f"Unexpected record: {seed}")
    fixture = read(fixture_path)
    physics_path = scope / "physics-audit.json"
    actions_path = scope / "trace" / "actions.jsonl"
    physics = read(physics_path)
    quality = physics["release_quality"]
    require(fixture["layout"] == seed and fixture["case"] == category and fixture["card"] == card
            and fixture["initial"]["state_sha256"] == row["initial_sha256"]
            and fixture["initial"]["bddl_sha256"] == fixture["end"]["bddl_sha256"] == BDDL_SHA
            and fixture["physics_audit"] == "physics-audit.json" and fixture["actions"] == "trace/actions.jsonl"
            and fixture["frames"] == "frames" and fixture["release_quality"] == quality
            and all(row[k] == v for k, v in quality.items()) and physics["controlled_object"] == SAUCE,
            f"Fixture identity, object or qualification differs: {seed}")
    records = physics["steps"]
    final = fixture["end"]["step"]
    start, opening = (quality[k] for k in ("open_command_after_step", "measured_open_step"))
    require([r["step"] for r in records] == list(range(final + 1)) and type(start) is type(opening) is int
            and 0 <= start < opening <= final - 30 and records[start]["held"][SAUCE] is True,
            f"Missing sampled physical grasp or 30-step release window: {seed}")
    after = records[opening:opening + 31]
    require(not after[-1]["held"][SAUCE], f"Sauce did not detach by the end of the scored window: {seed}")
    if treatment == "sauce_inside":
        require(all(r["goals"][SAUCE] and not r["held"][SAUCE] for r in after[-20:]),
                f"Sauce positive does not stay inside and unheld: {seed}")
    else:
        dx, dy = (after[-1]["positions"][SAUCE][i] - after[-1]["positions"]["basket_1"][i] for i in (0, 1))
        require(not any(r["goals"][SAUCE] for r in after) and (abs(dx) > .070 or abs(dy) > .085),
                f"Sauce outside condition does not match the physical record: {seed}")
    failures = [k for k, passed in quality["checks"].items() if not passed]
    require(quality["eligible"] == (not failures) == (seed != 929345), f"Unexpected archived qualification: {seed}")
    return {"layout_seed": seed, "card": card, "controlled_object": SAUCE, "source_category": category,
            "treatment": treatment, "legacy_qualification": quality["eligible"], "failed_checks": failures,
            "controlled_release_command_step": start, "measured_open_step": opening,
            "first_unheld_since_open_command": next(r["step"] for r in records[start:opening + 31]
                                                    if not r["held"][SAUCE]),
            "scored_native_steps": [opening, opening + 30], "initial_state_sha256": row["initial_sha256"],
            "fixture": fixture_path.relative_to(root).as_posix(), "fixture_sha256": sha(fixture_path),
            "physics_audit": physics_path.relative_to(root).as_posix(), "physics_audit_sha256": sha(physics_path),
            "actions": actions_path.relative_to(root).as_posix(), "actions_sha256": sha(actions_path),
            "frames": (scope / "frames").relative_to(root).as_posix(), "original_frame_steps": fixture["frame_steps"]}


def build(root):
    lock = root / "stage3d-freeze.json"
    require(sha(lock) == FREEZE_SHA, "Changed stage 3d freeze")
    checked = {entry["manifest"]: entry for entry in read(lock)["collections"]}
    archives, trials = [], []
    for folder, (category, treatment, seeds) in GROUPS.items():
        for card in (2, 3):
            name = f"{folder}-{card}/manifest-{card}.json"
            collection = root / name
            expected = checked[name]
            require(sha(collection) == expected["sha256"], f"Changed archived collection: {name}")
            manifest = read(collection)
            actual = [row for row in manifest["records"] if row["case"] == category]
            require(manifest["complete"] and manifest["card"] == card and manifest["frame_stride"] == 5
                    and manifest["bddl_sha256"] == BDDL_SHA and manifest["collector_sha256"] == expected["collector_sha256"]
                    and [r["layout"] for r in actual] == [s for s in seeds if (s % 2) == (card - 2)],
                    f"Changed collection settings or layout allocation: {name}")
            archives.append({"card": card, "manifest": name, "manifest_sha256": expected["sha256"],
                             "collector_sha256": expected["collector_sha256"], "frame_stride": 5})
            trials.extend(attempt(root, collection, card, category, treatment, row) for row in actual)
    trials.sort(key=lambda r: r["layout_seed"])
    require([r["layout_seed"] for r in trials] == list(range(929336, 929352))
            and len({r["initial_state_sha256"] for r in trials}) == 16,
            "A prespecified attempt or unique initial state is missing")
    return {"schema": "v2-sauce-archived-controls-plan/1", "scope": "previously_seen_development_controls",
            "object": SAUCE, "bddl_sha256": BDDL_SHA, "stage3d_freeze_sha256": FREEZE_SHA, "source_archives": archives,
            "counts": {"scheduled": 16, "qualified_sauce_inside": 8, "qualified_sauce_outside": 7, "unqualified": 1},
            "qualification_note": "Retain all 16 for false-assertion auditing. Exclude 929345 only from qualified outside coverage.",
            "attempts": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="identity-link directory containing the sealed stage 3d collections")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
