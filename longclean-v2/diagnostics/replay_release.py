"""Replay frozen action logs, then optionally extend one preterm release as a probe."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.out.resolve()
    sys.path.insert(0, str(study.parent))

    import numpy as np
    from common import CHEESE, LABELS, SAUCE, SOUP, read, sha, write
    from run import Worker, fingerprint, replay_prefix
    from vision import Frame, Progress, basket_pixels, cloud, color_mask, load_locator

    frozen = read(study / "frozen.json")
    if any(sha(study.parent / name) != expected for name, expected in frozen["code"].items()):
        raise RuntimeError("The replay runner does not match the frozen pilot")
    bddl = study / read(study / "plan.json")["bddl"]
    locate = load_locator()
    probes = (
        ("924122-s34-moved", SOUP, 350),
        ("924122-s35-moved", SAUCE, 230),
        ("924130-s34-sham", SOUP, 470),
        ("924109-s34-moved", SOUP, 386),
        ("924109-s35-moved", SOUP, 382),
        ("924101-s34-moved", SAUCE, 250),
    )
    reports = []
    for tag, obj, event_step in probes:
        trace = study / "pilot" / tag / "replan"
        history = read(trace / "result.json")
        fixture = study / "fixtures" / str(history["layout_seed"])
        tracker = Progress(locate)
        events, window, continuation = [], [], []

        def diagnostic(row, frame, origin):
            target_surface = []
            for i, image in enumerate(frame.images):
                points = cloud(frame, i)[basket_pixels(frame, i, origin) & color_mask(image, LABELS[obj])]
                target_surface.append({
                    "count": len(points),
                    "xyz_median": np.median(points, axis=0).tolist() if len(points) else None,
                })
            audit = row["audit"]
            pos = audit["positions"][obj]
            basket = audit["positions"]["basket_1"]
            return {
                "step": row["step"], "target": obj, "assigned": row["target"],
                "eef": frame.geometry["eef"].tolist(), "gap": frame.gap,
                "opening_actions": sum(a[-1] < -.5 for a in row["actions"]),
                "sensor_target_surface": target_surface,
                "audit_xyz": pos, "audit_height_above_basket_origin": pos[2] - basket[2],
                "audit_in_basket": audit["goals"][obj], "audit_held_soup": audit["held_soup"],
                "audit_transitions": row["transitions"],
            }

        with Worker(bddl, out / tag, history["layout_seed"]) as worker:
            replay_prefix(worker, read(fixture / "prefix.json"), tracker)
            if history["moved"]:
                worker.call({"op": "move", "offset": read(fixture / "fixture.json")["offset"]})
            if fingerprint(worker.current) != history["trial_checkpoint"]:
                raise RuntimeError(f"Checkpoint differs: {tag}")
            fork = worker.current["step"]
            origin = worker.current["audit"]["site_xyz"][:2]
            checked = 0
            with (trace / "actions.jsonl").open() as stream:
                for row in map(json.loads, stream):
                    if row["start_step"] < fork:
                        continue
                    actions = worker.step(row["actions"], row["target"], row["instruction"])
                    if (fingerprint(worker.current) != fingerprint(row)
                            or worker.current["audit"] != row["audit"]
                            or worker.current["transitions"] != row["transitions"]):
                        raise RuntimeError(f"Action replay differs: {tag}:{row['step']}")
                    checked += 1
                    frame = Frame.read(worker.npz)
                    if row["step"] <= event_step - 5 and row["target"] == obj:
                        origin = frame.geometry["eef"][:2]
                    event = tracker.observe(frame, row["target"], actions)
                    if event:
                        events.append({"step": row["step"], "matches_audit": worker.goal(row["target"]), **event})
                    if abs(event_step - row["step"]) <= 35:
                        window.append(diagnostic(row, frame, origin))
                    if row["step"] >= event_step + 30:
                        break

            expected = [e for e in history["state_events"] if e["step"] <= worker.current["step"]]
            if events != expected or not any(e["step"] == event_step for e in events):
                raise RuntimeError(f"Frozen observer evidence differs: {tag}")
            if history["terminal_reason"] == "observed_complete" and not worker.terminal():
                for _ in range(3):
                    actions = worker.step([[0, 0, 0, 0, 0, 0, -1]] * 10, obj, "diagnostic open-gripper hold")
                    row = {**worker.current, "target": obj, "actions": actions}
                    continuation.append(diagnostic(row, Frame.read(worker.npz), origin))
                    if worker.terminal():
                        break
        reports.append({"tag": tag, "blocks_exactly_replayed": checked, "events": events,
                        "event_step": event_step, "window": window, "open_hold_probe": continuation})
        print(tag, checked, "verified", flush=True)
    write(out / "release-replay.json", reports)


if __name__ == "__main__":
    main()
