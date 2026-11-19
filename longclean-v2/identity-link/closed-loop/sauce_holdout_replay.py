"""Locate the RGB-D evidence loss in the recorded 931401 sauce release."""

import argparse
from pathlib import Path

import numpy as np

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import SAUCE, read, sha, write
from actual_physics_worker import worker_class
from basket_stall_probe import archive, replay_row, replay_rows, verify_sources
from dense_online_diagnostic import checked, jsonl, load_arm
from dense_release_observer import DenseReleaseObserver, OPEN_GAP


LAYOUT, START, LAST, CUTOFF = 931401, 250, 280, 300
SUMMARY_SHA = "b3d8f538150a80809df5479957fcf379bf1dae13ab80becd017f9dcc8a2655ca"


def feature_counts(frozen, frame, pending, index):
    world = frozen.cloud(frame, index)
    colored = frozen.colored_pocket(frame, index, SAUCE, pending.origin)
    offset = world - frame.geometry["eef"]
    away_from_hand = (offset[..., 2] < -.073) | (np.linalg.norm(offset[..., :2], axis=-1) > .065)
    if index == 0:
        new = pending.baseline.geometry["depths"][0] - frame.geometry["depths"][0] > .009
    else:
        new = np.zeros(colored.shape, dtype=bool)
        if np.any(colored):
            new[colored] = pending.old_points.query(world[colored])[0] > .012
    return {"colored_pocket_pixels": int(colored.sum()), "new_before_hand_exclusion": int((colored & new).sum()),
            "after_hand_exclusion_before_change": int((colored & away_from_hand).sum()),
            "final_pixels": int((colored & away_from_hand & new).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=GPU, required=True, type=int)
    for name in ("study", "source", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    verify_sources()
    import run as api

    root = args.source.resolve()
    summary = read(checked(root, "summary.json", SUMMARY_SHA))
    case, = [row for row in summary["rows"] if row["layout_seed"] == LAYOUT]
    source = load_arm(root, {**case["baseline"], "layout_seed": LAYOUT, "arm": "adaptive1"})
    original = replay_rows(source, CUTOFF)
    prefix = source["result"]["fork"]["step"]
    observer = DenseReleaseObserver()
    frozen = observer._frozen
    worker_type = worker_class(api)
    output = args.output.resolve()
    captured, events = [], []
    with worker_type(bddl, output, LAYOUT) as worker:
        require(api.fingerprint(worker.current) == source["result"]["initial"], "Source initial state changed")
        for old in original:
            applied = replay_row(worker, old)
            step = old["step"]
            if step < prefix:
                continue
            frame = api.Frame.read(worker.npz)
            if step == prefix:
                observer.start(frame, step)
                continue
            pending = observer._coarse.pending
            event_rows = observer.observe(frame, applied, step)
            events.extend({**e, "step": step} for e in event_rows)
            if START <= step <= LAST:
                pending = pending or observer._coarse.pending
                require(pending is not None and pending.target == SAUCE and pending.step == START,
                        "Expected the same recorded release window")
                measure = [feature_counts(frozen, frame, pending, i) for i in range(2)]
                counts, center = frozen.deposit(frame, pending)
                require(counts == [c["final_pixels"] for c in measure], "Evidence breakdown disagrees with the observer")
                captured.append({"step": step, "holdout_observation": archive(api, worker, output, f"release-{step}.npz"),
                                 "gap_m": frame.gap, "open_gate": frame.gap >= OPEN_GAP, "channels": measure,
                                 "deposit_center_found": center is not None, "events": event_rows})
        terminal = api.fingerprint(worker.current)
    require(jsonl(output / "actions.jsonl") == original, "Published replay differs from the original")
    actual = jsonl(output / "grasp-audit.jsonl")
    require(actual == [source["physical"][s] for s in (0, *[r["step"] for r in original])], "Unpaired physical replay")
    physical = {r["step"]: r for r in actual}
    require([r["step"] for r in captured] == list(range(START, LAST + 1)), "Missing one-step release samples")
    archived = [e for e in source["result"]["state_events"] if e["step"] <= CUTOFF]
    require(len(archived) == len(events) and all(all(left.get(k) == v for k, v in right.items())
            for left, right in zip(archived, events, strict=True)), "Replay changed the frozen observer outputs")
    for row in captured:
        sample = physical[row["step"]]
        row["evaluation_only"] = {"official_goal": sample["goals"][SAUCE], "held": sample["held_by_target"][SAUCE]}
    write(output / "summary.json", {"schema": "v2-holdout-sauce-release-diagnostic/1", "layout": LAYOUT,
          "source_sha256": SUMMARY_SHA, "producer_sha256": sha(Path(__file__)), "source_result_sha256": source["entry"]["result_sha256"],
          "exact_replay_intervals": len(original), "terminal": terminal,
          "actions_sha256": sha(output / "actions.jsonl"), "grasp_audit_sha256": sha(output / "grasp-audit.jsonl"),
          "thresholds_pixels_head_wrist": [34, 45], "capture": captured})
    print("SAUCE_HOLDOUT_REPLAY", len(captured), len(original), flush=True)


if __name__ == "__main__":
    main()
