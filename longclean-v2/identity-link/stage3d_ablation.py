"""Compare local plane fitting and release-window sampling on identical actions."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from time import process_time_ns
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval import stable_positive, trace_rows
from common import SOUP, array_hash, read, sha, write
from vision import Frame
import tracker

ARMS = ("base", "dense", "plane", "combo")
FROZEN_REPORT_SHA = "df6c03483645c19a61a303e1667d09d114908656cd5a8044f13dea833d3f5d15"


class WindowedObserver:
    """5-step background cadence; stepwise updates only during pending landing."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.dense = arm in ("dense", "combo")
        self.lid = None
        if arm in ("plane", "combo"):
            from plane_lid import inspect_lids
            self.lid = inspect_lids
        self.core = tracker.IdentityObserver()
        self.actions = []
        self.calls = self.off_grid_calls = self.fit_calls = self.processing_ns = self.pending_gaps = 0
        self.last_input_step = 0

    def inspect(self, frame):
        self.fit_calls += 1
        return self.lid(frame)

    def inspect_original(self, frame):
        self.fit_calls += 1
        return self.original_lid(frame)

    def observe(self, frame, actions, step):
        start = process_time_ns()
        if step - self.last_input_step != len(actions):
            raise ValueError(f"Input actions must cover every control step up to {step}")
        active = self.dense and self.core.pending is not None
        if active:
            last_legal = self.core.pending.step + 40
            self.pending_gaps += max(0, min(step - 1, last_legal) - self.last_input_step)
        self.last_input_step = step
        self.actions.extend(actions)
        if step % 5 and not active:
            return []
        self.calls += 1
        self.off_grid_calls += int(step % 5 != 0)
        self.original_lid = tracker.inspect_lids
        detector = self.inspect_original if self.lid is None else self.inspect
        with patch.object(tracker, "inspect_lids", detector):
            events = self.core.observe(frame, self.actions, step)
        self.actions.clear()
        self.processing_ns += process_time_ns() - start
        return events


def factory_base():
    return WindowedObserver("base")


def factory_dense():
    return WindowedObserver("dense")


def factory_plane():
    return WindowedObserver("plane")


def factory_combo():
    return WindowedObserver("combo")


def exact_frame(path, recorded):
    hashes = recorded["observation_sha256"]
    with np.load(path, allow_pickle=False) as data:
        if not hashes or set(hashes) != set(data) or any(array_hash(data[k]) != h for k, h in hashes.items()):
            raise ValueError(f"Observation hash mismatch: {path}")
    return Frame.read(path)


def input_rows(folder, dense):
    if dense is None:
        yield from trace_rows(folder)
        return
    start, end = dense["window"]
    recorded = {r["step"]: r for r in dense["records"]}
    if sorted(recorded) != list(range(start, end + 1)):
        raise ValueError("Dense archive has gaps")
    previous = 0
    for step, path, actions, row in trace_rows(folder):
        if start < step <= end:
            realized = []
            for tick in range(previous + 1, step + 1):
                di = recorded[tick]
                realized.extend(di["actions"])
                if tick == step and (di["state_sha256"] != row["state_sha256"] or
                                     di["observation_sha256"] != row["observation_sha256"]):
                    raise ValueError("Dense replay differs from original grid")
                yield tick, dense["source"] / "frames" / f"{tick}.npz", di["actions"], di
            if realized != actions:
                raise ValueError("Dense replay differs from original actions")
        else:
            yield step, path, actions, row
        previous = step


def score(folder, arms, dense=None, measure=False):
    observers = {arm: WindowedObserver(arm) for arm in arms}
    events = {name: [] for name in observers}
    timeline = []
    for step, path, actions, recorded in input_rows(folder, dense):
        frame = exact_frame(path, recorded)
        audit = recorded["audit"]
        for name, method in observers.items():
            output = method.observe(frame, actions, step)
            events[name].extend({"step": step, "correct": audit["goals"][SOUP] and not audit["held_soup"], **e}
                                for e in output if e["event"] == "in_basket")
        timeline.append({"step": step, "audit": audit, "transitions": recorded.get("transitions", [])})
    # Keep the official positive event on the original five-step scoreboard
    # for all lanes. Dense audit is used only to validate event-time commits.
    # Re-read its transitions: the dense path splits original blocks.
    grid = [{"step": t, "audit": r["audit"], "transitions": r.get("transitions", [])}
            for t, _, _, r in trace_rows(folder)]
    positive, censored = stable_positive(grid)
    methods = {}
    for name, observer in observers.items():
        commits = events[name]
        hits = [r["step"] for r in commits if r["correct"] and positive is not None
                and positive <= r["step"] <= positive + 30]
        false = [r for r in commits if not r["correct"]]
        methods[name] = {"hit": min(hits, default=None), "commits": commits, "false": false,
                         "observer_frames": observer.calls, "off_grid_frames": observer.off_grid_calls,
                         "fit_calls": observer.fit_calls, "missing_steps_in_pending": observer.pending_gaps}
        if measure:
            methods[name]["process_ms"] = round(observer.processing_ns / 1e6, 3)
    return {"positive_step": positive, "terminal_censored": censored, "frames_verified": len(timeline),
            "methods": methods}


def stats(rows, arms):
    return {arm: {"confirmed_in_30": sum(r["methods"][arm]["hit"] is not None for r in rows),
                  "false_events": sum(len(r["methods"][arm]["false"]) for r in rows),
                  **{key: round(sum(r["methods"][arm][key] for r in rows), 3)
                     for key in ("observer_frames", "off_grid_frames", "fit_calls", "process_ms",
                                 "missing_steps_in_pending")}}
            for arm in arms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("natural", type=Path)
    p.add_argument("original", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("dense", nargs="+", type=Path)
    a = p.parse_args()
    if a.out.exists() or sha(a.original) != FROZEN_REPORT_SHA:
        raise ValueError("Choose new output and the exact frozen report")
    old = read(a.original)
    frozen = read(HERE / "freeze.json")
    if frozen["source"] != old["source"] or any(sha(HERE / n) != s for n, s in frozen["source"].items()):
        raise ValueError("Stage 3b comparison source changed")
    dense_cases = {}
    for root in a.dense:
        manifest = read(root / "dense-summary.json")
        if not manifest["complete"] or manifest["report_sha256"] != FROZEN_REPORT_SHA:
            raise ValueError(f"Incomplete dense archive: {root}")
        for record in manifest["cases"]:
            seed = record["seed"]
            folder = root / str(seed)
            path = folder / "dense.json"
            if seed in dense_cases or sha(path) != record["dense_sha256"]:
                raise ValueError(f"Duplicate/mismatched dense archive: {seed}")
            dense_cases[seed] = {**read(path), "source": folder}
    coarse, dense = [], []
    for r in old["results"]:
        seed = r["seed"]
        folder = a.natural / str(seed)
        data = {"seed": seed, **score(folder, ("base", "plane"), measure=True)}
        if (data["positive_step"] != r["positive_step"] or
                data["methods"]["base"]["hit"] != r["scores"]["identity"]["hit"] or
                len(data["methods"]["base"]["false"]) != len(r["scores"]["identity"]["false"])):
            raise ValueError(f"Frozen baseline mismatched on {seed}")
        if seed in dense_cases:
            other = {"seed": seed, **score(folder, ARMS, dense_cases[seed], measure=True)}
            for arm in ("base", "plane"):
                x, y = other["methods"][arm], data["methods"][arm]
                if x["hit"] != y["hit"] or x["commits"] != y["commits"]:
                    raise ValueError(f"Coarse arms changed after aligning dense frames: {seed}")
            if any(other["methods"][arm]["missing_steps_in_pending"] for arm in ("dense", "combo")):
                raise ValueError(f"Uncovered pending window on {seed}")
            dense.append(other)
        coarse.append(data)
    report = {"stage": "development", "source": {
        name: sha(HERE / name) for name in ("stage3d_ablation.py", "plane_lid.py", "tracker.py")},
        "original_sha256": FROZEN_REPORT_SHA,
        "coarse_summary": stats(coarse, ("base", "plane")),
        "dense_summary": stats(dense, ARMS), "coarse": coarse, "dense": dense}
    write(a.out, report)
    print("coarse", report["coarse_summary"], flush=True)
    print("dense", report["dense_summary"], flush=True)


if __name__ == "__main__":
    main()
