"""Audit RGB-D at individual control steps; compare source hashes at original chunk boundaries."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "placement-next"))

import numpy as np
from common import SOUP, read, sha, write
from observer import PlacementObserver, grasp_vote, near_basket
from run import Worker, fingerprint, replay_prefix
from vision import Frame, Progress, cloud, color_mask


def inks(image):
    r, g, b = np.moveaxis(np.asarray(image, dtype=np.int16), -1, 0)
    yellow = (r > 70) & (g > 58) & (g > b + 35) & (r > b + 44)
    blue = (b > r + 18) & (b > g + 6) & (b > 52)
    return blue, yellow


def digest(frame):
    eef = frame.geometry["eef"]
    wrist = cloud(frame, 1)
    delta = wrist - eef
    grip_roi = (np.linalg.norm(delta[..., :2], axis=-1) < .055) & (delta[..., 2] > -.10) & (delta[..., 2] < .02)
    x, y, z = np.moveaxis(wrist, -1, 0)
    pocket = (-.055 < x) & (x < .065) & (.195 < y) & (y < .335) & (.468 < z) & (z < .67)
    blue, yellow = inks(frame.images[1])
    pocket_stats = []
    for ink in (blue, yellow):
        points = wrist[ink & pocket]
        pocket_stats.append([len(points), np.median(points, axis=0).tolist() if len(points) else None])
    return {"gap": frame.gap, "eef": eef.tolist(), "grasp_vote": grasp_vote(frame),
            "over": near_basket(eef), "grip_ink": [int((grip_roi & ink).sum()) for ink in (blue, yellow)],
            "pocket_ink": pocket_stats}


def main():
    study = ROOT / "study4"
    fresh = ROOT / "placement-next/fresh"
    out = Path(sys.argv[1]).resolve()
    source = read(study / "frozen.json")["code"]
    assert all(sha(ROOT / name) == expected for name, expected in source.items())
    candidate = read(ROOT / "placement-next/candidate-freeze.json")["source"]
    assert sha(ROOT / "placement-next/observer.py") == candidate["observer.py"]
    bddl = study / read(study / "plan.json")["bddl"]
    assert sha(bddl) == read(study / "plan.json")["bddl_sha256"]
    traces = []
    for seed, start, end in ((925109, 330, 460), (925111, 290, 420)):
        fixture = fresh / str(seed)
        trace = fixture / "trace"
        h = read(trace / "result.json")
        detector = PlacementObserver()
        samples, anchors, raw_frames = [], [], []
        with Worker(bddl, out / str(seed), seed) as worker:
            replay_prefix(worker, read(fixture / "prefix.json"), Progress())
            if h["moved"]:
                worker.call({"op": "move", "offset": read(fixture / "fixture.json")["offset"]})
            assert fingerprint(worker.current) == h["trial_checkpoint"]
            fork = worker.current["step"]
            detector.clear_view = Frame.read(worker.npz)
            with (trace / "actions.jsonl").open() as stream:
                for row in map(json.loads, stream):
                    if row["start_step"] < fork:
                        continue
                    split = start <= row["step"] <= end
                    actions = row["actions"]
                    for chunk in ([a] for a in actions) if split else [actions]:
                        worker.step(chunk, row["target"], row["instruction"])
                        sample = Frame.read(worker.npz)
                        if split:
                            raw_frames.append(Path(worker.npz).read_bytes())
                            a = worker.current["audit"]
                            samples.append({"step": worker.current["step"], "action_grip": chunk[-1][-1],
                                            **digest(sample), "audit_held": a["held_soup"],
                                            "audit_xyz": a["positions"][SOUP], "audit_in": a["goals"][SOUP]})
                    assert fingerprint(worker.current) == fingerprint(row) and worker.current["audit"] == row["audit"], (seed, row["step"])
                    history = [(s, t, p.tolist()) for s, t, p in detector.votes]
                    events = detector.observe(sample, actions, row["step"])
                    if split:
                        anchors.append({"step": row["step"], "buffer": history, "events": events})
                    if row["step"] >= end:
                        break
        assert len(samples) == len(raw_frames)
        frames = out / str(seed) / "frames"
        frames.mkdir()
        for sample, data in zip(samples, raw_frames, strict=True):
            (frames / f'{sample["step"]}.npz').write_bytes(data)
        traces.append({"layout": seed, "samples": samples, "anchors": anchors})
        print("verified", seed, len(samples), flush=True)
    write(out / "probe.json", traces)


if __name__ == "__main__":
    main()
