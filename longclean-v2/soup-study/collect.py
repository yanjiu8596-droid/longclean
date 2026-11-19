"""Generate controlled paired soup releases independently of either observer."""
from pathlib import Path
from shutil import copyfile
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from probe import ROOT
from common import SOUP, V1, read, sha, write
from run import Worker

DEV = (926090, 926091)
BLIND = tuple(range(926100, 926108))


def branch(out, bddl, seed, condition, asset):
    samples = []
    with Worker(bddl, out, seed) as worker:
        initial = worker.current["state_sha256"]
        a = worker.current["audit"]
        start = np.array(a["site_xyz"])
        basket = np.array(a["positions"]["basket_1"])
        soup = np.array(a["positions"][SOUP])
        grab = soup + asset["grasp_xyz_offset_m"]
        high = grab + [0, 0, .16]
        safe = start.copy()
        safe[2] = max(.74, start[2])
        side = -1 if seed % 2 else 1
        over = np.array([basket[0] + side * .022, basket[1] - .025, .67])
        if condition == "outside":
            over[1] = basket[1] - .096
        lower = over.copy()
        lower[2] = .555 if condition == "inside" else .60
        rotation = np.array(asset["base_rotation_matrix"])
        stages = [(safe, 16, -1), (high, 33, -1), (grab, 44, -1), (grab, 24, 1),
                  (high, 29, 1), (over, 44, 1), (lower, 28, 1), (lower, 20, -1), (over, 36, -1)]
        frames = out / "frames"
        frames.mkdir()
        copyfile(worker.npz, frames / "0.npz")
        held, held_at_release, release = False, False, None
        for stage, (point, count, grip) in enumerate(stages):
            if stage == 4:
                held = worker.current["audit"]["held_soup"]
                if not held:
                    break
            if stage == 7:
                held_at_release = worker.current["audit"]["held_soup"]
                release = worker.current["step"] + 1
            for _ in range(count):
                now = worker.current["audit"]
                dp = 1.4 * (point - np.asarray(now["site_xyz"])) / .05
                dr = Rotation.from_matrix(rotation @ np.asarray(now["site_rotation"]).T).as_rotvec() / .5
                action = np.r_[np.clip(np.r_[dp, dr], -.85, .85), grip].tolist()
                worker.step([action], None, "scripted placement challenge")
                info = worker.current
                step = info["step"]
                # Store all labels, with observer frames at a deterministic 2-step cadence.
                samples.append({"step": step, "grip": grip, "in": info["audit"]["goals"][SOUP],
                                "held": info["audit"]["held_soup"],
                                "xyz": info["audit"]["positions"][SOUP]})
                if step % 2 == 0 and (stage >= 3 or step % 10 == 0):
                    copyfile(worker.npz, frames / f"{step}.npz")
        end = worker.current
    return {"layout": seed, "condition": condition, "initial": initial, "grasped": held,
            "held_at_release": held_at_release, "release": release, "samples": samples,
            "final_predicate": end["audit"]["goals"][SOUP], "num_steps": end["step"]}


def main():
    card, out = int(sys.argv[1]), Path(sys.argv[2]).resolve()
    if card not in (2, 3):
        raise ValueError("Expected GPU 2 or 3")
    study = ROOT / "study4"
    plan = read(study / "plan.json")
    bddl = study / plan["bddl"]
    assert sha(bddl) == plan["bddl_sha256"]
    asset = read(V1 / "grasp-reference.json")
    # The plan assigns each layout to a single card without looking at truth or observations.
    for i, seed in enumerate((*DEV, *BLIND)):
        if i % 2 != card - 2:
            continue
        root = out / ("dev" if seed in DEV else "independent") / str(seed)
        root.mkdir(parents=True)
        a = branch(root / "inside", bddl, seed, "inside", asset)
        b = branch(root / "outside", bddl, seed, "outside", asset)
        assert a["initial"] == b["initial"]
        write(root / "pair.json", [a, b])
        print("recorded", seed, flush=True)


if __name__ == "__main__":
    main()
