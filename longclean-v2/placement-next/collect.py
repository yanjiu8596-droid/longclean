"""Collect fresh observation traces with the frozen policy, independently of the candidate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("card", type=int, choices=(2, 3))
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    study, out = a.study.resolve(), a.out.resolve()
    sys.path.insert(0, str(study.parent))
    from common import CHEESE, COMMIT, GPU, OFFSETS, REPO, read, sha, write
    from run import Pi05, Worker, evaluate, fingerprint, make_prefix, replay_prefix
    from vision import Progress, load_locator

    freeze = read(study / "frozen.json")
    assert all(sha(study.parent / name) == digest for name, digest in freeze["code"].items())
    assert COMMIT == subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU[a.card] and int(os.environ["MUJOCO_EGL_DEVICE_ID"]) == a.card
    bddl = study / read(study / "plan.json")["bddl"]
    assert sha(bddl) == read(study / "plan.json")["bddl_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    policy, locator = Pi05(GPU[a.card]), load_locator()
    for i in range(a.card - 2, 16, 2):
        seed = 925100 + i
        run = out / str(seed)
        run.mkdir()
        prefix = make_prefix(policy, bddl, run, seed)
        prefix_ok = not prefix.get("failure") and prefix["observed_cheese"]
        moved = (i // 2) % 2 == 0
        method = "fixed" if (i // 2) % 4 < 2 else "replan"
        design = {"seed": seed, "moved": moved, "method": method, "policy_seed": 40 + i % 4,
                  "prefix_ok": prefix_ok, "prefix_failure": prefix.get("failure"),
                  "initial": prefix["initial"], "soup_before": None}
        if prefix_ok:
            design["soup_before"] = locator(run / "checkpoint.npz")
        if prefix_ok and moved:
            with Worker(bddl, run / "boundary", seed) as worker:
                replay_prefix(worker, prefix, Progress())
                assert worker.goal(CHEESE)
                for offset in OFFSETS:
                    boundary = worker.call({"op": "move", "offset": offset})
                    if "rejected" not in boundary:
                        design.update(offset=offset, perturbed_checkpoint=fingerprint(boundary))
                        break
        design["collected"] = prefix_ok and (not moved or "offset" in design)
        write(run / "fixture.json", design)
        if design["collected"]:
            evaluate(policy, locator, bddl, run, run / "trace", seed, design["policy_seed"], method, moved)
        print("recorded", seed, design["collected"], flush=True)


if __name__ == "__main__":
    main()
