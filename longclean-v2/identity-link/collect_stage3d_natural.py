"""Collect a separate, candidate-independent stage 3d policy holdout."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from collect import FRAME_STRIDE, NATURAL_BUDGET, capture, collect


HERE = Path(__file__).resolve().parent
TRACKS = ("soup",) * 6 + ("sauce_then_soup", "full_task")


def assignments(card):
    return [
        {"layout": 930100 + i, "card": card, "policy_seed": 100 + i % 8, "track": TRACKS[i // 2]}
        for i in range(card - 2, 16, 2)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("card", type=int, choices=(2, 3))
    p.add_argument("study", type=Path, help="Locked study4 root")
    p.add_argument("out", type=Path, help="New output root shared by the two cards")
    a = p.parse_args()
    study, out = a.study.resolve(), a.out.resolve()
    sys.path.insert(0, str(study.parent))
    from common import COMMIT, FULL_TASK, GPU, REPO, SAUCE, SOUP, read, sha, write

    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU[a.card] or os.environ.get(
        "MUJOCO_EGL_DEVICE_ID"
    ) != str(a.card) or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("CUDA UUID and EGL pin must both match the selected card")

    freeze_path, plan_path = study / "frozen.json", study / "plan.json"
    freeze, plan = read(freeze_path), read(plan_path)
    if sha(plan_path) != freeze["plan_sha256"] or any(
        sha(study.parent / name) != expected for name, expected in freeze["code"].items()
    ):
        raise RuntimeError("Frozen runner does not match study4")
    repo_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if repo_head != COMMIT:
        raise RuntimeError(f"Unexpected LongClean commit: {repo_head}")
    bddl = study / plan["bddl"]
    if sha(bddl) != plan["bddl_sha256"]:
        raise RuntimeError("BDDL does not match the study4 plan")
    original_sha = sha(HERE / "collect.py")
    if original_sha != read(HERE / "freeze.json")["source"]["collect.py"] or (FRAME_STRIDE, NATURAL_BUDGET) != (5, 360):
        raise RuntimeError("Original natural collection code differs")

    from run import Pi05, Worker, fingerprint, instruction

    class SampledWorker(Worker):
        def step(self, actions, target, task):
            applied = super().step(actions, target, task)
            if self.current["step"] % FRAME_STRIDE == 0 or self.terminal():
                capture(self, self.path.parent / "frames")
            return applied

    jobs = assignments(a.card)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"manifest-{a.card}.json"
    if manifest_path.exists() or any((out / str(job["layout"])).exists() for job in jobs):
        raise FileExistsError(f"Stage 3d output already used for card {a.card}: {out}")
    manifest = {
        "schema": "identity-stage3d-natural-v1", "cohort": "natural", "phase": "blind",
        "card": a.card, "collector_sha256": sha(Path(__file__)), "original_collector_sha256": original_sha,
        "commit": repo_head, "study_freeze_sha256": sha(freeze_path), "study_plan_sha256": sha(plan_path),
        "bddl_sha256": plan["bddl_sha256"], "frame_stride": FRAME_STRIDE, "budget_steps": NATURAL_BUDGET,
        "planned": jobs, "records": [], "complete": False,
    }
    write(manifest_path, manifest)
    policy = Pi05(GPU[a.card])
    prompts = {"soup": (SOUP, instruction(SOUP)), "sauce": (SAUCE, instruction(SAUCE)),
               "full_task": (None, FULL_TASK)}
    for job in jobs:
        record = collect(job, out, bddl, policy, SampledWorker, fingerprint, None, prompts, write)
        manifest["records"].append(f'{job["layout"]}/fixture.json')
        write(manifest_path, manifest)
        print("recorded", job["layout"], job["track"], record["end"]["step"], flush=True)
    manifest["complete"] = True
    write(manifest_path, manifest)


if __name__ == "__main__":
    main()
