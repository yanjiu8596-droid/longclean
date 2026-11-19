"""Collect candidate-independent RGB-D, realized actions and separate simulation audit."""
import argparse
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys


NATURAL_BLIND = tuple(range(929100, 929124))
NATURAL_SMOKE = (929090, 929091)
SCRIPTED_BLIND = tuple(range(929200, 929208))
SCRIPTED_SMOKE = (929190, 929191)
NATURAL_BUDGET = 360
SAUCE_PREFIX_STEPS = 150
FRAME_STRIDE = 5


def assignments(cohort, phase):
    seeds = {
        ("natural", "blind"): NATURAL_BLIND, ("natural", "smoke"): NATURAL_SMOKE,
        ("scripted", "blind"): SCRIPTED_BLIND, ("scripted", "smoke"): SCRIPTED_SMOKE,
    }[cohort, phase]
    # Both cards get the same mix; the next run is never selected from past outcomes.
    return [
        {"layout": seed, "card": 2 + i % 2, "policy_seed": 80 + i % 8 if cohort == "natural" else None,
         "track": ("soup", "soup", "soup", "soup", "sauce_then_soup", "full_task")[(i // 2) % 6]
         if cohort == "natural" else "scripted_soup"}
        for i, seed in enumerate(seeds)
    ]


def capture(worker, frame_dir):
    step = worker.current["step"]
    path = frame_dir / f"{step}.npz"
    if not path.exists():
        copyfile(worker.npz, path)


def collect(job, out, bddl, policy, worker_type, fingerprint, scripted_soup, prompts, write):
    path = out / str(job["layout"])
    frames = path / "frames"
    frames.mkdir(parents=True, exist_ok=False)
    if policy is not None:
        policy.reset(job["policy_seed"])
    with worker_type(bddl, path / "trace", job["layout"]) as worker:
        initial = worker.current
        capture(worker, frames)
        if policy is None:
            # The oracle-aided controller is only a physical positive control.
            # These observations cannot count toward the frozen natural holdout.
            scripted_soup(worker)
        else:
            track = job["track"]
            while worker.current["step"] < NATURAL_BUDGET and not worker.terminal():
                step = worker.current["step"]
                prompt_name = ("sauce" if track == "sauce_then_soup" and step < SAUCE_PREFIX_STEPS
                               else "full_task" if track == "full_task" else "soup")
                target, prompt = prompts[prompt_name]
                actions = policy.act(worker, prompt, limit=min(2 * FRAME_STRIDE, NATURAL_BUDGET - step))
                if len(actions) == 0:
                    raise RuntimeError(f"Policy returned no actions at step {step}")
                # Commit every action from the original 10-step block without further
                # inference between halves; only the simulator can cut a block short.
                for start in range(0, len(actions), FRAME_STRIDE):
                    worker.step(actions[start:start + FRAME_STRIDE], target, prompt)
                    if worker.terminal():
                        break
        capture(worker, frames)
        end = worker.current
    record = {**job, "initial": fingerprint(initial), "end": fingerprint(end),
              "initial_audit": initial["audit"], "end_audit": end["audit"],
              "official_full_success": all(end["audit"]["goals"].values()),
              "environment_done": bool(end.get("done", False)),
              "budget_steps": NATURAL_BUDGET if policy is not None else None,
              "frame_steps": sorted(int(frame.stem) for frame in frames.glob("*.npz")),
              "frames": "frames", "actions": "trace/actions.jsonl"}
    write(path / "fixture.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=int, choices=(2, 3))
    parser.add_argument("study", type=Path, help="Frozen study4 directory")
    parser.add_argument("out", type=Path, help="Fresh output root per cohort and phase")
    parser.add_argument("--cohort", choices=("natural", "scripted"), default="natural")
    parser.add_argument("--phase", choices=("smoke", "blind"), required=True)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.out.resolve()
    sys.path.insert(0, str(study.parent))
    from common import COMMIT, FULL_TASK, GPU, REPO, SAUCE, SOUP, V1, read, sha, write

    freeze_path, plan_path = study / "frozen.json", study / "plan.json"
    freeze, plan = read(freeze_path), read(plan_path)
    if sha(plan_path) != freeze["plan_sha256"] or not all(
            sha(study.parent / name) == digest for name, digest in freeze["code"].items()):
        raise RuntimeError("Frozen runner does not match study4")
    repo_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if COMMIT != repo_head:
        raise RuntimeError(f"Unexpected LongClean commit: {repo_head}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU[args.card] or str(args.card) != os.environ.get(
            "MUJOCO_EGL_DEVICE_ID"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID must match the selected card")
    bddl = study / plan["bddl"]
    if sha(bddl) != plan["bddl_sha256"]:
        raise RuntimeError("BDDL does not match the frozen plan")

    from run import Pi05, Worker, fingerprint, instruction, scripted_soup

    class SampledWorker(Worker):
        def step(self, actions, target, task):
            applied = super().step(actions, target, task)
            if self.current["step"] % FRAME_STRIDE == 0 or self.terminal():
                capture(self, self.path.parent / "frames")
            return applied

    jobs = [job for job in assignments(args.cohort, args.phase) if job["card"] == args.card]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"manifest-{args.card}.json"
    # Plan and intent land before the first model invocation. Partial output remains
    # inspectable, but a pre-existing root for this card cannot be overwritten.
    if manifest_path.exists() or any((out / str(job["layout"])).exists() for job in jobs):
        raise FileExistsError(f"Output already contains a collection for card {args.card}: {out}")
    manifest = {"schema": "identity-link-collection-v1", "cohort": args.cohort, "phase": args.phase,
                "card": args.card, "collector_sha256": sha(Path(__file__)), "commit": repo_head,
                "study_freeze_sha256": sha(freeze_path), "study_plan_sha256": sha(plan_path),
                "bddl_sha256": plan["bddl_sha256"], "frame_stride": FRAME_STRIDE,
                "grasp_reference_sha256": sha(V1 / "grasp-reference.json") if args.cohort == "scripted" else None,
                "planned": jobs, "records": [], "complete": False}
    write(manifest_path, manifest)
    policy = Pi05(GPU[args.card]) if args.cohort == "natural" else None
    prompts = {"soup": (SOUP, instruction(SOUP)), "sauce": (SAUCE, instruction(SAUCE)),
               "full_task": (None, FULL_TASK)}
    for job in jobs:
        record = collect(job, out, bddl, policy, SampledWorker, fingerprint, scripted_soup, prompts, write)
        manifest["records"].append(str(Path(str(job["layout"])) / "fixture.json"))
        write(manifest_path, manifest)
        print("recorded", job["layout"], job["track"], record["end"]["step"], flush=True)
    manifest["complete"] = True
    write(manifest_path, manifest)


if __name__ == "__main__":
    main()
