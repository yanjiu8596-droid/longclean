"""Recover single-step RGB-D around frozen release windows from exact realized actions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import COMMIT, GPU, REPO, SAUCE, SOUP, array_hash, read, sha, write

REPORT_SHA = "df6c03483645c19a61a303e1667d09d114908656cd5a8044f13dea833d3f5d15"
DEFAULT_SEEDS = (929103, 929107, 929114, 929115, 929118)
FINGERPRINT = ("step", "bddl_sha256", "raw_sha256", "state_sha256", "observation_sha256")


def exact(current, recorded, seed):
    differences = [key for key in (*FINGERPRINT, "audit") if current[key] != recorded[key]]
    if differences:
        raise RuntimeError(f"{seed}:{recorded['step']} anchor mismatch: {differences}")


def load_registration():
    path = ROOT / "lid-geometry" / "challenge" / "render.py"
    spec = importlib.util.spec_from_file_location("identity_dense_registration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.registered_masks, sha(path)


def replay_case(Simulator, bddl, source, target, result, register_masks):
    import numpy as np

    seed = result["seed"]
    trace = result["scores"]["identity"]["trace"]
    pending = [event for event in trace if event["event"] == "identity_pending"]
    if len(pending) != 1:
        raise ValueError(f"{seed}: select a case with exactly one original pending release")
    stamp = pending[0]["step"]
    first, last = stamp - 5, stamp + 40
    source = source / str(seed)
    target = target / str(seed)
    target.mkdir()
    frames = target / "frames"
    frames.mkdir()
    fixture = read(source / "fixture.json")
    if fixture["layout"] != seed or not 0 < first <= last <= fixture["end"]["step"]:
        raise ValueError(f"Invalid fixture/release window for {seed}")
    anchors, dense = [], []
    scene = Simulator(bddl, target, seed, False)

    def store(state, actions):
        step = state["step"]
        if register_masks:
            masks = register_masks(scene)
            if array_hash(scene.env.get_sim_state()) != state["state_sha256"]:
                raise RuntimeError(f"{seed}:{step} registration changed simulation state")
            np.save(frames / f"{step}-mask.npy", masks[SOUP], allow_pickle=False)
            np.save(frames / f"{step}-sauce-mask.npy", masks[SAUCE], allow_pickle=False)
        (target / "observation.npz").replace(frames / f"{step}.npz")
        dense.append({**state, "actions": actions, "archived_anchor": step in fixture["frame_steps"]})

    try:
        exact(scene.capture(), {**fixture["initial"], "audit": fixture["initial_audit"]}, seed)
        anchors.append(0)
        with (source / "trace" / "actions.jsonl").open(encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                start, step = row["start_step"], row["step"]
                actions = row["actions"]
                if start != scene.frame or not actions or start + len(actions) != step:
                    raise RuntimeError(f"{seed}: noncontiguous realized actions at {step}")
                blocks = [[a] for a in actions] if start < last and step > first else [actions]
                transitions, regrasp = [], 0
                for chunk in blocks:
                    state = scene.advance(chunk)
                    if state["executed"] != len(chunk):
                        raise RuntimeError(f"{seed}:{state['step']} replay consumed different actions")
                    transitions.extend(state["transitions"])
                    regrasp += state["completed_regrasp_steps"]
                    if first <= state["step"] <= last:
                        store(state, chunk)
                exact(state, row, seed)
                if (transitions != row["transitions"] or regrasp != row["completed_regrasp_steps"]
                        or state["done"] != row["done"]):
                    raise RuntimeError(f"{seed}:{step} block audit differs")
                anchors.append(step)
        exact(state, {**fixture["end"], "audit": fixture["end_audit"]}, seed)
        if anchors != fixture["frame_steps"] or [r["step"] for r in dense] != list(range(first, last + 1)):
            raise RuntimeError(f"{seed}: incomplete replay")
    finally:
        scene.close()
    record = {"seed": seed, "positive_step": result["positive_step"],
              "frozen_hit": result["scores"]["identity"]["hit"], "release": pending[0],
              "window": [first, last], "verified_anchor_steps": anchors, "records": dense}
    write(target / "dense.json", record)
    print(f"{seed}: {len(anchors)} exact anchors, {len(dense)} dense observations", flush=True)
    return {"seed": seed, "original_card": fixture["card"], "anchors": len(anchors),
            "frames": len(dense), "window": [first, last], "release": stamp,
            "dense_sha256": sha(target / "dense.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("card", type=int, choices=tuple(GPU))
    p.add_argument("study", type=Path, help="Frozen study4 directory")
    p.add_argument("collection", type=Path, help="Original natural-output archive")
    p.add_argument("report", type=Path, help="Original independent.json archive")
    p.add_argument("out", type=Path, help="New output root")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--no-masks", action="store_true", help="Skip evaluator-only wrist truth masks")
    args = p.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != GPU[args.card]
            or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.card)
            or os.environ.get("MUJOCO_GL") != "egl"):
        raise RuntimeError("Set the authorized card UUID and matching EGL device before replay")
    if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("LongClean source commit differs")
    study = args.study.resolve()
    frozen = read(study / "frozen.json")
    if (sha(study / "plan.json") != frozen["plan_sha256"]
            or any(sha(ROOT / name) != digest for name, digest in frozen["code"].items())):
        raise RuntimeError("Frozen study4 runner differs from the current source")
    plan = read(study / "plan.json")
    bddl = study / plan["bddl"]
    if sha(bddl) != plan["bddl_sha256"] or sha(args.report) != REPORT_SHA:
        raise RuntimeError("Frozen BDDL or replay report differs")
    report = read(args.report)
    results = {r["seed"]: r for r in report["results"]}
    if len(set(args.seeds)) != len(args.seeds) or any(
            seed not in results or results[seed]["positive_step"] is None for seed in args.seeds):
        raise ValueError("Expected distinct positive seeds from the original archive")

    from simulator import Simulator, ensure_libero_config

    register, renderer_sha = (None, None) if args.no_masks else load_registration()
    ensure_libero_config()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"stage": "diagnostic_replay_on_prior_cohort", "card": args.card,
               "bddl_sha256": plan["bddl_sha256"], "report_sha256": REPORT_SHA,
               "replay_code_sha256": sha(Path(__file__)), "registration_code_sha256": renderer_sha,
               "planned": list(args.seeds), "cases": [], "complete": False}
    path = args.out / "dense-summary.json"
    write(path, summary)
    for seed in args.seeds:
        summary["cases"].append(replay_case(Simulator, bddl, args.collection, args.out, results[seed], register))
        write(path, summary)
    summary["complete"] = True
    write(path, summary)


if __name__ == "__main__":
    main()
