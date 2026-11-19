"""Replay the frozen original evaluator as an independent smoke control."""
import argparse
import json
from pathlib import Path

from stage4_fixture import HERE, GPU, read, require, sha, validate_gpu, validate_study, write


def cases(output):
    plan, frozen = read(output / "plan.json"), read(output / "frozen.json")
    require(plan["phase"] == "smoke" and len(plan["cases"]) == 2 and plan["cases"] == frozen["cases"],
            "Parity replay must use both sealed smoke layouts, never the pilot")
    require(sha(output / "plan.json") == frozen["plan_sha256"], "Smoke plan hash differs")
    for name, digest in frozen["code"].items():
        require(sha(HERE / name) == digest, f"Unexpected sealed source: {name}")
    return plan["cases"]


def legacy_replay(args):
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study)
    from run import Pi05, evaluate, load_locator

    policy = Pi05(GPU[args.card])
    locate = load_locator()
    for case in cases(args.output):
        if case["card"] != args.card:
            continue
        fixture = args.output / "fixtures" / str(case["layout_seed"])
        require(sha(fixture / "prefix.json") == case["prefix_sha256"] and
                sha(fixture / "fixture.json") == case["fixture_sha256"], "Changed smoke fixture")
        location = args.output / "legacy-parity" / case["tag"]
        result = evaluate(policy, locate, bddl, fixture, location, case["layout_seed"], case["policy_seed"],
                          "replan", case["moved"])
        print("LEGACY", case["tag"], result["success"], result["steps"], flush=True)


def trace(path):
    end = 0
    with path.open(encoding="utf-8") as f:
        for row in map(json.loads, f):
            require(row["start_step"] == end and row["step"] - end == len(row["actions"]),
                    f"Noncontiguous realized actions: {path}:{row['step']}")
            end = row["step"]
            yield row


def commitments(events):
    keys = ("step", "object", "event", "matches_audit")
    return [tuple(event.get(k) for k in keys) for event in events
            if event.get("writer", "legacy") != "stage3d_plane" and event.get("applied", True)]


def compare(args):
    rows = []
    for case in cases(args.output):
        tag = case["tag"]
        a_dir = args.output / "legacy-parity" / tag
        b_dir = args.output / "episodes" / tag / "baseline"
        a, b = read(a_dir / "result.json"), read(b_dir / "result.json")
        fields = ("success", "terminal_goals", "steps", "after_fork_steps", "terminal_reason",
                  "prefix_sha256", "base_checkpoint", "trial_checkpoint", "initial_motion_observation",
                  "schedule_changes", "target_runs", "observed_terminal", "completed_reassigned_chunks",
                  "completed_regrasp_steps")
        for key in fields:
            require(a[key] == b[key], f"{tag} changed original outcome field: {key}")
        require(commitments(a["state_events"]) == commitments(b["state_events"]),
                f"{tag} changed original state-writing events")
        split = iter(trace(b_dir / "actions.jsonl"))
        anchors = []
        split_blocks = 0
        for original in trace(a_dir / "actions.jsonl"):
            actions, transitions, regrasp = [], [], 0
            while True:
                block = next(split)
                split_blocks += 1
                require(block["target"] == original["target"] and block["instruction"] == original["instruction"]
                        and block["step"] <= original["step"], f"{tag} differing inference block at {block['step']}")
                actions.extend(block["actions"])
                transitions.extend(block["transitions"])
                regrasp += block["completed_regrasp_steps"]
                if block["step"] == original["step"]:
                    break
            checkpoint = ("raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256", "audit", "done")
            require(actions == original["actions"] and transitions == original["transitions"] and
                    regrasp == original["completed_regrasp_steps"] and
                    all(block[k] == original[k] for k in checkpoint), f"{tag} differs at anchor {original['step']}")
            anchors.append(original["step"])
        require(next(split, None) is None and anchors and anchors[-1] == a["steps"],
                f"{tag} traces terminate at different steps")
        rows.append({"tag": tag, "layout_seed": case["layout_seed"], "anchors": len(anchors),
                     "split_blocks": split_blocks, "step": anchors[-1], "same": True,
                     "legacy_trace_sha256": sha(a_dir / "actions.jsonl"),
                     "split_trace_sha256": sha(b_dir / "actions.jsonl")})
    report = {"schema": "v2-stage4-smoke-parity/1", "gate": len(rows) == 2 and all(r["same"] for r in rows),
              "smoke_freeze_sha256": sha(args.output / "frozen.json"), "cases": rows}
    write(args.output / "legacy-parity.json", report)
    print("PARITY", sum(r["anchors"] for r in rows), "original decision anchors", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("replay", "compare"))
    p.add_argument("--card", type=int, choices=GPU)
    p.add_argument("--study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.stage == "replay":
        require(args.card is not None and args.study is not None, "Replay requires card and study")
        args.study = args.study.resolve()
        legacy_replay(args)
    else:
        require(not (args.output / "legacy-parity.json").exists(), "Parity summary already exists")
        compare(args)


if __name__ == "__main__":
    main()
