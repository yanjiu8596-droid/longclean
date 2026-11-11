"""Validate and summarize completed fixed-state backbone study episodes."""

from collections import Counter, defaultdict
from math import isfinite, sqrt
from pathlib import Path
from random import Random
from statistics import median
import json
import re

from run_native_osc import validate_actions


PAIR_FIELDS = (
    "init_state_sha256", "bddl_sha256", "instruction", "observation_sha256",
    "sim_state_sha256", "controller", "control_hz", "settle_frames",
)


def audit_trace(output: Path, episode: dict) -> dict:
    first_goals = {}
    peak = steps = calls = 0
    trace = None
    with (output / "actions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if trace is not None and (trace["success"] or trace["done"]):
                raise ValueError(f"Actions continue past the official terminal step: {output}")
            trace = json.loads(line)
            actions = trace["actions"]
            validate_actions(actions)
            if (trace["start_step"] != steps or trace["executed"] != len(actions)
                    or len(actions) > episode["chunk_horizon"] or trace["instruction"] != episode["instruction"]):
                raise ValueError(f"Action trace differs from episode protocol at step {steps}: {output}")
            steps += len(actions)
            calls += 1
            goals = [json.dumps(goal["predicate"]) for goal in trace["audit_goals"] if goal["satisfied"]]
            peak = max(peak, len(goals))
            for goal in goals:
                first_goals.setdefault(goal, steps)
    if trace is None or (steps, calls, trace["success"]) != (episode["steps"], episode["vla_calls"], episode["success"]):
        raise ValueError(f"Action trace and terminal evidence disagree: {output}")
    reason = "simulator_success" if trace["success"] else "simulator_done" if trace["done"] else "step_budget"
    if reason != episode["terminal_reason"] or (reason == "step_budget" and steps != episode["step_budget"]):
        raise ValueError(f"Unexpected episode termination: {output}")
    for key in ("episode_elapsed_s", "policy_inference_s", "cuda_peak_reserved_mib"):
        value = episode.get(key)
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key} in {output}")
    if episode["cuda_peak_reserved_mib"] == 0:
        raise ValueError(f"Missing CUDA memory evidence: {output}")
    return {"peak_targets_at_chunk_boundaries": peak, "first_observed_goal_steps": first_goals}


def verify_pair(rows: dict) -> None:
    baseline = next(iter(rows.values()))
    for name in PAIR_FIELDS:
        if name not in baseline or any(name not in row or row[name] != baseline[name] for row in rows.values()):
            raise ValueError(f"Unpaired {name}: {baseline['task_id']} index {baseline['init_state_index']}")
    if (not baseline["instruction"].strip() or baseline["control_hz"] != 20 or baseline["settle_frames"] != 10
            or not {"camera1", "camera2"} <= baseline["observation_sha256"].keys()):
        raise ValueError("Expected the fixed LIBERO instruction and native dual-camera protocol")
    digests = [baseline[key] for key in ("init_state_sha256", "bddl_sha256", "sim_state_sha256")]
    digests.extend(baseline["observation_sha256"].values())
    if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in digests):
        raise ValueError("Expected SHA-256 digests for the paired LIBERO assets and observations")


def verify_split_assets(groups) -> None:
    bddl_by_scene = {}
    starts = set()
    for group in groups:
        scene = group["case"]["scene"]
        row = next(iter(group["episodes"].values()))
        start = scene, row["init_state_sha256"]
        if start in starts or bddl_by_scene.setdefault(scene, row["bddl_sha256"]) != row["bddl_sha256"]:
            raise ValueError(f"Initial-state overlap or BDDL drift in scene {scene:02d}")
        starts.add(start)


def select_horizon(groups: list[dict]) -> dict:
    scores = {}
    for chunk in (5, 25):
        name = f"pi05-{chunk}"
        by_scene = {scene: 0 for scene in (1, 2, 8)}
        calls = 0
        for group in groups:
            episodes = group["episodes"]
            by_scene[group["case"]["scene"]] += int(episodes[name]["success"]) - int(episodes["pi05-10"]["success"])
            calls += episodes[name]["vla_calls"]
        net_wins = sum(by_scene.values())
        scores[str(chunk)] = {"net_wins": net_wins, "net_wins_by_scene": {str(k): v for k, v in by_scene.items()},
                              "total_vla_calls": calls, "eligible": net_wins >= 2 and min(by_scene.values()) >= 0}
    eligible = [chunk for chunk, score in scores.items() if score["eligible"]]
    selected = min(eligible, key=lambda chunk: (-scores[chunk]["net_wins"], scores[chunk]["total_vla_calls"], int(chunk))) if eligible else "10"
    return {"selected_chunk": int(selected), "candidates": scores}


def paired_interval(differences: dict[int, list[int]]) -> list[float]:
    rng = Random(0)
    count = sum(map(len, differences.values()))
    resamples = sorted(
        sum(sum(rng.choices(values, k=len(values))) for values in differences.values()) / count
        for _ in range(10_000)
    )
    return [round(resamples[249], 6), round(resamples[9749], 6)]


def summarize(groups: list[dict], baseline: str) -> dict:
    def metrics(cases):
        count = len(cases)
        variants = {}
        comparisons = {}
        for name in cases[0]["episodes"]:
            episodes = [case["episodes"][name] for case in cases]
            successes = sum(episode["success"] for episode in episodes)
            rate = successes / count
            z2 = 1.96**2
            center = (rate + z2 / (2 * count)) / (1 + z2 / count)
            half = 1.96 * sqrt((rate * (1 - rate) + z2 / (4 * count)) / count) / (1 + z2 / count)
            variants[name] = {
                "successes": successes, "episodes": count, "success_rate": round(rate, 6),
                "wilson_95_ci": [round(max(0, center - half), 6), round(min(1, center + half), 6)],
                "median_episode_s": median(row["episode_elapsed_s"] for row in episodes),
                "median_policy_inference_s": median(row["policy_inference_s"] for row in episodes),
                "total_vla_calls": sum(row["vla_calls"] for row in episodes),
                "peak_cuda_reserved_mib": max(row["cuda_peak_reserved_mib"] for row in episodes),
                "terminal_reasons": Counter(row["terminal_reason"] for row in episodes),
            }
            reference = "pi05-10" if baseline == "smolvla-10" and name in ("pi05-5", "pi05-25") else baseline
            if name == reference:
                continue
            differences = defaultdict(list)
            both_succeeded = []
            wins = losses = 0
            for case, row in zip(cases, episodes):
                other = case["episodes"][reference]
                difference = int(row["success"]) - int(other["success"])
                differences[case["case"]["scene"]].append(difference)
                wins += difference == 1
                losses += difference == -1
                if row["success"] and other["success"]:
                    both_succeeded.append(row["steps"] - other["steps"])
            comparisons[name] = {
                "reference": reference, "wins": wins, "losses": losses, "net_wins": wins - losses,
                "success_rate_delta": round((wins - losses) / count, 6),
                "paired_bootstrap_95_ci": paired_interval(differences),
                "both_succeeded": len(both_succeeded),
                "median_step_delta_both_succeeded": median(both_succeeded) if both_succeeded else None,
            }
        return {"paired_conditions": count, "variants": variants, "comparisons": comparisons}

    by_scene = {scene: [case for case in groups if case["case"]["scene"] == scene] for scene in (1, 2, 8)}
    return {**metrics(groups), "by_scene": {str(scene): metrics(cases) for scene, cases in by_scene.items() if cases}}
