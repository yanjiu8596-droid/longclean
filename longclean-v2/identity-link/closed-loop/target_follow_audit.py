"""CPU audit of the stage 4 target-follow 2×2 development diagnostic."""

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
sys.path.insert(0, str(V2))
from common import (CAMERAS, CHEESE, CHUNK, GOALS, GPU, PREFIX_TASK, SAUCE, SOUP,
                    array_hash, read, sha, write)

WINDOW = 360
STRIDE = 5
BDDL = "32f2219ef8e19d4ffe1c68ad17a2493dcf0b8601a6d143f2866bab06aff8a0ac"
SMOKE_FREEZE = "428f6c06819003a2b31ec9e4563e0d89684a8eda7d286a5a7a919f9b13968f24"
LEGACY = {2: "af44740fb0f269e973314bea99db6e1dc61abd209810326c42a2b78896cdc059",
          3: "4c935e0569f04ccbb1efe101a1d95bbb5306d9896b377f1c9dc906d724be0db1"}
LAYOUTS = {2: ((931000, 46, False), (931200, 46, True)),
           3: ((931005, 47, False), (931201, 47, True))}
PROMPTS = {"standard": "put the alphabet soup can in the basket",
           "descriptive": "put the soup can with the blue and yellow label in the basket"}
ORDER = [(False, "standard"), (True, "descriptive"), (True, "standard"), (False, "descriptive")]
COMPARISONS = [((True, "standard"), (True, "descriptive")),
               ((False, "standard"), (False, "descriptive")),
               ((True, "standard"), (False, "standard")),
               ((True, "descriptive"), (False, "descriptive"))]
FINGERPRINT = ("step", "raw_sha256", "bddl_sha256", "state_sha256", "observation_sha256")
OBSERVATIONS = {"eef_pos", "eef_quat", "gripper_qpos"} | {
    name + suffix for name in CAMERAS for suffix in ("", "_depth", "_intrinsic", "_extrinsic")}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def stamp(row):
    result = {k: row[k] for k in FINGERPRINT}
    require(type(result["step"]) is int and result["step"] >= 0 and result["bddl_sha256"] == BDDL
            and all(digest(result[k]) for k in FINGERPRINT[1:4])
            and set(result["observation_sha256"]) == OBSERVATIONS
            and all(digest(h) for h in result["observation_sha256"].values()), "Malformed fingerprint")
    return result


def checked_path(root, name, checksum):
    require(isinstance(name, str) and digest(checksum), f"Missing path or digest: {name}")
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root) and path.is_file()
            and sha(path) == checksum, f"Artifact path/checksum mismatch: {name}")
    return path


def expected_rows(card):
    rows = []
    for layout, seed, fresh in LAYOUTS[card]:
        start = card - 2 + 2 * int(fresh)
        for prefix, prompt in ORDER[start:] + ORDER[:start]:
            rows.append({"layout_seed": layout, "policy_seed": seed, "card": card, "fresh": fresh,
                         "prefix": prefix, "prompt_id": prompt, "prompt": PROMPTS[prompt]})
    return rows


def matches(row, expected):
    return all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items())


def classify(first):
    if not first:
        return "neither"
    return {frozenset((SOUP,)): "soup_first", frozenset((SAUCE,)): "sauce_first",
            frozenset((SAUCE, SOUP)): "simultaneous"}[frozenset(first["objects"])]


def first_entry(first_steps, objects, start):
    entered = {obj: first_steps[obj] for obj in objects if obj in first_steps}
    if not entered:
        return None
    step = min(entered.values())
    return {"step": step, "suffix_step": step - start,
            "objects": sorted(obj for obj, when in entered.items() if when == step)}


def audit_trace(path, expected, initial, fork, prefix):
    """Reconstruct the exact first native action step from transitions, including ties."""
    prefix_chunks = prefix["chunks"] if prefix else []
    last, start = 0, fork["step"]
    goals = dict.fromkeys(GOALS, False)
    first_steps, suffix, boundary = {}, [], []
    tail = None
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            endpoint = stamp(row)
            n = len(row["actions"])
            require(not tail or not (tail["done"] or all(goals.values())), f"Actions after terminal state: {path}")
            require(row["start_step"] == last and endpoint["step"] == last + n
                    and 0 < n <= CHUNK and row["executed"] == n and endpoint["step"] <= start + WINDOW
                    and endpoint["raw_sha256"] == initial["raw_sha256"]
                    and type(row["done"]) is bool, f"Invalid control interval: {path}:{last}")
            require(all(isinstance(action, list) and len(action) == 7
                        and all(type(a) in (int, float) and isfinite(a) for a in action)
                        for action in row["actions"]), f"Invalid realized OSC action: {path}:{last}")
            is_prefix = index < len(prefix_chunks)
            if is_prefix:
                chunk = prefix_chunks[index]
                require(row["target"] == CHEESE and row["instruction"] == PREFIX_TASK
                        and endpoint == stamp(chunk["end"]) and row["actions"] == chunk["actions"],
                        f"Executed prefix differs from frozen actions: {path}:{last}")
            else:
                require(last >= start and (last - start) % STRIDE == 0 and n <= STRIDE
                        and row["target"] == SOUP and row["instruction"] == expected["prompt"],
                        f"Unexpected suffix stride or instruction: {path}:{last}")
                if suffix:
                    require(len(suffix[-1]["actions"]) == STRIDE, f"Unexpected short nonfinal suffix half: {path}")
                else:
                    require(last == start and goals == {o: bool(prefix and o == CHEESE) for o in GOALS},
                            f"Prompt was not applied to the expected untreated goals: {path}")
                suffix.append(row)
            touched, prev = set(), last
            for tr in row["transitions"]:
                when, obj, value = tr["step"], tr["object"], tr["in_basket"]
                if when != prev:
                    touched.clear()
                require(type(when) is int and last < when <= endpoint["step"] and prev <= when and obj in GOALS
                        and type(value) is bool and obj not in touched and goals[obj] is not value,
                        f"Invalid native predicate transition: {path}:{tr}")
                # The same object can leave/reenter in one block, but never toggle twice at one action step.
                touched.add(obj)
                prev = when
                goals[obj] = value
                if value and not is_prefix:
                    first_steps.setdefault(obj, when)
            official = row["audit"]["goals"]
            require(set(official) == set(GOALS) and all(type(v) is bool for v in official.values())
                    and goals == official and type(row["audit"]["held_soup"]) is bool,
                    f"Transitions do not reconstruct official endpoint: {path}:{last}")
            last, tail = endpoint["step"], row
            if not is_prefix and (last - start) % CHUNK == 0:
                boundary.append((last, goals.copy(), row["audit"]["held_soup"], row["done"]))
    require(suffix and tail and (tail["done"] or all(goals.values()) or last == start + WINDOW),
            f"Missing suffix or truncated nonterminal trajectory: {path}")
    group = []
    for row in suffix:
        if (row["start_step"] - start) % CHUNK == 0:
            group.append({"start_step": row["start_step"], "sensor_steps": [], "end_step": row["step"],
                          "requested": CHUNK})
        group[-1]["sensor_steps"].append(row["step"])
        group[-1]["end_step"] = row["step"]
    return first_steps, tail, group, boundary, suffix


def summarize(records):
    by_layout = {r[0]: {} for cases in LAYOUTS.values() for r in cases}
    for r in records:
        by_layout[r["layout_seed"]][r["prefix"], r["prompt_id"]] = r
    summaries = []
    for prefix, prompt in ORDER:
        rows = [by_layout[k][prefix, prompt] for k in sorted(by_layout)]
        counts = Counter(r["category"] for r in rows)
        qualified = bool(not counts["prefix_unavailable"] and counts["soup_first"] >= 3
                         and all(r["category"] == "soup_first" for r in rows if r["fresh"]))
        summaries.append({"prefix": prefix, "prompt_id": prompt, "expected_layouts": len(rows),
                          "executed_layouts": len(rows) - counts["prefix_unavailable"],
                          "missing_layouts": [r["layout_seed"] for r in rows if r["category"] == "prefix_unavailable"],
                          "no_soup_layouts": [r["layout_seed"] for r in rows if r["status"] == "complete"
                                              and SOUP not in r["first_entry_steps"]],
                          "counts": {k: counts[k] for k in ("soup_first", "sauce_first", "simultaneous", "neither",
                                                            "prefix_unavailable")},
                          "soup_first_both_fresh": all(r["category"] == "soup_first" for r in rows if r["fresh"]),
                          "stable_soup_first_at_boundary": sum(r.get("soup_pending_sauce_boundary") is not None for r in rows),
                          "ready_for_observer_diagnostic_only": qualified})
    contrasts = []
    for source, changed in COMPARISONS:
        paired = []
        for layout, rows in sorted(by_layout.items()):
            a, b = rows[source], rows[changed]
            if "prefix_unavailable" in (a["category"], b["category"]):
                change = "unavailable_pair"
            else:
                before, after = a["category"] == "soup_first", b["category"] == "soup_first"
                change = "gain" if after and not before else "loss" if before and not after else "no_change"
            paired.append({"layout_seed": layout, "fresh": a["fresh"], "from": a["category"], "to": b["category"],
                           "from_first": a.get("first_soup_sauce_entry"), "to_first": b.get("first_soup_sauce_entry"),
                           "change": change})
        contrasts.append({"from": {"prefix": source[0], "prompt_id": source[1]},
                          "to": {"prefix": changed[0], "prompt_id": changed[1]},
                          "paired": paired, "counts": dict(Counter(row["change"] for row in paired))})
    preference = [(True, "standard"), (True, "descriptive"), (False, "standard"), (False, "descriptive")]
    passed = {(r["prefix"], r["prompt_id"]) for r in summaries if r["ready_for_observer_diagnostic_only"]}
    selected = next(({"prefix": p, "prompt_id": i} for p, i in preference if (p, i) in passed), None)
    return summaries, contrasts, selected


def observation(path, fingerprint):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == OBSERVATIONS and
                all(array_hash(archive[k]) == fingerprint["observation_sha256"][k] for k in OBSERVATIONS),
                f"Saved RGB-D/state does not match trace observation: {path}")


def audited_setup(root, row, layout, fresh, seen):
    name = f"{layout}/prefix-setup/prefix.json"
    require(type(row["layout_seed"]) is int and row["layout_seed"] == layout and row["fresh"] is fresh
            and row["prefix"] == name, f"Invalid prefix setup identity: {layout}")
    path = checked_path(root, name, row["sha256"])
    p = read(path)
    first, end = stamp(p["initial"]), stamp(p["end"])
    require(first == stamp(row["initial"]) and end == stamp(row["end"]) and first["step"] == 0
            and first["raw_sha256"] == end["raw_sha256"] and p["observed_cheese"] is row["observed_cheese"]
            and type(p["observed_cheese"]) is bool,
            f"Manifest and actual prefix setup disagree: {layout}")
    available = ("failure" not in p and bool(p["chunks"]) and p["chunks"][-1]["cheese_truth"] is True
                 and stamp(p["chunks"][-1]["end"]) == end and end["step"] % CHUNK == 0)
    if seen:
        require(available and row["fixture_sha256"] == seen["fixture_sha256"]
                and row["sha256"] == seen["prefix_sha256"]
                and first["raw_sha256"] == seen["raw_sha256"]
                and first["state_sha256"] == seen["initial_state_sha256"], f"Sealed seen prefix differs: {layout}")
    else:
        require(row["fixture_sha256"] is None, f"Fresh layout incorrectly reused a fixture: {layout}")
    reason = None if available else p.get("failure", "cheese_prefix_unavailable")
    require(row["available"] is available and row["reason"] == reason, f"Prefix availability differs: {layout}")
    return p


def audit_arm(root, record, expected, setup, prefix):
    tag = "cheese-prefix" if expected["prefix"] else "no-prefix"
    name = f"{expected['layout_seed']}/{tag}/{expected['prompt_id']}/result.json"
    require(record["status"] == "complete" and record["result"] == name, f"Unexpected record location: {name}")
    saved = checked_path(root, name, record["result_sha256"])
    r = read(saved)
    require(matches(r, expected) and r["schema"] == "v2-stage4-target-follow-arm/1" and r["status"] == "complete"
            and r["motion"] is False and r["budget"] == WINDOW and r["actions"] == "actions.jsonl"
            and r["fixture_sha256"] == setup["fixture_sha256"]
            and r["prefix_record"] == (setup["prefix"] if expected["prefix"] else None)
            and r["prefix_sha256"] == (setup["sha256"] if expected["prefix"] else None), f"Arm metadata differs: {name}")
    initial, fork = stamp(r["initial"]), stamp(r["fork"])
    require(initial == stamp(setup["initial"]) and fork == stamp(setup["end"] if expected["prefix"] else setup["initial"]),
            f"Not the shared within-layout initial state and quiet fork: {name}")
    placement = r["gpu_placement"]
    require(placement["uuid"] == GPU[expected["card"]] and placement["egl_device_id"] == str(expected["card"]),
            f"Wrong CUDA/EGL device: {name}")
    elapsed = r["elapsed_seconds"]
    require(type(elapsed) in (float, int) and isfinite(elapsed) and elapsed >= 0, f"Invalid wall time: {name}")
    observation(checked_path(saved.parent, "fork.npz", r["fork_npz_sha256"]), fork)
    action_path = checked_path(saved.parent, r["actions"], r["actions_sha256"])
    first_steps, tail, model, boundary, suffix = audit_trace(action_path, expected, initial, fork,
                                                          prefix if expected["prefix"] else None)
    first = first_entry(first_steps, (SOUP, SAUCE), fork["step"])
    any_first = first_entry(first_steps, GOALS, fork["step"])
    end, goals = stamp(tail), tail["audit"]["goals"]
    reason = "goal" if all(goals.values()) else "environment_done" if tail["done"] else "budget"
    held = next((row["step"] for row in suffix if row["audit"]["held_soup"]), None)
    require(r["terminal_checkpoint"] == end and r["terminal_goals"] == goals and r["terminal_reason"] == reason
            and r["suffix_steps"] == end["step"] - fork["step"] and r["model_chunks"] == model
            and r["model_calls"] == len(model) and r["first_soup_sauce_entry"] == first
            and r["first_any_entry"] == any_first and r["first_sampled_soup_held_step"] == held,
            f"Result differs from independent native-step reconstruction: {name}")
    evidence = r["first_entry_evidence"]
    require(set(evidence) == set(first_steps), f"Missing first-entry evidence: {name}")
    verified = set()
    for obj, step in first_steps.items():
        hit = evidence[obj]
        sample = next(row for row in suffix if row["start_step"] < step <= row["step"])
        file = f"first-entry-observation-{sample['step']}.npz"
        require(hit["event_step"] == step and hit["sample_step"] == sample["step"] and hit["file"] == file,
                f"Evidence sample mislabeled: {name}:{obj}")
        obs_path = checked_path(saved.parent, file, hit["sha256"])
        if obs_path not in verified:
            observation(obs_path, stamp(sample))
            verified.add(obs_path)
    category = classify(first)
    stable = None
    if category == "soup_first":
        # This is optional evidence of headroom, not a claim that the new writer would commit.
        at = next((x for x in boundary if x[0] >= first["step"]), None)
        if (at and at[1][SOUP] and not at[1][SAUCE] and not at[2] and not at[3]
                and at[0] < end["step"] and first_steps.get(SAUCE, at[0] + 1) > at[0]):
            stable = {"step": at[0], "suffix_step": at[0] - fork["step"], "goals": at[1], "held_soup": at[2]}
    return {**expected, "status": "complete", "category": category, "initial": initial, "fork": fork,
            "result": name, "result_sha256": record["result_sha256"], "actions_sha256": r["actions_sha256"],
            "fork_npz_sha256": r["fork_npz_sha256"], "first_soup_sauce_entry": first, "first_any_entry": any_first,
            "first_entry_steps": first_steps, "first_sampled_soup_held_step": held,
            "soup_pending_sauce_boundary": stable, "terminal": end, "terminal_reason": reason,
            "terminal_goals": goals, "suffix_steps": r["suffix_steps"], "elapsed_seconds": elapsed,
            "elapsed_minutes": round(elapsed / 60, 5), "model_calls": len(model)}


def historical(rows, legacy_paths):
    summary = []
    for card, path in legacy_paths.items():
        require(path.is_file() and sha(path) == LEGACY[card], f"Original soup-wording run changed: {path}")
        old = read(path)
        old_source = old["source"]
        seed, policy_seed, _ = LAYOUTS[card][0]
        new, = [r for r in rows if r["layout_seed"] == seed and r["prefix"] and r["prompt_id"] == "standard"]
        require(old["layout_seed"] == seed and old["policy_seed"] == policy_seed and old["card"] == card
                and old["freeze_sha256"] == SMOKE_FREEZE and old_source["prompt"] == PROMPTS["standard"]
                and old_source["suffix_budget"] == WINDOW, f"Unexpected historical reference: {path}")
        old_steps = {obj: step for obj, step in old_source["first_official_enter_steps"].items() if step is not None}
        same_fork = stamp(old_source["fork"]) == new["fork"]
        if old["moved"]:
            require(not same_fork, f"Moved historical fork incorrectly matches a quiet fork: {seed}")
        require(old["moved"] is (card == 2), f"Unexpected historical displacement: {path}")
        checks = {"actions_sha256": old_source["actions_sha256"] == new["actions_sha256"],
                  "terminal": stamp(old_source["terminal"]) == new["terminal"],
                  "first_entry_steps": old_steps == new["first_entry_steps"]}
        summary.append({"layout_seed": seed, "historical_sha256": LEGACY[card], "historical_moved": old["moved"],
                        "comparable_fork": same_fork and not old["moved"], "exact_checks": None if old["moved"] else checks,
                        "matches_original": None if old["moved"] else same_fork and all(checks.values()),
                        "historical_first_entry_steps": old_steps, "new_first_entry_steps": new["first_entry_steps"],
                        "explanation": "Historical run displaced the soup; the factorial cell is quiet."
                        if old["moved"] else "Same intended quiet fork, budget, seed and original soup wording."})
    return summary


def audit(root, out, smoke, legacy_paths):
    require(not out.exists() and sha(smoke) == SMOKE_FREEZE, "Audit output exists or original freeze differs")
    rows, stamps, initials, seen_results = [], [], [], set()
    sealed = read(smoke)
    frozen_cases = {r["layout_seed"]: r for r in sealed["cases"]}
    core = {key: sealed["code"][key] for key in ("../../common.py", "../../run.py", "../../simulator.py", "../../vision.py")}
    require(all(sha(HERE / name) == digest for name, digest in core.items()), "A frozen native source changed")
    runner = HERE / "target_follow.py"
    code = sha(runner)
    for card, layouts in LAYOUTS.items():
        plan_path = root / f"target-follow-plan-{card}.json"
        manifest_path = root / f"target-follow-{card}.json"
        p, m = read(plan_path), read(manifest_path)
        expected = expected_rows(card)
        require(matches(p, {"schema": "v2-stage4-target-follow-plan/1", "card": card, "scope": "development_only",
                            "smoke_freeze_sha256": SMOKE_FREEZE, "bddl_sha256": BDDL, "source_sha256": code,
                            "motion": False, "budget": WINDOW, "policy_chunk": CHUNK, "observer_stride": STRIDE})
                and p["cases"] == expected, f"Frozen development plan differs: card {card}")
        require(matches(m, {"schema": "v2-stage4-target-follow-manifest/1", "card": card, "complete": True,
                            "plan_sha256": sha(plan_path)}) and len(m["prefixes"]) == len(layouts)
                and len(m["records"]) == len(expected), f"Incomplete or changed per-card manifest: {card}")
        stamps.append({"card": card, "plan_sha256": sha(plan_path), "manifest_sha256": sha(manifest_path)})
        setups = {}
        for (layout, _, fresh), setup in zip(layouts, m["prefixes"], strict=True):
            parsed = audited_setup(root, setup, layout, fresh, frozen_cases.get(layout))
            setups[layout] = (setup, parsed)
            initials.append(stamp(setup["initial"]))
        for expect, record in zip(expected, m["records"], strict=True):
            require(matches(record, expect), f"Run order differs from immutable plan: {record}")
            setup, prefix = setups[expect["layout_seed"]]
            if expect["prefix"] and not setup["available"]:
                require(record["status"] == "prefix_unavailable" and record["reason"] == setup["reason"]
                        and record["result"] is None and record["result_sha256"] is None,
                        f"Unavailable layout did not retain its prespecified position: {record}")
                rows.append({**expect, "status": record["status"], "category": "prefix_unavailable",
                             "reason": setup["reason"], "prefix_record_sha256": setup["sha256"]})
            else:
                arm = audit_arm(root, record, expect, setup, prefix)
                require(arm["result"] not in seen_results, f"Duplicate experiment result: {arm['result']}")
                seen_results.add(arm["result"])
                rows.append(arm)
    require(len(initials) == 4 and all(len({fp[key] for fp in initials}) == 4
                                       for key in ("raw_sha256", "state_sha256")), "Layouts share an initial state")
    cells, contrasts, selected = summarize(rows)
    parity = historical(rows, legacy_paths)
    archive = {"schema": "v2-stage4-target-follow-audit/1", "scope": "development_only",
               "smoke_freeze_sha256": SMOKE_FREEZE, "runner_sha256": code, "audit_sha256": sha(Path(__file__)),
               "frozen_native_source_sha256": core,
               "manifests": stamps, "primary": "First native false-to-true official entry of soup versus sauce after fork; ties separate.",
               "gate_definition": "All four layouts available and soup first in at least three including both fresh layouts.",
               "historical_defaults": parity, "cells": cells, "paired_contrasts": contrasts, "records": rows,
               "recommendation": selected, "pilot_release": False,
               "total_realized_rollouts": len(seen_results), "total_assigned_cells": len(rows)}
    write(out, archive)
    print("TARGET_FOLLOW_AUDIT", len(seen_results), "/", len(rows),
          [(c["prefix"], c["prompt_id"], c["counts"]) for c in cells], "recommendation", selected, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke-freeze", type=Path, help="Path to the sealed stage4-smoke/frozen.json")
    p.add_argument("--legacy", type=Path, help="Directory containing progress-policy-{2,3}.json")
    a = p.parse_args()
    root = a.run.resolve()
    smoke = a.smoke_freeze or root.parent / "stage4-smoke" / "frozen.json"
    if a.smoke_freeze is None and not smoke.is_file():
        smoke = HERE / "stage4-smoke-frozen.json"
    legacy_paths = {card: (a.legacy or root.parent / "stage4-policy-smoke") / f"progress-policy-{card}.json"
                    for card in LAYOUTS}
    if a.legacy is None and not all(path.is_file() for path in legacy_paths.values()):
        legacy_paths = {card: HERE / f"stage4-policy-{card}.json" for card in LAYOUTS}
    audit(root, a.out.resolve(), smoke.resolve(), legacy_paths)


if __name__ == "__main__":
    main()
