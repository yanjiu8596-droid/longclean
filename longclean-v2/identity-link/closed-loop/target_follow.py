"""2×2 development check of target following: cheese context and soup wording."""
import argparse
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from stage4_fixture import GPU, read, require, sha, validate_gpu, validate_study, write
from common import BUDGET, CHEESE, CHUNK, GOALS, SAUCE, SOUP
from legacy_parity import cases, trace
from progress import STRIDE


SUFFIX_BUDGET = 360
LAYOUTS = {2: (931000, 931200), 3: (931005, 931201)}
POLICY_SEEDS = {2: 46, 3: 47}
PROMPTS = {
    "standard": "put the alphabet soup can in the basket",
    "descriptive": "put the soup can with the blue and yellow label in the basket",
}
ARMS = ((False, "standard"), (True, "descriptive"), (True, "standard"), (False, "descriptive"))


def planned(card):
    rows = []
    for seen_index, layout in enumerate(LAYOUTS[card]):
        offset = card - 2 + 2 * seen_index
        for prefix, prompt_id in ARMS[offset:] + ARMS[:offset]:
            rows.append({"card": card, "layout_seed": layout, "policy_seed": POLICY_SEEDS[card],
                         "fresh": bool(seen_index), "prefix": prefix, "prompt_id": prompt_id,
                         "prompt": PROMPTS[prompt_id]})
    return rows


def check_prefix(prefix):
    """Gate fresh prefix qualification only on the unchanged prefix's valid end."""
    return ("failure" not in prefix and bool(prefix["chunks"])
            and prefix["chunks"][-1]["cheese_truth"] is True
            and prefix["chunks"][-1]["end"] == prefix["end"])


def prepare_prefix(api, bddl, policy, smoke, output, layout, seen):
    dest = output / str(layout) / "prefix-setup"
    dest.mkdir(parents=True, exist_ok=False)
    prefix_path = dest / "prefix.json"
    fixture_sha = None
    if seen is None:
        prefix = api.make_prefix(policy, bddl, dest, layout)
        require(read(prefix_path) == prefix, "Original prefix builder did not save the emitted prefix")
    else:
        f = smoke / "fixtures" / str(layout)
        fixture_sha = sha(f / "fixture.json")
        require(fixture_sha == seen["fixture_sha256"] and sha(f / "prefix.json") == seen["prefix_sha256"],
                "Sealed smoke prefix changed")
        copyfile(f / "prefix.json", prefix_path)
        prefix = read(prefix_path)
        require(prefix["initial"]["state_sha256"] == seen["initial_state_sha256"]
                and prefix["initial"]["raw_sha256"] == seen["raw_sha256"], "Seen prefix starts changed")
    available = check_prefix(prefix)
    if seen is not None:
        require(available, "Prescribed seen cheese prefix is not reusable")
    record = {"layout_seed": layout, "fresh": seen is None, "available": available,
              "reason": None if available else prefix.get("failure", "cheese_prefix_unavailable"),
              "observed_cheese": prefix["observed_cheese"], "initial": prefix["initial"], "end": prefix["end"],
              "prefix": prefix_path.relative_to(output).as_posix(), "sha256": sha(prefix_path),
              "fixture_sha256": fixture_sha}
    return prefix, record


def first_entry(rows, allowed, fork_step):
    entered = [tr for row in rows for tr in row["transitions"]
               if tr["object"] in allowed and tr["in_basket"] and tr["step"] > fork_step]
    if not entered:
        return None
    first = min(tr["step"] for tr in entered)
    return {"step": first, "suffix_step": first - fork_step,
            "objects": sorted({tr["object"] for tr in entered if tr["step"] == first})}


def run_arm(api, bddl, policy, root, row, prefix, setup):
    destination = (root / str(row["layout_seed"]) /
                   ("cheese-prefix" if row["prefix"] else "no-prefix") / row["prompt_id"])
    model, object_frames = [], []
    first_evidence = {}
    clock = perf_counter()
    with api.Worker(bddl, destination, row["layout_seed"]) as worker:
        initial = api.fingerprint(worker.current)
        require(initial == setup["initial"], "Within-layout raw initialization changed")
        if row["prefix"]:
            api.replay_prefix(worker, prefix, api.Progress())
            require(api.fingerprint(worker.current) == setup["end"], "Unequal cheese-fork replay")
        fork = api.fingerprint(worker.current)
        start = fork["step"]
        expected_goals = {obj: bool(row["prefix"] and obj == CHEESE) for obj in GOALS}
        require(worker.current["audit"]["goals"] == expected_goals and not worker.terminal()
                and not worker.current["audit"]["held_soup"] and start % CHUNK == 0,
                "Invalid initial goal or control state")
        copyfile(worker.npz, destination / "fork.npz")
        limit = start + SUFFIX_BUDGET
        require(limit <= BUDGET, "Existing prefix does not leave the fixed diagnostic budget")
        policy.reset(row["policy_seed"])
        while worker.current["step"] < limit and not worker.terminal():
            step = worker.current["step"]
            require(step % CHUNK == 0, "Unexpected inference boundary")
            issued = policy.act(worker, row["prompt"], limit=CHUNK)
            require(len(issued) == CHUNK, f"Unexpected action length at {step}")
            samples = []
            for k in range(0, CHUNK, STRIDE):
                actual = worker.step(issued[k:k + STRIDE], SOUP, row["prompt"])
                current = worker.current
                stamped = current["step"]
                samples.append(stamped)
                object_frames.append({"step": stamped, "held_soup": current["audit"]["held_soup"],
                                      "goals": current["audit"]["goals"]})
                for event in current["transitions"]:
                    name = event["object"]
                    if event["in_basket"] and name not in first_evidence:
                        image = destination / f"first-entry-observation-{stamped}.npz"
                        if not image.exists():
                            copyfile(worker.npz, image)
                        first_evidence[name] = {"event_step": event["step"], "sample_step": stamped,
                                                "file": image.name, "sha256": sha(image)}
                require(0 < len(actual) <= STRIDE, "Invalid realized observer stride")
                if worker.terminal():
                    break
            model.append({"start_step": step, "end_step": worker.current["step"], "sensor_steps": samples,
                          "requested": CHUNK})
        final = worker.current
        terminal = api.fingerprint(final)
        require(terminal["step"] == limit or worker.terminal(), "Diagnostic stopped early without an environment stop")
    suffix = [r for r in trace(destination / "actions.jsonl") if r["start_step"] >= start]
    require(suffix and terminal["step"] - start == sum(len(r["actions"]) for r in suffix) <= SUFFIX_BUDGET,
            "Missing or excessive realized diagnostic controls")
    held = next((f["step"] for f in object_frames if f["held_soup"]), None)
    result = {**row, "schema": "v2-stage4-target-follow-arm/1", "status": "complete",
              "prefix_record": setup["prefix"] if row["prefix"] else None,
              "prefix_sha256": setup["sha256"] if row["prefix"] else None,
              "fixture_sha256": setup["fixture_sha256"],
              "initial": initial, "fork": fork, "motion": False, "budget": SUFFIX_BUDGET,
              "actions": "actions.jsonl", "actions_sha256": sha(destination / "actions.jsonl"),
              "fork_npz_sha256": sha(destination / "fork.npz"),
              "suffix_steps": terminal["step"] - start, "terminal_checkpoint": terminal,
              "terminal_goals": final["audit"]["goals"],
              "terminal_reason": ("goal" if all(final["audit"]["goals"].values())
                                  else "environment_done" if final["done"] else "budget"),
              "model_chunks": model, "model_calls": len(model), "gpu_placement": policy.placement,
              "first_soup_sauce_entry": first_entry(suffix, {SOUP, SAUCE}, start),
              "first_any_entry": first_entry(suffix, GOALS, start), "first_entry_evidence": first_evidence,
              "first_sampled_soup_held_step": held, "elapsed_seconds": round(perf_counter() - clock, 6)}
    target = destination / "result.json"
    write(target, result)
    return target, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", required=True, choices=GPU, type=int)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--smoke", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    smoke, out = a.smoke.resolve(), a.output.resolve()
    require(out != smoke and not out.is_relative_to(smoke), "Keep development separate from frozen smoke")
    validate_gpu(a.card)
    bddl, _ = validate_study(a.study.resolve())
    seen, = [r for r in cases(smoke) if r["card"] == a.card]
    require(seen["layout_seed"] == LAYOUTS[a.card][0] and seen["policy_seed"] == POLICY_SEEDS[a.card],
            "Unexpected layout assignment in frozen smoke")
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"target-follow-plan-{a.card}.json"
    marker = out / f"target-follow-{a.card}.json"
    require(not plan_file.exists() and not marker.exists(), "No resampling a separate attempt")

    import run as api

    require(api.instruction(SOUP) == PROMPTS["standard"], "The original soup wording changed")
    plan = {"schema": "v2-stage4-target-follow-plan/1", "card": a.card, "scope": "development_only",
            "smoke_freeze_sha256": sha(smoke / "frozen.json"), "bddl_sha256": sha(bddl),
            "source_sha256": sha(Path(__file__)), "motion": False, "budget": SUFFIX_BUDGET,
            "policy_chunk": CHUNK, "observer_stride": STRIDE, "cases": planned(a.card)}
    write(plan_file, plan)
    manifest = {"schema": "v2-stage4-target-follow-manifest/1", "complete": False, "card": a.card,
                "plan_sha256": sha(plan_file), "prefixes": [], "records": []}
    write(marker, manifest)
    policy = api.Pi05(GPU[a.card])
    for layout in LAYOUTS[a.card]:
        prefix, setup = prepare_prefix(api, bddl, policy, smoke, out, layout,
                                       seen if layout == seen["layout_seed"] else None)
        manifest["prefixes"].append(setup)
        write(marker, manifest)
        print("TARGET_PREFIX", layout, setup["available"], setup["reason"], flush=True)
        for row in (r for r in plan["cases"] if r["layout_seed"] == layout):
            if row["prefix"] and not setup["available"]:
                record = {**row, "status": "prefix_unavailable", "reason": setup["reason"],
                          "result": None, "result_sha256": None}
            else:
                target, result = run_arm(api, bddl, policy, out, row, prefix, setup)
                record = {**row, "status": "complete", "result": target.relative_to(out).as_posix(),
                          "result_sha256": sha(target)}
                print("TARGET_FOLLOW", layout, row["prefix"], row["prompt_id"], result["first_soup_sauce_entry"],
                      flush=True)
            manifest["records"].append(record)
            write(marker, manifest)
    manifest["complete"] = True
    write(marker, manifest)


if __name__ == "__main__":
    main()
