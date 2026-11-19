"""Audit holdout assignments, trigger times and native endpoints on CPU."""

import argparse
from itertools import combinations
from math import dist
from pathlib import Path

from basket_retreat_audit import check_arm
from basket_retreat_fresh import LAST_TRIGGER, LAYOUTS, gate
from basket_stall_probe import GAP, LOWER, MAX_SPREAD, UPPER, WINDOW
from common import BUDGET, CHEESE, CHUNK, SAUCE, SOUP, read, sha, write
from dense_online_diagnostic import checked, load_arm, require
from target_follow_audit import observation, stamp


RUNNER_SHA = "9461213ab03c14d480a9f813aff61769c9740f741d453842558363b81219e94a"


def find_trigger(root, layout, baseline):
    folder = root / "shadow" / str(layout)
    samples = read(folder / "samples.json")
    release = next((event["step"] for event in baseline["result"]["state_events"]
                    if event["event"] == "in_basket" and event["object"] == SAUCE), None)
    endpoint = baseline["endpoint"]
    expected_steps = [] if release is None else [s for s in endpoint if release <= s <= release + WINDOW and s % CHUNK == 0]
    require([r["step"] for r in samples] == expected_steps, "Archived sensor window is incomplete")
    first = None
    for index, r in enumerate(samples):
        step = r["step"]
        model_target = baseline["decisions"].get(step, {}).get("target")
        saved = r["observation"]
        require(saved["fingerprint"] == stamp(endpoint[step]) and r["target"] == model_target
                and r["first_sauce_commit"] == release and r["check"]["open_gap"] is (r["gap_m"] >= GAP)
                and r["check"]["above_basket"] is all(lo <= x <= hi for x, lo, hi in zip(r["eef_m"], LOWER, UPPER)),
                "Recorded observation or scheduling gate changed")
        observation(checked(folder, saved["file"], saved["sha256"]), saved["fingerprint"])
        group = samples[max(0, index - 2): index + 1]
        elapsed = [p["step"] for p in group]
        spread = max((dist(a["eef_m"], b["eef_m"]) for a, b in combinations(group, 2)), default=0.)
        trigger = elapsed == [step - 20, step - 10, step] and spread <= MAX_SPREAD and all(
            all(p["check"].values()) and p["target"] == SOUP and p["local_can_vote"] is None for p in group)
        require(r["trigger"] is trigger and elapsed == r["window_steps"], "Trigger differs from the saved observations")
        if first is None and trigger:
            first = r
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "source-root", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args()
    root = a.run.resolve()
    require(not a.out.exists(), "Output already exists")
    _, previous = gate(a.source_root.resolve())
    s = read(root / "summary.json")
    plan = read(checked(root, "plan.json", s["plan_sha256"]))
    require(s["producer_sha256"] == plan["runner_sha256"] == RUNNER_SHA and plan["layouts"] == list(LAYOUTS)
            and plan["budget"] == BUDGET and [r["layout_seed"] for r in s["rows"]] == list(LAYOUTS), "Changed cohort or budget")
    cases = []
    for row in s["rows"]:
        layout, status = row["layout_seed"], row["status"]
        result = {"layout_seed": layout, "status": status}
        cases.append(result)
        if status == "skipped_after_safety_stop":
            require(s["safety_stop"], "Unjustified missing layout")
            continue
        fixture = read(checked(root, f"fixtures/{layout}/fixture.json", row["fixture_sha256"]))
        require(fixture == row["fixture"], "Changed prefix fixture")
        for key, known in previous.items():
            digest = fixture["initial"][key]
            require(digest not in known, "Duplicate initial state")
            known.add(digest)
        if not fixture["qualified"]:
            require(status == "common_cheese_prefix_unavailable" and "baseline" not in row, "Unqualified prefix was executed")
            continue
        base = load_arm(root, {**row["baseline"], "layout_seed": layout, "arm": "adaptive1"})
        terminal = base["physical"][base["result"]["terminal"]["step"]]
        success = all(terminal["goals"].values()) and not any(terminal["held_by_target"].values())
        require(row["baseline"]["official_success_all_unheld"] is success and base["result"]["initial"] == fixture["initial"],
                "Baseline score or fixture differs")
        trigger = find_trigger(root, layout, base)
        require(row["first_trigger"] == trigger, "Published first trigger is not the first valid decision")
        result.update(baseline=success, trigger_step=None if trigger is None else trigger["step"])
        if status == "paired_trigger_complete":
            require(trigger is not None and trigger["step"] <= LAST_TRIGGER, "Paired branch started outside the budget")
            case = {"layout_seed": layout, "cutoff": trigger["step"], "policy_seed": 48 + layout % 2}
            for mode in ("hold", "retreat"):
                item = {"result": f"continuations/{layout}/{mode}/result.json", "result_sha256": row[mode]["result_sha256"]}
                arm = check_arm(root, item, case, base, mode)
                require(arm == row[mode], "Published score differs from archived continuation")
                result[mode] = arm["official_success_all_unheld"]
            result["candidate"] = result["retreat"]
        elif status in ("same_path_no_trigger", "same_path_not_enough_budget"):
            require(trigger is None or trigger["step"] > LAST_TRIGGER, "Missing eligible continuation")
            result["candidate"] = success
        else:
            require(s["safety_stop"] and status in ("unsafe_trigger", "baseline_observer_safety_stop"), "Unrecognized experiment status")
            result["candidate"] = False
        require(result["candidate"] is row["candidate_success_all_unheld"], "Published candidate result differs")
    completed = [r for r in cases if "baseline" in r]
    paired = [r for r in completed if "retreat" in r]
    names = ("qualifying_prefixes", "triggered_pair_layouts", "baseline_success_all_unheld", "candidate_success_all_unheld",
             "candidate_wins", "candidate_losses", "hold_success_all_unheld_on_triggered", "retreat_success_all_unheld_on_triggered")
    values = (len(completed), len(paired), sum(r["baseline"] for r in completed), sum(r["candidate"] for r in completed),
              sum(not r["baseline"] and r["candidate"] for r in completed), sum(r["baseline"] and not r["candidate"] for r in completed),
              sum(r["hold"] for r in paired), sum(r["retreat"] for r in paired))
    require(all(s["metrics"][key] == value for key, value in zip(names, values, strict=True)), "Published aggregate differs")
    write(a.out, {"schema": "v2-basket-retreat-holdout-audit/1", "summary_sha256": sha(root / "summary.json"),
          "auditor_sha256": sha(Path(__file__)), "cases": cases, "metrics": s["metrics"],
          "meets_development_gate": s["meets_development_gate"], "safety_stop": s["safety_stop"]})


if __name__ == "__main__":
    main()
