"""Holdout development rollout of the frozen basket-stall retreat."""

import argparse
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, require, validate_gpu, validate_study
from common import BUDGET, CHEESE, CHUNK, GOALS, SAUCE, SOUP, read, sha, write
from actual_progress_replay import ActualProgress
from actual_physics_worker import worker_class
from basket_retreat_pair import FIRST_GRASP_WINDOW, LIFT, TRANSLATE, run_arm as continue_arm
from basket_retreat_audit import check_arm
from basket_stall_probe import WINDOW, archive, sensor_position, sensor_sample, verify_sources
from dense_online import prepare_prefix, run_arm as uninterrupted_arm, SAFETY_MARKER
from dense_online_diagnostic import checked, load_arm, RUNNER_SHA
from dense_release_observer import DenseReleaseObserver, frozen_module
from target_follow_audit import observation


HERE = Path(__file__).resolve().parent
DEV_AUDIT_SHA = "30d8dc3dc345d1c6ec16c116db73405016287d3cde68eb427650e53e4c6983a4"
COMBINED_SHA = "7688d5a9f4297ab11e55e41c28e319120771a1efffe2a023490bf83086fe9341"
LAYOUTS = (931400, 931401, 931402, 931403)
LAST_TRIGGER = BUDGET - LIFT - TRANSLATE - FIRST_GRASP_WINDOW


def gate(source):
    pins = {
        "dense_online.py": RUNNER_SHA,
        "basket_stall_probe.py": "149f17ade335e6205b5c0a7bcf3d3f0e8567b2fa96790f1dfddcc44f90743448",
        "basket_retreat_pair.py": "33960485c95107bd7fc5f1cbf874f219ba3b2f9f4b7f10204fbc998d3d68c855",
        "basket_retreat_audit.py": "93490520ff0a64c793a36b7d6d463bec409107d9964cb8b4f5ebe6df12eb2b92",
    }
    for name, digest in pins.items():
        require(sha(HERE / name) == digest, f"Changed frozen code: {name}")
    audit = read(checked(source, "stage4-basket-retreat-pair/audit.json", DEV_AUDIT_SHA))
    require([p["layout_seed"] for p in audit["pairs"]] == [931303, 931307] and all(
        p["arms"]["retreat"]["official_success_all_unheld"] and not p["arms"]["hold"]["official_success"]
        and not p["arms"]["retreat"]["completed_goal_reversals"] and not p["arms"]["retreat"]["intervention_failure"]
        for p in audit["pairs"]), "Seen recovery did not qualify for holdout development")
    seen = read(checked(source, "stage4-dense-online-extra/combined.json", COMBINED_SHA))
    return pins, {key: {row["initial"][key] for row in seen["fixtures"]} for key in ("raw_sha256", "state_sha256")}


class KnownPrefixProgress:
    def __init__(self, observer):
        self.observer = observer

    def complete(self, target):
        return target == CHEESE or target in self.observer.complete


def shadow_observer(folder, samples):
    class ShadowObserver(DenseReleaseObserver):
        def __init__(self):
            super().__init__()
            self.first_sauce = None
            self.progress = KnownPrefixProgress(self)

        def observe(self, frame, actions, step):
            events = super().observe(frame, actions, step)
            if self.first_sauce is None and any(e["object"] == SAUCE and e["event"] == "in_basket" for e in events):
                self.first_sauce = step
            if self.first_sauce is not None and step % CHUNK == 0 and step <= self.first_sauce + WINDOW:
                # The only missing input is the next actual scheduler target. Join its recorded value before any motion.
                sample = sensor_sample(step, frame, SOUP, self.progress, self, samples)
                file = f"boundary-{step}.npz"
                copyfile(frame.source, folder / file)
                sample.update(file=file, sha256=sha(folder / file), first_sauce_commit=self.first_sauce)
                samples.append(sample)
            return events

    return ShadowObserver


def save_initial(api, base):
    class Worker(base):
        def __init__(self, bddl, folder, seed, video=False):
            super().__init__(bddl, folder, seed, video)
            self.initial_snapshot = archive(api, self, folder, "initial.npz")
            write(folder / "initial-observation.json", self.initial_snapshot)

    return Worker


def trigger_from_shadow(api, baseline, samples, folder):
    decision = baseline["decisions"]
    resolved = []
    for provisional in samples:
        step = provisional["step"]
        fingerprint = api.fingerprint(baseline["endpoint"][step])
        path = checked(folder, provisional["file"], provisional["sha256"])
        observation(path, fingerprint)
        target = decision.get(step, {}).get("target")
        row = {**provisional, "target": target, "check": {**provisional["check"],
               "remaining_target_is_soup": target == SOUP and provisional["check"]["remaining_target_is_soup"]}}
        targets_match = all(decision.get(s, {}).get("target") == SOUP for s in row["window_steps"])
        row["base_trigger"] = provisional["base_trigger"] and targets_match
        row["trigger"] = provisional["trigger"] and targets_match
        row["observation"] = {"file": row.pop("file"), "sha256": row.pop("sha256"), "fingerprint": fingerprint}
        resolved.append(row)
    write(folder / "samples.json", resolved)
    return next((row for row in resolved if row["trigger"]), None)


def baseline_summary(root, path, result, source):
    final = source["physical"][result["terminal"]["step"]]
    require(result["terminal_native_audit"]["goals"] == final["goals"] and result["terminal_held_by_target"] == final["held_by_target"],
            "Baseline terminal score differs from the recorded sidecar")
    return {"result": path.relative_to(root).as_posix(), "result_sha256": sha(path), "reason": result["terminal_reason"],
            "terminal_step": result["terminal"]["step"], "official_success": result["official_success"],
            "official_success_all_unheld": result["official_success"] and not any(final["held_by_target"].values()),
            "first_sampled_soup_grasp": next((s for s, row in source["physical"].items() if row["held_by_target"][SOUP]), None)}


def calculate(rows):
    eligible = [r for r in rows if "baseline" in r]
    paired = [r for r in rows if "retreat" in r and "hold" in r]
    baseline = sum(r["baseline"]["official_success_all_unheld"] for r in eligible)
    candidate = sum(r["candidate_success_all_unheld"] for r in eligible)
    wins = sum(r["candidate_success_all_unheld"] and not r["baseline"]["official_success_all_unheld"] for r in eligible)
    losses = sum(r["baseline"]["official_success_all_unheld"] and not r["candidate_success_all_unheld"] for r in eligible)
    return {"assigned_layouts": len(LAYOUTS), "qualifying_prefixes": len(eligible), "triggered_pair_layouts": len(paired),
            "baseline_success_all_unheld": baseline, "candidate_success_all_unheld": candidate,
            "candidate_wins": wins, "candidate_losses": losses,
            "hold_success_all_unheld_on_triggered": sum(r["hold"]["official_success_all_unheld"] for r in paired),
            "retreat_success_all_unheld_on_triggered": sum(r["retreat"]["official_success_all_unheld"] for r in paired)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--card", choices=GPU, type=int, required=True)
    for name in ("study", "source-root", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    validate_gpu(args.card)
    bddl, _ = validate_study(args.study.resolve())
    verify_sources()
    pins, seen = gate(args.source_root.resolve())
    import run as api

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stop = output / SAFETY_MARKER
    plan = {"schema": "v2-basket-retreat-holdout-plan/1", "layouts": list(LAYOUTS), "bddl_sha256": sha(bddl),
            "runner_sha256": sha(Path(__file__)), "code": pins, "seen_audit_sha256": DEV_AUDIT_SHA,
            "budget": BUDGET, "last_usable_trigger": LAST_TRIGGER, "card": args.card, "recovery_controls": LIFT + TRANSLATE,
            "comparison": ["uninterrupted_adaptive1", "hold_if_triggered", "retreat_if_triggered"]}
    write(output / "plan.json", plan)
    native_worker = worker_class(api)
    capturing_worker = save_initial(api, native_worker)
    policy, locate = api.Pi05(GPU[args.card]), api.load_locator()
    rows = []
    for layout in LAYOUTS:
        row = {"layout_seed": layout}
        rows.append(row)
        if stop.exists():
            row["status"] = "skipped_after_safety_stop"
            continue
        case = {"layout_seed": layout, "policy_seed": 48 + layout % 2, "card": args.card}
        fixture_path = output / "fixtures" / str(layout)
        fixture = prepare_prefix(api, policy, bddl, case, fixture_path)
        row["fixture"] = fixture
        row["fixture_sha256"] = sha(fixture_path / "fixture.json")
        for field, known in seen.items():
            digest = fixture["initial"][field]
            require(digest not in known, f"Reused archived or holdout initialization on layout {layout}")
            known.add(digest)
        print("RETREAT_HOLDOUT_PREFIX", layout, fixture["qualified"], flush=True)
        if not fixture["qualified"]:
            row["status"] = "common_cheese_prefix_unavailable"
            write(output / "progress.json", {"complete": False, "rows": rows})
            continue
        folder = output / "shadow" / str(layout)
        folder.mkdir(parents=True)
        samples = []
        base_path, result = uninterrupted_arm(api, capturing_worker, ActualProgress, frozen_module(),
              shadow_observer(folder, samples), policy, locate, bddl, case, "adaptive1", fixture_path, stop)
        entry = {"layout_seed": layout, "arm": "adaptive1", "result": base_path.relative_to(output).as_posix(),
                 "result_sha256": sha(base_path)}
        base = load_arm(output, entry)
        row["baseline"] = baseline_summary(output, base_path, result, base)
        trigger = trigger_from_shadow(api, base, samples, folder)
        row.update(first_trigger=trigger, candidate_success_all_unheld=False)
        if stop.exists():
            row["status"] = "baseline_observer_safety_stop"
        elif trigger is None or trigger["step"] > LAST_TRIGGER:
            row["status"] = "same_path_no_trigger" if trigger is None else "same_path_not_enough_budget"
            row["candidate_success_all_unheld"] = row["baseline"]["official_success_all_unheld"]
        else:
            cut = trigger["step"]
            truth = [base["physical"][s] for s in trigger["window_steps"]]
            row["trigger_truth_for_evaluation"] = truth
            if any(any(t["held_by_target"].values()) or t["goals"] != {obj: obj != SOUP for obj in GOALS} for t in truth):
                row["status"] = "unsafe_trigger"
                write(stop, {"layout_seed": layout, "trigger_step": cut, "truth_for_evaluation": truth})
            else:
                initial = read(base_path.parent / "initial-observation.json")
                home = sensor_position(api.Frame.read(checked(base_path.parent, initial["file"], initial["sha256"])))
                fixed = {**case, "first_trigger": trigger, "initial": initial,
                         "initial_eef_m": home, "source_policy_seed": case["policy_seed"]}
                order = ("hold", "retreat") if layout % 2 == 0 else ("retreat", "hold")
                for mode in order:
                    dest = output / "continuations" / str(layout) / mode
                    continue_arm(api, native_worker, policy, bddl, fixed, base, mode, dest)
                    path = dest / "result.json"
                    item = {"result": path.relative_to(output).as_posix(), "result_sha256": sha(path)}
                    row[mode] = check_arm(output, item, {**case, "cutoff": cut}, base, mode)
                    print("RETREAT_HOLDOUT_ARM", layout, mode, row[mode]["terminal_reason"],
                          row[mode]["official_success_all_unheld"], flush=True)
                row["status"] = "paired_trigger_complete"
                row["candidate_success_all_unheld"] = row["retreat"]["official_success_all_unheld"]
        write(output / "progress.json", {"complete": False, "rows": rows})
        print("RETREAT_HOLDOUT_LAYOUT", layout, row["status"], row["baseline"]["official_success_all_unheld"],
              row["candidate_success_all_unheld"], flush=True)
    metrics = calculate(rows)
    paired = [r for r in rows if "retreat" in r]
    reached = all(r["retreat"]["intervention_failure"] is None and not r["retreat"]["completed_goal_reversals"] for r in paired)
    advance = metrics["triggered_pair_layouts"] >= 2 and metrics["candidate_wins"] >= 1 and metrics["candidate_losses"] == 0
    write(output / "summary.json", {"schema": "v2-basket-retreat-holdout/1", "plan_sha256": sha(output / "plan.json"),
          "producer_sha256": sha(Path(__file__)), "rows": rows, "metrics": metrics, "safety_stop": stop.exists(),
          "meets_development_gate": bool(reached and advance and not stop.exists()), "independent_pilot_release": False})


if __name__ == "__main__":
    main()
