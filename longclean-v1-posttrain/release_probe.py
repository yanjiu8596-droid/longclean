"""Screen native Scene 08 layouts with a disjoint policy seed, then score frozen pairs."""

from argparse import ArgumentParser
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import random
from shutil import copytree
import tempfile

from common import ARMS, Lab, file_sha, read_json, write_json
from score import audit_episode, paired, per_arm


PROTOCOL = {
    "schema": "v1-scene08-release-probe/1",
    "layout_seed_start": 864200,
    "first_batch": 96,
    "maximum_layouts": 160,
    "probe_seeds": [9, 11],
    "formal_seeds": [71, 72],
    "hold_frames": 3,
    "release_frames": 5,
    "wait_steps": 100,
    "screen_horizon": 260,
    "minimum_stalled_layouts": 6,
    "maximum_stalled_layouts": 12,
    "released_controls": 8,
    "control_order_seed": 20261120,
    "historical_check": {"init": 44, "seeds": [7, 9, 11]},
    "selected_training_steps": {"natural": 600, "corrected": 400},
}


def first_run(rows, predicate, consecutive):
    for active, frames in groupby(rows, key=predicate):
        if active:
            for count, row in enumerate(frames, 1):
                if count == consecutive:
                    return row["step"]
    return None


def screen(rows, plan):
    sub = rows[:plan["screen_horizon"] + 1]
    held = first_run(sub, lambda row: row["soup_held"], plan["hold_frames"])
    if held is None:
        return {"label": "no_stable_hold", "hold_step": None, "check_step": None, "release_step": None}
    check = held + plan["wait_steps"]
    release = first_run(sub[held:check + 1], lambda row: row["soup_in"] and not row["soup_held"],
                        plan["release_frames"])
    if release is not None:
        label = "released_in_window"
    elif check >= len(sub):
        label = "short_observation"
    elif all(row["soup_held"] for row in sub[check + 1 - plan["hold_frames"]:check + 1]):
        label = "still_held_without_release"
    else:
        label = "grasp_lost_without_release"
    return {"label": label, "hold_step": held, "check_step": check, "release_step": release}


def prior_identities(lab, primary, development):
    from evaluate import historical

    raw, settled = historical(lab, primary)
    for path in (primary / "test-plan.json", development / "test-plan.json"):
        for row in read_json(path)["cases"]:
            raw.add(row["initial"]["raw_sha256"])
            settled.add(row["initial"]["settled_sha256"])
    raw.update(row["raw_sha256"] for row in read_json(development / "geometry.json")["candidates"])
    return raw, settled


def probe(lab, primary, development, run, uuid, plan, plan_sha):
    from evaluate import rollout
    from policy import load_service

    known_raw, known_settled = prior_identities(lab, primary, development)
    outcomes = []
    service = None

    def run_case(cohort, layout, seed, fixed=None):
        nonlocal service
        target = run / "eval" / cohort / str(layout) / str(seed) / "base"
        marker = target / "result.json"
        if (target / "rejected.json").exists():
            return read_json(target / "rejected.json")
        if not marker.exists():
            if service is None:
                print("LOADING_BASE_POLICY", flush=True)
                service = load_service(lab)
            pending = Path(tempfile.mkdtemp(prefix="probe-", dir=run / "tmp"))
            try:
                result = rollout(lab, service, pending, layout=layout, seed=seed, gpu=uuid, fixed=fixed)
            except RuntimeError:
                log = (pending / "simulator.log").read_text(encoding="utf-8", errors="replace")
                if not any(message in log for message in ("sampled reset collides", "already satisfies a placement goal")):
                    raise
                write_json(pending / "rejected.json", {"rejected": "native_reset_rejected", "layout_seed": layout,
                                                       "probe_seed": seed})
            else:
                result.update(cohort=cohort, arm="base", test_plan_sha256=plan_sha, adapter_sha256=None)
                write_json(pending / "result.json", result)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(target)
            if (target / "rejected.json").exists():
                return read_json(target / "rejected.json")
        result = read_json(marker)
        case = {"cohort": cohort, "layout_seed": layout, "initial": result["initial"]}
        result = audit_episode(run, case, seed, "base", plan_sha, uuid, {})
        rows = [json.loads(line) for line in (target / "audit.jsonl").read_bytes().splitlines()]
        return {"layout_seed": layout, "probe_seed": seed, "initial": result["initial"],
                "screen": screen(rows, plan), "probe_official_success": result["success"],
                "probe_steps": result["steps"], "actions_sha256": result["actions_sha256"],
                "audit_sha256": result["audit_sha256"]}

    fixed = plan["historical_check"]
    anchors = [run_case("historical", fixed["init"], seed, fixed=fixed["init"]) for seed in fixed["seeds"]]
    print("ANCHOR_DONE", [(row["probe_seed"], row["screen"]["label"], row["probe_official_success"])
                          for row in anchors], flush=True)
    first = plan["layout_seed_start"]
    stalled = []
    for offset, layout in enumerate(range(first, first + plan["maximum_layouts"])):
        seed = plan["probe_seeds"][offset % len(plan["probe_seeds"])]
        row = run_case("probe", layout, seed)
        if "initial" in row:
            raw = row["initial"]["raw_sha256"]
            settled = row["initial"]["settled_sha256"]
            if raw in known_raw or settled in known_settled:
                row["rejected"] = "duplicate_raw_or_settled_state"
            known_raw.add(raw)
            known_settled.add(settled)
            if "rejected" not in row and row["screen"]["label"] == "still_held_without_release":
                stalled.append(row)
        outcomes.append(row)
        label = row["rejected"] if "rejected" in row else row["screen"]["label"]
        print("PROBE", offset + 1, layout, seed, label, "stalled", len(stalled), flush=True)
        if len(outcomes) >= plan["first_batch"] and len(stalled) >= plan["maximum_stalled_layouts"]:
            break

    # Order the prompt-release controls using randomness fixed in the protocol.
    control = [row for row in outcomes if "rejected" not in row and row["screen"]["label"] == "released_in_window"]
    random.Random(plan["control_order_seed"]).shuffle(control)
    ready = len(stalled) >= plan["minimum_stalled_layouts"] and len(control) >= plan["released_controls"]
    report = {"protocol_sha256": plan_sha, "historical_checks": anchors, "layouts_attempted": len(outcomes),
              "counts": dict(Counter(row["rejected"] if "rejected" in row else row["screen"]["label"]
                                     for row in outcomes)), "eligible_for_formal": ready, "probes": outcomes}
    write_json(run / "probe-summary.json", report)
    print("PROBES_COMPLETE", len(outcomes), report["counts"], "eligible_for_formal", ready, flush=True)
    if not ready:
        return

    formal = run / "formal"
    cases = []
    for name, group in (("probe_stall", stalled[:plan["maximum_stalled_layouts"]]),
                        ("prompt_release", control[:plan["released_controls"]])):
        for index, row in enumerate(group):
            cases.append({"cohort": name, "layout_seed": row["layout_seed"], "initial": row["initial"],
                          "probe_seed": row["probe_seed"], "inference_seeds": plan["formal_seeds"],
                          "representative_video": index == 0})
    target = formal / "test-plan.json"
    assignment = {"schema": "v1-scene08-release-probe-pairs/1", "gpu_uuid": uuid, "cases": cases,
                  "protocol_sha256": plan_sha, "probe_summary_sha256": file_sha(run / "probe-summary.json")}
    if target.exists() and read_json(target) != assignment:
        raise ValueError("The saved matched comparison plan differs")
    for arm, expected_step in plan["selected_training_steps"].items():
        parent = primary / "train" / arm
        selected = read_json(parent / "selected.json")
        rel = Path("train") / arm / selected["selected_path"]
        if selected["step"] != expected_step or file_sha(primary / rel / "adapter_model.safetensors") != selected["adapter_sha256"]:
            raise ValueError(f"Unexpected frozen adapter for {arm}")
        target_adapter = formal / rel
        if not target_adapter.exists():
            copytree(primary / rel, target_adapter)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            if file_sha(target_adapter / name) != file_sha(primary / rel / name):
                raise ValueError(f"Copied adapter differs for {arm}")
        write_json(formal / "train" / arm / "selected.json", selected)
    write_json(target, assignment)
    print("FORMAL_PLAN_FROZEN", len(cases), len(cases) * len(ARMS) * len(plan["formal_seeds"]),
          file_sha(target), flush=True)


def formal_summary(run):
    formal = run / "formal"
    path = formal / "test-plan.json"
    plan = read_json(path)
    proto = read_json(run / "protocol.json")
    if (plan["protocol_sha256"] != file_sha(run / "protocol.json")
            or plan["probe_summary_sha256"] != file_sha(run / "probe-summary.json")):
        raise ValueError("Comparison lost its frozen screening provenance")
    cases = plan["cases"]
    for key in ("raw_sha256", "settled_sha256"):
        if len({case["initial"][key] for case in cases}) != len(cases):
            raise ValueError("Duplicate matched layout")
    selected = {arm: read_json(formal / "train" / arm / "selected.json") for arm in ARMS[2:]}
    if {arm: row["step"] for arm, row in selected.items()} != proto["selected_training_steps"]:
        raise ValueError("Comparison used an unexpected checkpoint")
    adapters = {arm: row["adapter_sha256"] for arm, row in selected.items()}
    plan_sha = file_sha(path)
    results = {}
    for case in cases:
        if case["probe_seed"] in case["inference_seeds"] or case["inference_seeds"] != proto["formal_seeds"]:
            raise ValueError("Comparison reused the selection policy seed")
    pairs = [{arm: audit_episode(formal, case, seed, arm, plan_sha, plan["gpu_uuid"], adapters) for arm in ARMS}
             for case in cases for seed in case["inference_seeds"]]
    for group in ("all", "probe_stall", "prompt_release"):
        subset = [pair for pair in pairs if group == "all" or pair["base"]["cohort"] == group]
        results[group] = {"arms": {arm: per_arm([pair[arm] for pair in subset]) for arm in ARMS},
                          "contrasts": {f"corrected_vs_{arm}": {metric: paired(subset, "corrected", arm, metric)
                                                               for metric in ("success", "release")}
                                        for arm in ("base", "rule", "natural")}}
    hard = results["probe_stall"]["contrasts"]["corrected_vs_base"]["success"]
    ordinary = results["prompt_release"]["contrasts"]["corrected_vs_base"]["success"]
    return {"episodes": len(pairs) * len(ARMS), "layouts": len(cases), "test_plan_sha256": plan_sha,
            "met_success_target": len(hard["positive_layouts"]) >= 2 and hard["net"] > 0 and ordinary["losses"] == 0,
            "groups": results, "pairs": pairs}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-protocol", "probe", "score"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--lab", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    run = args.run.resolve()
    path = run / "protocol.json"
    if args.stage == "score":
        result = formal_summary(run)
        write_json(run / "formal" / "summary.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2), flush=True)
        return
    if not all((args.lab, args.primary, args.development, args.gpu_uuid)):
        parser.error("Supply the lab, primary and development runs, and authorized GPU UUID")
    lab = Lab(args.lab.resolve())
    primary, development = args.primary.resolve(), args.development.resolve()
    if not all(item.is_relative_to(lab.root) for item in (run, primary, development)) or run in (primary, development):
        parser.error("Use distinct output and reference paths on the personal lab disk")
    plan = {**PROTOCOL, "gpu_uuid": args.gpu_uuid,
            "prior_plan_sha256": file_sha(primary / "test-plan.json"),
            "prior_replay_sha256": file_sha(primary / "replay-manifest.json"),
            "development_plan_sha256": file_sha(development / "test-plan.json"),
            "development_geometry_sha256": file_sha(development / "geometry.json")}
    if path.exists() and read_json(path) != plan:
        raise ValueError("The existing probe protocol differs")
    if args.stage == "freeze-protocol":
        write_json(path, plan)
        print("PROTOCOL_FROZEN", file_sha(path), flush=True)
        return
    if not path.exists():
        raise ValueError("Freeze protocol before running a probe")
    lab.add_imports()
    probe(lab, primary, development, run, args.gpu_uuid, plan, file_sha(path))


if __name__ == "__main__":
    main()
