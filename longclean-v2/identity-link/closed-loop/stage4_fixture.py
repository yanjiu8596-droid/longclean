"""Prespecify, screen and seal the stage 4 paired LIBERO fixtures."""
import argparse
from collections import Counter
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
IDENTITY = HERE.parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(IDENTITY)]
from common import BUDGET, CHUNK, COMMIT, GPU, REPO, read, sha, write


ARMS = ("baseline", "candidate")
TRIAL_COUNTS = {"smoke": 1, "pilot": 6}  # per card
POOLS = {"smoke": (931000, 8), "pilot": (931100, 24)}  # start, count per card
INDEPENDENT_SHA = "d2bea4d08278d50acb670795f03bd91aee5e5fd68b4fb424096d55100d739696"
OBSERVER_FREEZE_SHA = "4b72ee2bed52a346b4329c0577519843fe5891a33e8a276973bedbe415914a7c"
BDDL_SHA = "32f2219ef8e19d4ffe1c68ad17a2493dcf0b8601a6d143f2866bab06aff8a0ac"


def pool(phase, card):
    start, count = POOLS[phase]
    return [start + 2 * i + card - 2 for i in range(count)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_study(study):
    plan, old = read(study / "plan.json"), read(study / "frozen.json")
    bddl = study / plan["bddl"]
    require(old["plan_sha256"] == sha(study / "plan.json")
            and plan["bddl_sha256"] == BDDL_SHA == sha(bddl), "study4 plan or BDDL changed")
    require(plan["v1_commit"] == COMMIT
            and subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == COMMIT,
            "Unexpected LongClean revision")
    require((plan["budget"], plan["chunk"]) == (BUDGET, CHUNK) == (780, 10), "Unexpected control budget")
    for name, digest in old["code"].items():
        require(sha(ROOT / name) == digest, f"study4 source changed: {name}")
    return bddl, old


def validate_gpu(card):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU[card]
            and os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(card)
            and os.environ.get("MUJOCO_GL") == "egl", "EGL and CUDA must pin the same card")


def code_lock(old):
    independent = IDENTITY / "stage3d-independent.json"
    observer_freeze = IDENTITY / "stage3d-freeze.json"
    require(sha(independent) == INDEPENDENT_SHA and sha(observer_freeze) == OBSERVER_FREEZE_SHA,
            "Stage 3d gate evidence changed")
    report, prior = read(independent), read(observer_freeze)
    require(report["gate"] is True and report["freeze_sha256"] == OBSERVER_FREEZE_SHA
            and prior["candidate"] == {"factory": "stage3d_ablation:factory_plane", "cadence": "coarse"},
            "Selected observer did not pass stage 3d")
    files = set(HERE / name for name in ("stage4_fixture.py", "run_closed_loop.py", "progress.py", "audit4.py",
                                        "legacy_parity.py", "stage4_gpu.sh", "stage4_pipeline.sh"))
    for name, digest in prior["source_sha256"].items():
        path = (IDENTITY / name).resolve()
        require(sha(path) == digest, f"Stage 3d source changed: {name}")
        files.add(path)
    files.update(ROOT / name for name in old["code"])
    files.update((independent, observer_freeze))
    return {Path(os.path.relpath(path, HERE)).as_posix(): sha(path) for path in sorted(files)}


def screen(args, bddl):
    from run import Pi05, load_locator, prepare_fixture

    card, phase, output = args.card, args.phase, args.output
    (output / "fixtures").mkdir(parents=True, exist_ok=True)
    marker = output / f"screen-{card}.json"
    known = pool(phase, card)
    info = {"schema": "v2-closed-loop-screen/1", "phase": phase, "card": card,
            "pool": known, "screened": [], "qualified": [], "complete": False}
    if marker.exists():
        old = read(marker)
        require(all(old[key] == info[key] for key in ("schema", "phase", "card", "pool")), "Conflicting output root")
        if old["complete"]:
            print("SCREEN_REUSED", phase, card, old["qualified"], flush=True)
            return
    write(marker, info)
    policy, locate = Pi05(GPU[card]), load_locator()
    for seed in known:
        folder = output / "fixtures" / str(seed)
        record = prepare_fixture(policy, locate, bddl, folder, seed)
        info["screened"].append({"layout_seed": seed, "eligible": record["eligible"], "reason": record.get("reason"),
                                "fixture_sha256": sha(folder / "fixture.json"),
                                "prefix_sha256": sha(folder / "prefix.json")})
        if record["eligible"]:
            info["qualified"].append(seed)
        info["complete"] = len(info["qualified"]) == TRIAL_COUNTS[phase]
        write(marker, info)
        print("FIXTURE", phase, card, seed, record["eligible"], record.get("offset"), record.get("reason"), flush=True)
        if info["complete"]:
            return
    raise RuntimeError(f"Prespecified pool exhausted: {phase} card {card} qualified {len(info['qualified'])}")


def freeze(args, bddl, old):
    output, phase = args.output, args.phase
    require(not (output / "plan.json").exists() and not (output / "frozen.json").exists(), "Freeze already exists")
    code = code_lock(old)
    cases, raw, states = [], set(), set()
    for card in GPU:
        screen_file = output / f"screen-{card}.json"
        s = read(screen_file)
        require(s["complete"] and s["card"] == card and s["phase"] == phase and s["pool"] == pool(phase, card),
                f"Incomplete {phase} screening: card {card}")
        require([r["layout_seed"] for r in s["screened"]] == s["pool"][:len(s["screened"])],
                f"Nonsequential {phase} screening: card {card}")
        qualified = [r for r in s["screened"] if r["eligible"]]
        require(len(qualified) == TRIAL_COUNTS[phase]
                and [r["layout_seed"] for r in qualified] == s["qualified"], "Qualification count changed")
        for slot, row in enumerate(qualified):
            seed = row["layout_seed"]
            folder = output / "fixtures" / str(seed)
            fixture, prefix = read(folder / "fixture.json"), read(folder / "prefix.json")
            require(fixture["layout_seed"] == seed and fixture["eligible"] and fixture["observed_cheese"]
                    and fixture["prefix"] == prefix["end"] and fixture["perturbed_checkpoint"]["bddl_sha256"] == BDDL_SHA
                    and sha(folder / "fixture.json") == row["fixture_sha256"]
                    and sha(folder / "prefix.json") == row["prefix_sha256"], f"Changed fixture: {seed}")
            initial = prefix["initial"]
            require(initial["raw_sha256"] not in raw and initial["state_sha256"] not in states,
                    f"Duplicate simulator initialization: {seed}")
            raw.add(initial["raw_sha256"])
            states.add(initial["state_sha256"])
            moved = (card == 2) if phase == "smoke" else slot < 4
            policy_seed = 46 + (slot + card - 2) % 2
            order = ARMS if (slot + card - 2) % 2 == 0 else ARMS[::-1]
            cases.append({"tag": f"{seed}-s{policy_seed}-{'moved' if moved else 'sham'}",
                          "layout_seed": seed, "policy_seed": policy_seed, "moved": moved, "card": card,
                          "order": list(order), "offset": fixture["offset"] if moved else None,
                          "fixture_sha256": row["fixture_sha256"], "prefix_sha256": row["prefix_sha256"],
                          "raw_sha256": initial["raw_sha256"], "initial_state_sha256": initial["state_sha256"]})
    counts = Counter(r["moved"] for r in cases)
    require(counts == ({True: 1, False: 1} if phase == "smoke" else {True: 8, False: 4}), "Wrong condition split")
    plan_path = output / "plan.json"
    write(plan_path, {"schema": "v2-closed-loop-pair-plan/1", "phase": phase, "v1_commit": COMMIT,
                      "bddl_sha256": sha(bddl), "study4_sha256": sha(args.study / "frozen.json"),
                      "arms": list(ARMS), "budget": BUDGET, "policy_chunk": CHUNK, "observer_stride": 5,
                      "screen_sha256": {str(card): sha(output / f"screen-{card}.json") for card in GPU},
                      "eligibility": "Unchanged study4 prefix and quiet and moved reference qualification; no arm results used.",
                      "cases": cases})
    write(output / "frozen.json", {"schema": "v2-closed-loop-freeze/1", "plan_sha256": sha(plan_path),
                                   "code": code, "cases": cases})
    print("FROZEN", phase, len(cases), 2 * len(cases), sha(plan_path), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("screen", "freeze"))
    p.add_argument("--phase", required=True, choices=TRIAL_COUNTS)
    p.add_argument("--card", type=int, choices=GPU)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.study, args.output = args.study.resolve(), args.output.resolve()
    bddl, old = validate_study(args.study)
    if args.stage == "screen":
        if args.card is None:
            p.error("screen requires --card")
        validate_gpu(args.card)
        screen(args, bddl)
    else:
        freeze(args, bddl, old)


if __name__ == "__main__":
    main()
