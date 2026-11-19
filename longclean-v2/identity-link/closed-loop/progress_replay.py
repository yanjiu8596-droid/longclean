"""CPU-only regression of the Stage 4 wiring on one sealed Stage 3d positive."""
import argparse
from pathlib import Path

import numpy as np

from progress import ClosedLoopProgress
from common import CHUNK, SOUP, read, sha, write
from stage3d_ablation import factory_plane
from stage3d_gate import exact_file, load_frame, pin_inputs, replay_rows, require
from vision import Frame, Schedule


LAYOUT = 930100


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", type=Path, required=True)
    cli.add_argument("--score", type=Path, required=True)
    cli.add_argument("--score-sha", required=True)
    cli.add_argument("--smoke", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    require(not args.out.exists(), "Refusing to overwrite regression evidence")
    exact_file(args.score, args.score_sha)
    freeze, _, episodes = pin_inputs(args.freeze.resolve())
    require(freeze["candidate"] == {"factory": "stage3d_ablation:factory_plane", "cadence": "coarse"},
            "Unexpected frozen observer")
    score = read(args.score)
    require(score["freeze_sha256"] == sha(args.freeze), "Score and freeze disagree")
    expected, = [r["scores"]["candidate"]["commits"] for r in score["results"] if r["layout"] == LAYOUT]
    require(expected == [{"step": 140, "source": "identity_lid"}], "Unexpected sealed positive")
    kind, stride, directory, fixture, _ = next(r for r in episodes if r[3]["layout"] == LAYOUT)
    require(kind == "natural" and stride == 5, "Unexpected input cadence")
    smoke = read(args.smoke)
    moved, = [r["arms"]["baseline"] for r in smoke["cases"] if r["condition"] == "moved"]
    fork = moved["steps"] - moved["after_fork_steps"]
    require(fork > 0 and fork % CHUNK == 0, "Expected an offset on the planner grid")

    raw = factory_plane()
    progress = {"shadow": ClosedLoopProgress(write_soup=False), "active": ClosedLoopProgress(write_soup=True)}
    schedule = {k: Schedule(p, False, None) for k, p in progress.items()}
    records, writes, changes, direct = {k: [] for k in progress}, [], [], []
    calls = 0
    for step, path, realized, row, audit, _ in replay_rows(directory, fixture):
        frame = load_frame(path, row["observation_sha256"], np, Frame)
        absolute = fork + step
        observed = [e for e in raw.observe(frame, realized, step) if e["event"] == "in_basket"]
        direct.extend({k: e[k] for k in ("step", "source")} for e in observed)
        if step == 0:
            for p in progress.values():
                p.start_soup(frame, absolute)
        else:
            for arm, p in progress.items():
                commits = [e for e in p.observe_soup(frame, realized, absolute) if e["event"] == "in_basket"]
                records[arm].extend({"step": e["step"] - fork, "source": e["source"]} for e in commits)
                for e in commits:
                    require(e["step"] == absolute and e["applied"] is (arm == "active"), "Wrong writer or clock")
                    if e["applied"]:
                        require(audit["goals"][SOUP] and not audit["held_soup"]
                                and not all(audit["goals"].values()), "Expected a correct nonterminal write")
                        writes.append(e["step"])
        if absolute % CHUNK == 0:
            picked = {arm: s.next(absolute) for arm, s in schedule.items()}
            if SOUP in schedule["shadow"].order and SOUP not in schedule["active"].order:
                changes.append({"step": absolute, "picked": picked,
                                "eligible": {arm: list(s.order) for arm, s in schedule.items()}})
        calls += 1

    require(records["shadow"] == records["active"] == direct == expected, "Adapted and direct observations disagree")
    require(writes == [fork + expected[0]["step"]] and changes and changes[0]["step"] == writes[0],
            "Correct write did not reach the first eligible decision boundary")
    write(args.out, {"schema": "v2-stage4-cpu-wiring-regression/1", "layout": LAYOUT, "calls": calls,
                     "freeze_sha256": sha(args.freeze), "score_sha256": args.score_sha,
                     "smoke_audit_sha256": sha(args.smoke), "replay_source_sha256": sha(Path(__file__)),
                     "fixture_sha256": sha(directory / "fixture.json"),
                     "actions_sha256": sha(directory / "trace" / "actions.jsonl"), "fork_step": fork,
                     "original_commits": direct, "adapted_commits": records, "active_writes": writes,
                     "first_changed_eligibility": changes[0], "stationary_replay_only": True})
    print(f"CPU_WIRING_PASS {LAYOUT} {calls} {writes[0]}", flush=True)


if __name__ == "__main__":
    main()
