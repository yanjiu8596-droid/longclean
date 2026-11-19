"""Stage-4 soup-first mechanism diagnostic on the two sealed smoke forks only."""
import argparse
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

from stage4_fixture import GPU, read, require, sha, validate_gpu, validate_study, write
from common import BUDGET, CHEESE, CHUNK, SAUCE, SOUP
from legacy_parity import cases, trace
from progress import SOUP_WRITER, STRIDE, absolute_event
from progress_exercise import begin, replay_arm
from stage3d_ablation import factory_plane


SUFFIX_BUDGET = 360
SEALED = {2: (931000, 46, True), 3: (931005, 47, False)}
RELEASE_EVENTS = {"identity_pending", "identity_release_rejected"}


def generate(api, policy, locate, bddl, case, prefix, fixture, dest):
    """Produce the one candidate-independent π₀.₅ action stream and sensor audit."""
    task = api.instruction(SOUP)
    observer = factory_plane()
    sensor, model = [], []
    with api.Worker(bddl, dest, case["layout_seed"]) as worker:
        fork = begin(api, worker, prefix, fixture, case, api.Progress(locate), dest)
        start = fork["step"]
        limit = start + SUFFIX_BUDGET
        require(limit <= BUDGET, "Sealed fork does not leave the prescribed suffix budget")
        observer.observe(replace(api.Frame.read(worker.npz), source=None), [], 0)
        policy.reset(case["policy_seed"])
        while worker.current["step"] < limit and not worker.terminal():
            step = worker.current["step"]
            require(step % CHUNK == 0, "Unexpected π decision boundary")
            commands = policy.act(worker, task, limit=CHUNK)
            require(len(commands) == CHUNK, f"Unequal action block at {step}")
            samples = []
            for k in range(0, CHUNK, STRIDE):
                was_held = worker.current["audit"]["held_soup"]
                actual = worker.step(commands[k:k + STRIDE], SOUP, task)
                current = worker.current
                key = current["step"]
                frame = api.Frame.read(worker.npz)
                core = observer.core
                last_gap = core.grasp_gap
                recent_steps = [s.step + start for s in core.transport]
                returned = observer.observe(replace(frame, source=None), actual, key - start)
                observed = [absolute_event(e, start) for e in returned]
                opening = any(a[-1] < -.5 for a in actual)
                released = was_held and not current["audit"]["held_soup"]
                row = {"step": key, "gap_m": frame.gap, "prior_grasp_gap_m": last_gap,
                       "delta_from_prior_grasp_m": frame.gap - last_gap if last_gap is not None else None,
                       "prior_identity_steps": recent_steps,
                       "identity_age_steps": key - recent_steps[-1] if recent_steps else None,
                       "gripper_command_open": opening, "held_soup": current["audit"]["held_soup"],
                       "official": current["audit"]["goals"], "events": observed}
                if released or any(e["event"] in RELEASE_EVENTS for e in observed):
                    evidence = f"release-{key}.npz"
                    copyfile(worker.npz, dest / evidence)
                    row.update(evidence_file=evidence, evidence_sha256=sha(dest / evidence))
                sensor.append(row)
                samples.append(key)
                if worker.terminal():
                    break
            model.append({"start_step": step, "sensor_steps": samples, "requested": CHUNK,
                          "end_step": worker.current["step"]})
        end = api.fingerprint(worker.current)
        final = worker.current["audit"]
        done = worker.terminal()
        require(done or end["step"] == limit, "Unexpected short soup-only diagnostic")
    rows = [r for r in trace(dest / "actions.jsonl") if r["start_step"] >= start]
    expected = {r["step"]: r for r in rows}
    commands = [a for r in rows for a in r["actions"]]
    require(len(commands) == end["step"] - start and commands and len(commands) <= SUFFIX_BUDGET,
            "Invalid saved diagnostic actions")
    known = {SOUP, SAUCE, CHEESE}
    first_in = {name: min((tr["step"] for r in rows for tr in r["transitions"]
                           if tr["object"] == name and tr["in_basket"]), default=None) for name in known}
    source = {"fork": fork, "terminal": end, "terminal_audit": final,
              "first_official_enter_steps": first_in, "terminal_before_budget": done and end["step"] < limit,
              "policy_seed": case["policy_seed"], "prompt": task, "suffix_budget": SUFFIX_BUDGET,
              "action_steps": len(commands), "model_calls": len(model), "model_chunks": model,
              "gpu_placement": policy.placement, "actions_sha256": sha(dest / "actions.jsonl"),
              "fork_npz_sha256": sha(dest / "fork.npz"), "frames": sensor}
    write(dest / "source.json", source)
    return fork, commands, expected, source


def stage3d_events(arm):
    extra = {"writer", "applied", "held_soup", "matches_audit", "confirmed_in_audit",
             "evidence_file", "evidence_sha256"}
    return [{k: v for k, v in e.items() if k not in extra}
            for e in arm["events"] if e["writer"] == SOUP_WRITER]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", required=True, type=int, choices=GPU)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--smoke", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    smoke, out = a.smoke.resolve(), a.output.resolve()
    require(out != smoke and not out.is_relative_to(smoke), "Use a separate diagnostic output directory")
    validate_gpu(a.card)
    bddl, _ = validate_study(a.study.resolve())
    assigned = [r for r in cases(smoke) if r["card"] == a.card]
    require(len(assigned) == 1, "Expected a single sealed layout on this card")
    case, = assigned
    require((case["layout_seed"], case["policy_seed"], case["moved"]) == SEALED[a.card],
            "Unexpected smoke layout, seed, or perturbation")
    f = smoke / "fixtures" / str(case["layout_seed"])
    require(sha(f / "prefix.json") == case["prefix_sha256"] and sha(f / "fixture.json") == case["fixture_sha256"],
            "The shared smoke fork files changed")
    marker = out / f"progress-policy-{a.card}.json"
    dest = out / case["tag"]
    require(not marker.exists(), "Refusing to replace the one fixed diagnostic attempt")
    dest.mkdir(parents=True, exist_ok=False)
    import run as api

    prefix, fixture = read(f / "prefix.json"), read(f / "fixture.json")
    policy, locate = api.Pi05(GPU[a.card]), api.load_locator()
    fork, commands, observed, source = generate(api, policy, locate, bddl, case, prefix, fixture, dest / "source")
    arms = {arm: replay_arm(api, bddl, case, prefix, fixture, dest / arm, locate, arm, fork, commands, observed)
            for arm in case["order"]}
    emitted = [e for row in source["frames"] for e in row["events"]]
    require(all(stage3d_events(r) == emitted for r in arms.values()), "Plane observer output differed on identical inputs")
    wrong = sum(e["applied"] and not e["confirmed_in_audit"] for r in arms.values() for e in r["events"])
    wrong_candidate = sum(e["writer"] == SOUP_WRITER and e["event"] == "in_basket" and not e["confirmed_in_audit"]
                          for r in arms.values() for e in r["events"])
    valid = []
    for hit in arms["candidate"]["eligibility_changes"]:
        event, boundary = hit["event"], hit["boundary_after"]
        at = observed[event["step"]]["audit"]
        after = observed[boundary["step"]]["audit"]
        if (event["confirmed_in_audit"] and boundary["target"] == SAUCE and not boundary["official_success"]
                and at["goals"][CHEESE] and not at["goals"][SAUCE] and after["goals"][SOUP]
                and after["goals"][CHEESE] and not after["goals"][SAUCE] and not after["held_soup"]
                and (source["first_official_enter_steps"][SAUCE] is None
                     or source["first_official_enter_steps"][SAUCE] > boundary["step"])):
            valid.append({**hit, "audit_at_write": at, "audit_at_boundary": after})
    report = {"schema": "v2-stage4-progress-policy/1", "kind": "soup_first_policy_mechanism_smoke",
              "card": a.card, "layout_seed": case["layout_seed"], "policy_seed": case["policy_seed"],
              "moved": case["moved"], "freeze_sha256": sha(smoke / "frozen.json"),
              "source_sha256": sha(Path(__file__)), "record_dir": dest.relative_to(out).as_posix(),
              "source": source, "arms": arms, "valid_write": bool(valid), "valid_writes": valid,
              "wrong_writes": wrong, "wrong_candidate_observations": wrong_candidate}
    write(marker, report)
    print("PROGRESS_POLICY", case["layout_seed"], case["moved"], bool(valid), wrong, wrong_candidate, flush=True)


if __name__ == "__main__":
    main()
