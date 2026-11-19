"""Recover audited per-step demonstrations from the V1 release study."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np

from common import (
    ARCHIVE_SHA256, BUDGET, DEFAULT_LAB, FPS, Lab, REPLAY_FIELDS, RULE_SHA256, TASK_ID,
    STATE_ENCODING, VAL_LAYOUTS, file_sha, read_json, state_vector, write_json,
)
from simulator import Simulator


def read_archive(path):
    if file_sha(path) != ARCHIVE_SHA256:
        raise ValueError("Unexpected archived V1 release evidence")
    output = []
    with ZipFile(path) as source:
        summary = json.loads(source.read("summary.json"))
        if len(summary["pairs"]) != 120 or sha256(source.read("observe.py")).hexdigest() != RULE_SHA256:
            raise ValueError("Expected the 120-pair V1 archive and its frozen observer")
        for case in sorted(summary["pairs"], key=lambda item: (item["init"], item["seed"])):
            init, seed = case["init"], case["seed"]
            name = f"init{init:02d}-seed{seed:02d}"
            prefix = f"live/{name}-observed"
            episode_bytes = source.read(prefix + "/episode.json")
            action_bytes = source.read(prefix + "/actions.jsonl")
            episode = json.loads(episode_bytes)
            trace = [json.loads(line) for line in action_bytes.splitlines()]
            decision = json.loads(source.read(prefix + "/result.json"))
            executed = [action for row in trace for action in row["actions"]]
            changes = [record["step"] for record in decision["overrides"]]
            if (episode["task_id"], episode["control_hz"], episode["step_budget"]) != (TASK_ID, FPS, BUDGET):
                raise ValueError(f"Incompatible action contract in {name}")
            if (episode["init_state_index"], episode["seed"]) != (init, seed):
                raise ValueError(f"Inconsistent identity in {name}")
            if (not episode["success"] or not trace[-1]["success"]
                    or len(executed) != episode["steps"] or len(changes) != case["override_steps"]
                    or any(not 1 <= step <= len(executed) or executed[step - 1][-1] != -1 for step in changes)):
                raise ValueError(f"Inconsistent successful demonstration in {name}")
            trigger = case["trigger_step"]
            if (changes != sorted(set(changes))
                    or (trigger is not None and not (1 <= trigger <= len(executed) and executed[trigger - 1][-1] <= -0.5))
                    or (changes and (trigger is None or changes[0] <= trigger or changes[-1] >= trigger + 15))):
                raise ValueError(f"Invalid release supervision event in {name}")
            info = {
                "id": name,
                "init": init,
                "seed": seed,
                "layout_sha256": episode["init_state_sha256"],
                "initial_observation_sha256": episode["observation_sha256"],
                "settled_sha256": episode["sim_state_sha256"],
                "split": "validation" if init in VAL_LAYOUTS else "train",
                "task": episode["instruction"],
                "steps": episode["steps"],
                "trigger": trigger,
                "edits": changes,
                "episode_sha256": sha256(episode_bytes).hexdigest(),
                "actions_sha256": sha256(action_bytes).hexdigest(),
                "archive_path": prefix,
                "state_encoding": STATE_ENCODING,
            }
            output.append((info, episode, trace, np.asarray(executed, dtype=np.float32)))
    if len({item[0]["id"] for item in output}) != 120:
        raise ValueError("Archive contains duplicate episodes")
    train = {info["layout_sha256"] for info, *_ in output if info["split"] == "train"}
    val = {info["layout_sha256"] for info, *_ in output if info["split"] == "validation"}
    if train & val or (len(train), len(val)) != (12, 3):
        raise ValueError("Historical split is not disjoint by initial state")
    return output


def check(expected, actual, fields, context):
    differences = [field for field in fields if actual.get(field) != expected.get(field)]
    if differences:
        raise ValueError(f"{context} differs in {', '.join(differences)}")


def replay(lab, root, source):
    info, episode, trace, executed = source
    final = root / info["id"]
    if final.is_dir():
        previous = read_json(final / "verified.json")
        hashes = {"dense.npz": "dense_sha256", "actions.npy": "native_actions_sha256",
                  "audit.jsonl": "audit_sha256", "boundaries.jsonl": "boundaries_sha256"}
        if (all(previous.get(key) == value for key, value in info.items())
                and all(file_sha(final / name) == previous[key] for name, key in hashes.items())):
            return previous
        raise ValueError(f"Existing replay no longer matches its manifest: {final}")

    attempts = root.parent / "replay-attempts"
    attempts.mkdir(exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=info["id"] + "-", dir=attempts))
    boundaries = pending / "boundaries.jsonl"
    with Simulator(lab, pending, info["seed"], init=info["init"], dense=True) as sim, boundaries.open("w") as log:
        log.write(json.dumps({"step": 0, **sim.initial}, allow_nan=False) + "\n")
        check(episode, sim.initial, (*REPLAY_FIELDS, "init_state_sha256", "bddl_sha256", "control_hz"), info["id"])
        with np.load(pending / "observation.npz", allow_pickle=False) as raw:
            initial_state = state_vector(raw["eef_pos"], raw["eef_quat"], raw["gripper_qpos"])
        if not np.array_equal(initial_state, np.asarray(episode["initial_observation_state"], dtype=np.float32)):
            raise ValueError(f"{info['id']} initial model state differs")
        cursor = 0
        for row in trace:
            if row["start_step"] != cursor or row["executed"] != len(row["actions"]):
                raise ValueError(f"{info['id']} non-contiguous execution trace")
            cursor += row["executed"]
            boundary = sim.step(row["actions"])
            log.write(json.dumps({"step": cursor, **boundary}, allow_nan=False) + "\n")
            check(row, boundary, (*REPLAY_FIELDS, "executed", "success", "done"),
                  f"{info['id']} control step {cursor}")

    with np.load(pending / "dense.npz", allow_pickle=False) as arrays:
        lengths = [len(arrays[key]) for key in ("front", "wrist", "state")]
        if (lengths != [info["steps"] + 1] * 3 or arrays["state"].shape[1:] != (8,)
                or arrays["front"].shape[1:] != (256, 256, 3)
                or arrays["wrist"].shape[1:] != (256, 256, 3)
                or not np.array_equal(arrays["state"][0], initial_state)):
            raise ValueError(f"{info['id']} per-step frame contract differs")
    np.save(pending / "actions.npy", executed, allow_pickle=False)
    record = {**info, "boundaries_verified": len(trace) + 1,
              "dense_sha256": file_sha(pending / "dense.npz"),
              "native_actions_sha256": file_sha(pending / "actions.npy"),
              "audit_sha256": file_sha(pending / "audit.jsonl"), "boundaries_sha256": file_sha(boundaries)}
    write_json(pending / "verified.json", record)
    pending.rename(final)
    return record


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--case", help="Replay one ID for the smoke test")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--inspect", action="store_true", help="Print the split without launching a simulator")
    args = parser.parse_args()
    lab = Lab(args.lab)
    sources = read_archive(args.archive or lab.archive)
    if args.inspect:
        print(json.dumps({split: {
            "episodes": sum(info["split"] == split for info, *_ in sources),
            "edited_episodes": sum(info["split"] == split and bool(info["edits"]) for info, *_ in sources),
        } for split in ("train", "validation")}))
        return
    root = args.run / "replay"
    root.mkdir(parents=True, exist_ok=True)
    selected = [item for item in sources if args.case is None or item[0]["id"] == args.case]
    if not selected:
        raise ValueError(f"Unknown archive case: {args.case}")
    records, rejected = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        futures = [(source[0], workers.submit(replay, lab, root, source)) for source in selected]
        try:
            for info, result in futures:
                try:
                    record = result.result()
                except (ValueError, RuntimeError, OSError) as error:
                    rejected.append({"id": info["id"], "reason": str(error)})
                    print("REPLAY_REJECT", info["id"], str(error), flush=True)
                else:
                    records.append(record)
                    print("REPLAY_OK", info["id"], record["steps"], record["boundaries_verified"], flush=True)
        except KeyboardInterrupt:
            for _, future in futures:
                future.cancel()
            raise
    name = f"prepare-{args.case}.json" if args.case else "replay-manifest.json"
    write_json(args.run / name, {"archive_sha256": ARCHIVE_SHA256, "episodes": records, "rejected": rejected})
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
