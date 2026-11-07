"""Select untouched diagnostic initial states and balanced placement observations."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from probe_native_vision import TARGETS


# Frozen after auditing the initial states in all preceding vision reports.
UNUSED = {
    1: (17, 18, 23, 24, 30, 31, 38, 39, 45, 46, 47, 48, 49),
    2: (23, 31, 36, 38, 44, 47, 48),
    8: (18, 22, 23, 24, 37, 38, 39, 46, 47, 48, 49),
}


def audit(record):
    return {entry["predicate"][1]: entry["satisfied"] for entry in record["audit_goals"]}


def fresh_windows(run: Path, scenes: dict, development_hashes: set[str], count: int) -> dict:
    candidates = defaultdict(dict)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_bytes().splitlines()]
    for row in sorted(rows, key=lambda item: (item["init"], item["seed"], item["run"])):
        if row["phase"] != "direct_horizon_diagnostics" or not row["completed"]:
            continue
        scene, index = row["scene"], row["init"]
        if index not in UNUSED[scene] or row["init_state_sha256"] in development_hashes:
            continue
        name = row.get("artifact", row["run"])
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid artifact directory {name}")
        folder = run / name
        final = json.loads((folder / "episode.json").read_bytes())
        actions = [json.loads(line) for line in (folder / "actions.jsonl").read_bytes().splitlines()]
        timeline = [(action["start_step"] + action["executed"], audit(action)) for action in actions]
        for label in scenes[scene]:
            target = TARGETS[label]
            terminal = audit(final)
            possibilities = [(terminal[target], 0, final["steps"])]
            first_placement = next((step for step, truth in timeline if truth[target]), None)
            for step, truth in timeline:
                if step % 10 or step < 60 or truth[target]:
                    continue
                another_placed = any(value for key, value in truth.items() if key != target)
                near_first = first_placement is not None and 0 < first_placement - step <= 30
                if another_placed or near_first:
                    possibilities.append((False, 1 if another_placed else 2, step))
            for truth, priority, step in possibilities:
                key = scene, label, truth
                rank = (priority, row["seed"], row["run"], -step)
                prior = candidates[key].get(index)
                if prior is None or rank < prior[0]:
                    candidates[key][index] = (rank, {**row, "decision_step": step}, folder)

    selected = {}
    for scene, labels in scenes.items():
        for label in labels:
            for truth in (False, True):
                key = scene, label, truth
                options = sorted(candidates[key].items(), key=lambda item: (item[1][0][0], item[0], item[1][0]))
                group = {}
                hashes = set()
                for index, (_, row, folder) in options:
                    initial_hash = row["init_state_sha256"]
                    if initial_hash in hashes:
                        continue
                    hashes.add(initial_hash)
                    group[index] = row, folder
                    if len(group) == count:
                        break
                if len(group) != count:
                    raise ValueError(f"insufficient fresh observations for {key}: {len(group)}")
                selected[key] = group
    return selected
