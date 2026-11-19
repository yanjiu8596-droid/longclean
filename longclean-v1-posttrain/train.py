"""Train one deterministic ablation of the V1 Scene 08 LoRA study."""

from argparse import ArgumentParser
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np
import torch

from common import (
    CAMERA_KEYS, DEFAULT_LAB, EXECUTE, HORIZON, Lab, METHODS, TRAIN_SEED,
    file_sha, read_json, write_json,
)
from export_dataset import pixel_array, repo_id
from policy import action_window, flow_loss, load_service, masked_mean, observation, parameter_digest


UPDATES = 600
SAVE_AT = (200, 400, 600)
ACCUMULATION = 4
LEARNING_RATE = 1e-4


class Samples:
    def __init__(self, dataset, split, cases):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.reader = LeRobotDataset(
            repo_id(split), root=dataset / split, download_videos=False, return_uint8=True,
        )
        self.cases = cases
        self.clean = self._by_layout(case for case in cases if not case["edits"])
        self.natural = self._by_layout(case for case in cases if not case["edits"] and case["trigger"] is not None)
        self.corrected = self._by_layout(case for case in cases if case["edits"])

    @staticmethod
    def _by_layout(cases):
        groups = defaultdict(list)
        for case in cases:
            groups[case["layout_sha256"]].append(case)
        return list(groups.values())

    @staticmethod
    def _anchor(case):
        return case["edits"][0] if case["edits"] else case["trigger"]

    def draw(self, arm, update_index, micro):
        rng = random.Random(TRAIN_SEED + ACCUMULATION * update_index + micro)
        pool = self.clean if micro < 2 else getattr(self, arm)
        if not pool:
            raise ValueError(f"Training pool for {arm} is empty")
        case = rng.choice(rng.choice(pool))
        if micro < 2:
            offset = rng.randrange(case["steps"])
        else:
            anchor = self._anchor(case)
            offset = rng.randrange(max(0, anchor - EXECUTE), min(anchor, case["steps"]))
        return self.get(case, offset)

    def validation(self):
        for case in self.cases:
            anchor = self._anchor(case)
            offset = max(0, anchor - EXECUTE // 2) if anchor is not None else case["steps"] // 2
            yield case["layout_sha256"], self.get(case, offset)

    def get(self, case, offset):
        first = case["first_index"] + offset
        sample = self.reader[first]
        if sample["episode_index"].item() != case["episode_index"] or sample["task"] != case["task"]:
            raise ValueError("Observation addresses the wrong source episode")
        # Future labels are read from the tabular column so RGB is decoded only at t.
        actions = self._actions(case)
        batch = observation(
            pixel_array(sample[CAMERA_KEYS[0]]), pixel_array(sample[CAMERA_KEYS[1]]),
            sample["observation.state"], case["task"],
        )
        batch["action"], valid = action_window(actions[offset:offset + HORIZON])
        return batch, valid, case["id"], offset

    def _actions(self, case):
        if not hasattr(self, "_tables"):
            self._tables = {}
        if case["id"] not in self._tables:
            first = case["first_index"]
            table = self.reader.hf_dataset.select_columns(["action"])[first:first + case["steps"]]["action"]
            self._tables[case["id"]] = np.stack(table).astype(np.float32)
        return self._tables[case["id"]]


def compute(service, sample):
    raw, valid, _, _ = sample
    policy = service["policy"].get_base_model()
    batch = service["preprocessor"](raw)
    return flow_loss(policy, batch, valid)


def validate(service, samples):
    groups = defaultdict(list)
    service["policy"].eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
        for index, (layout, sample) in enumerate(samples.validation()):
            torch.manual_seed(TRAIN_SEED + 10000 + index)
            value = float(compute(service, sample))
            if not np.isfinite(value):
                raise ValueError("Validation loss is not finite")
            groups[layout].append(value)
    service["policy"].train()
    return float(np.mean([np.mean(values) for values in groups.values()]))


def update(service, optimizer, samples):
    optimizer.zero_grad(set_to_none=True)
    values, frames = [], []
    for sample in samples:
        value = compute(service, sample)
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite flow matching loss")
        (value / ACCUMULATION).backward()
        values.append(float(value.detach()))
        frames.append({"id": sample[2], "offset": sample[3], "valid_steps": int(sample[1].sum())})
    trainable = [parameter for parameter in service["policy"].parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"train_loss": float(np.mean(values)), "grad_norm": float(norm), "frames": frames}


def optimizer_for(policy):
    trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Training has parameters outside the intended LoRA layers")
    optim = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.01,
    )
    return optim, sum(parameter.numel() for _, parameter in trainable)


def save_checkpoint(root, service, optimizer, state):
    attempts = root / "attempts"
    attempts.mkdir(exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=attempts))
    service["policy"].save_pretrained(pending / "adapter", safe_serialization=True)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(0),
    }, pending / "resume.pt")
    state = {**state, "adapter_sha256": file_sha(pending / "adapter/adapter_model.safetensors"),
             "resume_sha256": file_sha(pending / "resume.pt")}
    write_json(pending / "state.json", state)
    final = root / f"step-{state['step']:04d}"
    if final.exists():
        previous = read_json(final / "state.json")
        if any(previous[key] != state[key] for key in ("adapter_sha256", "data_sha256", "step", "arm")):
            raise ValueError(f"A conflicting checkpoint already exists at {final}")
        return previous
    pending.rename(final)
    return state


def restore(root, service, optimizer, previous):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = root / f"step-{previous['step']:04d}"
    adapter = path / "adapter/adapter_model.safetensors"
    statefile = path / "resume.pt"
    if file_sha(adapter) != previous["adapter_sha256"] or file_sha(statefile) != previous["resume_sha256"]:
        raise ValueError("Checkpoint checksum differs")
    set_peft_model_state_dict(service["policy"], load_file(str(adapter)))
    state = torch.load(statefile, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.random.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], 0)


def smoke(lab, run):
    root = run / "train-smoke"
    root.mkdir(exist_ok=True)
    if (root / "passed.json").exists():
        return
    source = run / "replay/init44-seed08"
    case = read_json(source / "verified.json")
    service = load_service(lab, training=True)
    base = service["policy"].get_base_model()
    trainable_before = parameter_digest(service["policy"], trainable=True)
    frozen_before = parameter_digest(service["policy"], trainable=False)
    optim, count = optimizer_for(service["policy"])
    actions = np.load(source / "actions.npy", allow_pickle=False)
    with np.load(source / "dense.npz", allow_pickle=False) as snap:
        front, wrist, states = [snap[key] for key in ("front", "wrist", "state")]

    def sample(t):
        batch = observation(front[t], wrist[t], states[t], case["task"])
        batch["action"], valid = action_window(actions[t:t + HORIZON])
        return batch, valid, case["id"], t

    service["policy"].eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
        raw, valid, _, _ = sample(0)
        batch = service["preprocessor"](raw)
        torch.manual_seed(TRAIN_SEED)
        expected, _ = base(batch)
        torch.manual_seed(TRAIN_SEED)
        masked = flow_loss(base, batch, valid)
        torch.testing.assert_close(masked, expected.float(), rtol=2e-5, atol=1e-6)
        valid_steps = torch.arange(HORIZON) < 1
        artificial = torch.ones(1, HORIZON, 7)
        artificial[:, 1:] = float("inf")
        torch.testing.assert_close(masked_mean(artificial, valid_steps), torch.tensor(1.0))
    service["policy"].train()
    anchor = case["edits"][0] - 5
    samples = [sample(anchor), sample(0), sample(anchor), sample(1)]
    started = perf_counter()
    first = update(service, optim, samples)
    checkpoint = save_checkpoint(root, service, optim, {
        "step": 1, "arm": "smoke", "data_sha256": case["dense_sha256"], "history": [],
    })
    trained = parameter_digest(service["policy"], trainable=True)
    continued = update(service, optim, samples)
    finished = parameter_digest(service["policy"], trainable=True)
    restore(root, service, optim, checkpoint)
    if parameter_digest(service["policy"], trainable=True) != trained:
        raise ValueError("LoRA reloaded values differ")
    repeated = update(service, optim, samples)
    if (not np.isclose(continued["train_loss"], repeated["train_loss"], rtol=1e-5, atol=1e-7)
            or parameter_digest(service["policy"], trainable=True) != finished):
        raise ValueError("Optimizer and RNG resume did not reproduce the next update")
    frozen_after = parameter_digest(service["policy"], trainable=False)
    if trained == trainable_before or frozen_after != frozen_before or first["grad_norm"] <= 0:
        raise ValueError("Expected nonzero adapter gradient and byte-identical frozen weights")
    write_json(root / "passed.json", {
        "parity_flow_loss": float(masked), "first_update": first, "resumed_update": repeated,
        "trainable_parameters": count, "targets": service["targets"],
        "frozen_sha256": frozen_before, "adapter_sha256_before": trainable_before,
        "adapter_sha256_after": finished, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "elapsed_seconds": perf_counter() - started,
    })
    print("TRAIN_SMOKE_OK", count, first["train_loss"], flush=True)


def train(lab, run, arm):
    root = run / "train" / arm
    root.mkdir(parents=True, exist_ok=True)
    status = root / "selected.json"
    data = run / "dataset/build-manifest.json"
    data_sha = file_sha(data)
    if status.is_file():
        if read_json(status)["data_sha256"] != data_sha:
            raise ValueError("Selected adapter uses a different data split")
        return
    if not (run / "train-smoke/passed.json").is_file() or not (run / "test-plan.json").is_file():
        raise ValueError("Finish the model smoke test and freeze independent layouts before training")
    dataset = read_json(data)["splits"]
    samples = {split: Samples(data.parent, split, info["episodes"]) for split, info in dataset.items()}
    service = load_service(lab, training=True)
    optim, count = optimizer_for(service["policy"])
    before = parameter_digest(service["policy"], trainable=False)
    initial = parameter_digest(service["policy"], trainable=True)
    service["policy"].train()
    start, history = 0, []
    checkpoints = sorted(root.glob("step-*/state.json"))
    if checkpoints:
        previous = read_json(checkpoints[-1])
        if (previous["data_sha256"], previous["arm"], previous["frozen_sha256"]) != (data_sha, arm, before):
            raise ValueError("Checkpoint does not match the frozen training inputs")
        restore(root, service, optim, previous)
        start, history = previous["step"], previous["history"]
    started = perf_counter()
    with (root / "updates.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        for step in range(start + 1, UPDATES + 1):
            metrics = update(service, optim, [samples["train"].draw(arm, step - 1, micro)
                                             for micro in range(ACCUMULATION)])
            log.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
            if step % 20 == 0:
                print("UPDATE", arm, step, round(metrics["train_loss"], 6), flush=True)
            if step not in SAVE_AT:
                continue
            loss = validate(service, samples["validation"])
            history.append({"step": step, "validation_loss": loss})
            save_checkpoint(root, service, optim, {
                "step": step, "arm": arm, "data_sha256": data_sha, "history": history,
                "frozen_sha256": before, "validation_loss": loss,
            })
            print("VALIDATION", arm, step, loss, flush=True)
    after = parameter_digest(service["policy"], trainable=False)
    if before != after or initial == parameter_digest(service["policy"], trainable=True):
        raise ValueError("Unexpected base or adapter history")
    best = min(history, key=lambda value: (value["validation_loss"], value["step"]))
    record = read_json(root / f"step-{best['step']:04d}" / "state.json")
    write_json(status, {
        **record, "selected_path": f"step-{best['step']:04d}/adapter",
        "all_validation": history, "updates": UPDATES, "accumulation": ACCUMULATION,
        "learning_rate": LEARNING_RATE, "trainable_parameters": count, "targets": service["targets"],
        "initial_trainable_sha256": initial, "final_frozen_sha256": after,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "session_seconds": perf_counter() - started,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=(*METHODS, "smoke"), required=True)
    args = parser.parse_args()
    torch.cuda.init()
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.reset_peak_memory_stats(0)
    lab = Lab(args.lab)
    if args.arm == "smoke":
        smoke(lab, args.run)
    else:
        train(lab, args.run, args.arm)


if __name__ == "__main__":
    main()
