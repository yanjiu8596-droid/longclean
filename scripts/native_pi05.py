"""Offline preflight and CUDA loading for the pinned LIBERO Pi0.5 checkpoint."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parent.parent
with (PROJECT / "versions.lock").open("rb") as stream:
    PIN = tomllib.load(stream)["models"]["pi05_libero"]

CAMERA_RENAME = {
    "observation.images.camera1": "observation.images.image",
    "observation.images.camera2": "observation.images.image2",
}


def model_directory(workspace: Path) -> Path:
    return workspace / "models" / PIN["repository"].split("/")[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def weights_path(model: Path, *, verify: bool = False) -> Path:
    path = model / "model.safetensors"
    if not path.is_file() or path.stat().st_size != PIN["weights_bytes"]:
        raise ValueError(f"Pi0.5 checkpoint is incomplete: {path}")
    if verify and sha256(path) != PIN["weights_sha256"]:
        raise ValueError(f"Pi0.5 checkpoint SHA-256 mismatch: {path}")
    return path


def prepare_tokenizer(model: Path) -> None:
    import sentencepiece
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import GemmaConverter

    source = model / "paligemma_tokenizer.model"
    if sha256(source) != PIN["tokenizer_sha256"]:
        raise ValueError(f"PaliGemma tokenizer SHA-256 mismatch: {source}")
    original = sentencepiece.SentencePieceProcessor(model_file=str(source))
    special = {f"{name}_token": original.id_to_piece(getattr(original, f"{name}_id")())
               for name in ("pad", "bos", "eos", "unk")}
    backend = GemmaConverter(SimpleNamespace(vocab_file=str(source), **special)).converted()
    backend.post_processor = TemplateProcessing(
        single=f"{special['bos_token']} $A", special_tokens=[(special["bos_token"], original.bos_id())],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **special)
    for task in (
        "put both the alphabet soup and the tomato sauce in the basket",
        "put both the cream cheese and the butter in the basket",
        "put both the alphabet soup and the cream cheese box in the basket",
    ):
        prompt = f"Task: {task}, State: 0 32 64 127 128 192 254 255;\nAction: "
        if tokenizer(prompt)["input_ids"] != original.encode(prompt, add_bos=True):
            raise ValueError(f"Converted PaliGemma token IDs differ for {task!r}")
    tokenizer.save_pretrained(model / "tokenizer")


def load_config(model: Path, device: str):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PI05Config.from_pretrained(str(model), local_files_only=True, device=device)
    if (config.input_features["observation.state"].shape != (8,)
            or config.output_features["action"].shape != (7,)
            or config.chunk_size != 50 or config.n_action_steps != 50
            or any(config.input_features[key].shape != (3, 256, 256) for key in CAMERA_RENAME.values())):
        raise ValueError("Unexpected Pi0.5 LIBERO observation or action contract")
    config.compile_model = False
    config.gradient_checkpointing = False
    return config


def make_processors(model: Path, config):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import processor_pi05  # Register the saved Pi0.5 processor step.

    tokenizer = model / "tokenizer"
    if not (tokenizer / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Run the offline Pi0.5 tokenizer preflight first: {tokenizer}")
    return make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model),
        preprocessor_overrides={
            "device_processor": {"device": str(config.device)},
            "rename_observations_processor": {"rename_map": CAMERA_RENAME},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
        },
    )


def open_service(model: Path):
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("Pi0.5 episode inference requires CUDA")
    path = weights_path(model)
    config = load_config(model, "cuda:0")
    preprocessor, postprocessor = make_processors(model, config)
    policy = PI05Policy(config)
    # The upstream OpenPI loader can return a randomly initialized policy on load failure.
    state = policy._convert_openpi_state_dict(load_file(str(path)))
    policy.load_state_dict(state, strict=True)
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    return {"policy": policy, "preprocessor": preprocessor, "postprocessor": postprocessor}


def cpu_preflight(model: Path) -> dict:
    import torch

    config = load_config(model, "cpu")
    pre, post = make_processors(model, config)
    images = {name: torch.full((1, 3, 256, 256), float(i)) for i, name in enumerate(CAMERA_RENAME)}
    observation = pre({
        **images, "observation.state": torch.zeros((1, 8)),
        "task": "put both the alphabet soup and the tomato sauce in the basket",
    })
    if any(not torch.equal(observation[new], images[old]) for old, new in CAMERA_RENAME.items()):
        raise ValueError("Pi0.5 camera mapping is incorrect")
    action = post(torch.zeros((1, 50, 7)))
    if action.shape != (1, 50, 7) or not torch.isfinite(action).all():
        raise ValueError("Pi0.5 action postprocessing failed")
    return {"policy": PIN["repository"], "revision": PIN["revision"], "state_dim": 8,
            "camera_mapping": CAMERA_RENAME, "action_shape": list(action.shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--prepare-tokenizer", action="store_true")
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    model = model_directory(args.workspace)
    if args.prepare_tokenizer:
        prepare_tokenizer(model)
    report = cpu_preflight(model)
    if args.verify_weights:
        weights_path(model, verify=True)
        report["weights_sha256"] = PIN["weights_sha256"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
