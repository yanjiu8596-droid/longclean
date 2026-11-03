#!/usr/bin/env python3
"""SmolVLA LIBERO model device for Kinex."""

from __future__ import annotations

import contextlib
import pathlib
import sys
from typing import Any

import yaml
from kinex.driver_protocol import run


HERE = pathlib.Path(__file__).resolve().parent
SERVICE: dict[str, Any] | None = None


def profile() -> dict[str, Any]:
    return yaml.safe_load((HERE / "profile.yaml").read_text(encoding="utf-8"))


def open_service(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    artifacts = {
        item["name"]: (HERE / item["path"]).resolve()
        for item in config["intrinsics"]["artifacts"]
    }
    model_path = artifacts["smolvla_libero"]
    base_model_path = artifacts["smolvlm2_500m_video_instruct"]
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"SmolVLA checkpoint is missing: {model_path}. "
            "Run scripts/download_smolvla.py first."
        )
    if not (base_model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"SmolVLM2 base model is missing: {base_model_path}. "
            "Run scripts/download_smolvla.py first."
        )

    compute = config["access"]["connection"]["compute"]
    if compute["kind"] != "cuda":
        raise ValueError("SmolVLA requires CUDA in this project")
    device = torch.device("cuda", int(compute["index"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; start the instance with a GPU")

    policy_config = SmolVLAConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    policy_config.vlm_model_name = str(base_model_path)
    policy = SmolVLAPolicy.from_pretrained(
        str(model_path),
        config=policy_config,
        local_files_only=True,
    )
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(base_model_path)},
        },
    )
    preprocessor.reset()
    postprocessor.reset()
    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": device,
    }


def infer(service: dict[str, Any], operation: str, args: dict[str, Any]) -> dict[str, Any]:
    if operation != "infer":
        raise ValueError(f"unsupported SmolVLA operation: {operation}")

    import numpy as np
    import torch
    from PIL import Image

    obs = args.get("obs")
    task = args.get("task")
    if not isinstance(obs, dict):
        raise ValueError("obs must be an object")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must not be empty")

    camera1 = _image_tensor(obs.get("camera1"), torch, np, Image)
    camera2 = _image_tensor(obs.get("camera2"), torch, np, Image)
    state = obs.get("state")
    if not isinstance(state, list) or len(state) != 8:
        raise ValueError("state must contain exactly 8 values")

    batch = {
        "observation.images.camera1": camera1,
        "observation.images.camera2": camera2,
        "observation.state": torch.tensor(state, dtype=torch.float32),
        "task": task.strip(),
    }
    batch = service["preprocessor"](batch)
    with torch.inference_mode():
        actions = service["policy"].predict_action_chunk(batch)
    actions = service["postprocessor"](actions)
    if not isinstance(actions, torch.Tensor):
        raise TypeError("SmolVLA postprocessor did not return a tensor")
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
        raise ValueError(f"unexpected action chunk shape: {tuple(actions.shape)}")

    return {"actions": actions[0].detach().cpu().tolist()}


def _image_tensor(
    path: Any,
    torch: Any,
    np: Any,
    image_module: Any,
) -> Any:
    if not isinstance(path, str) or not path:
        raise ValueError("camera path must be a non-empty string")
    image = image_module.open(path).convert("RGB")
    array = np.asarray(image)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
    # Match LeRobot's LIBERO/HuggingFaceVLA camera convention.
    return torch.flip(tensor, dims=[1, 2])


def close_service(service: dict[str, Any]) -> None:
    service.clear()


def handle(action: str, args: dict[str, Any]) -> dict[str, Any]:
    global SERVICE
    if action in {"read", "status"}:
        return {"loaded": SERVICE is not None}
    if action == "unload":
        service, SERVICE = SERVICE, None
        if service is not None:
            close_service(service)
        return {"loaded": False}

    config = profile()
    operation_names = [
        operation["name"]
        for operation in config["intrinsics"].get("operations", [])
    ]
    if action != "load" and action not in operation_names:
        raise ValueError(f"undeclared model operation: {action}")
    if SERVICE is None:
        SERVICE = open_service(config)
    if action == "load":
        return {"loaded": True}
    return infer(SERVICE, action, args)


def respond(action: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = handle(action, args)
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    run(respond, lambda: handle("unload", {}))


if __name__ == "__main__":
    main()

