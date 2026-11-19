"""Strict V1 base loading, Pi0.5 LoRA, and masked flow matching."""

from hashlib import sha256
import re

import numpy as np
import torch

from common import HORIZON, state_vector


def load_service(lab, *, training=False, adapter=None):
    lab.add_imports()
    import native_pi05

    if not training:
        service = native_pi05.open_service(lab.model)
        if adapter is not None:
            from peft import PeftModel

            service["policy"] = PeftModel.from_pretrained(service["policy"], str(adapter), is_trainable=False)
            service["policy"].eval()
        return service

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    config = native_pi05.load_config(lab.model, "cuda:0")
    config.dtype = torch.bfloat16
    config.train_expert_only = True
    config.gradient_checkpointing = True
    if config.rtc_training_max_delay or config.use_proprioceptive_memory:
        raise ValueError("The V1 training study expects the pinned non-RTC, stateless action model")
    base = PI05Policy(config)
    weights = native_pi05.weights_path(lab.model, verify=True)
    base.load_state_dict(base._convert_openpi_state_dict(load_file(str(weights))), strict=True)
    base.to("cuda:0")
    targets = base._get_default_peft_targets()
    if targets["modules_to_save"]:
        raise ValueError("V1 post-training must update only LoRA parameters")
    expected = {
        name for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear) and re.fullmatch(targets["target_modules"], name)
    }
    policy = get_peft_model(base, LoraConfig(
        target_modules=targets["target_modules"], r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    ))
    found = {name.removeprefix("base_model.model.") for name, module in policy.named_modules()
             if hasattr(module, "lora_A")}
    if not expected or found != expected:
        raise ValueError(f"LoRA targets differ: expected {sorted(expected)}, got {sorted(found)}")
    pre, post = native_pi05.make_processors(lab.model, config)
    return {"policy": policy, "preprocessor": pre, "postprocessor": post, "targets": sorted(found)}


def observation(front, wrist, state, task):
    def tensor(image):
        return torch.as_tensor(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().div_(255)

    return {
        "observation.images.camera1": tensor(front),
        "observation.images.camera2": tensor(wrist),
        "observation.state": torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
        "task": task,
    }


def read_observation(path, task):
    with np.load(path, allow_pickle=False) as raw:
        state = state_vector(raw["eef_pos"], raw["eef_quat"], raw["gripper_qpos"])
        return observation(raw["camera1"][::-1, ::-1].copy(), raw["camera2"][::-1, ::-1].copy(), state, task)


def action_window(actions):
    actual = min(len(actions), HORIZON)
    if not actual:
        raise ValueError("Empty action supervision window")
    window = np.empty((HORIZON, 7), dtype=np.float32)
    window[:actual] = actions[:actual]
    window[actual:] = window[actual - 1]
    if not np.isfinite(window).all():
        raise ValueError("Non-finite native action within the training window")
    return torch.from_numpy(window).unsqueeze(0), torch.arange(HORIZON) < actual


def masked_mean(losses, valid):
    useful = losses[..., :7].float().masked_fill(~valid[None, :, None], 0)
    return useful.sum() / (losses.shape[0] * valid.sum() * 7)


def flow_loss(base, batch, valid):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = base._preprocess_images(batch)
    states, state_masks = base._prepare_memory_states(batch)
    actions = base.prepare_action(batch)
    losses = base.model.forward(
        images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], actions,
        base.model.sample_noise(actions.shape, actions.device),
        base.model.sample_time(actions.shape[0], actions.device),
        prefix_mask=None, states=states, state_masks=state_masks,
    )
    return masked_mean(losses, valid.to(device=losses.device))


def parameter_digest(policy, *, trainable):
    digest = sha256()
    for name, value in policy.named_parameters():
        if value.requires_grad != trainable:
            continue
        digest.update(name.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().data)
    return digest.hexdigest()
