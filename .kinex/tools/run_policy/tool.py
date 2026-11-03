#!/usr/bin/env python3
"""Execute one SmolVLA inference through IK and the device scheduler.

Derived from Kinex 1.0.0's Apache-2.0 run-policy controller. The local tool
declares the SmolVLA observation fields in tool.yaml; upstream Kinex otherwise
strips undeclared nested observation keys while validating this tool's input.
"""

from kinex import RPCError, args, device, devices, finish, profile
from kinex.control import stop_on_error
from kinex.joint_control import command as joint_command, joint_model
from kinex.policy import cartesian_width, joint_sequence
from kinex.scalar_control import command as scalar_command
from kinex.validation import vector


SCALAR = {"gripper", "suction"}
JOINTED = {"joint", "dexterous_hand"}


def commanded(entry: dict) -> dict:
    name = entry["device"]
    prof = profile(name)
    kind = prof.get("type")
    if kind not in SCALAR | JOINTED:
        raise ValueError(
            f"{name} is {kind}; run_policy commands joint, "
            "dexterous_hand, gripper or suction devices"
        )
    model = None if kind in SCALAR else joint_model(prof)
    width = cartesian_width(entry, prof)
    return {
        "name": name,
        "profile": prof,
        "model": model,
        "spec": entry,
        "width": width or (1 if kind in SCALAR else len(model["joints"])),
    }


def share(action: list | dict, plan: list[dict]) -> dict[str, list]:
    if isinstance(action, dict):
        missing = [entry["name"] for entry in plan if entry["name"] not in action]
        if missing:
            raise ValueError(f"action names no value for {', '.join(missing)}")
        shares = {
            entry["name"]: vector(
                action[entry["name"]],
                entry["width"],
                f"action for {entry['name']}",
            )
            for entry in plan
        }
        return _scaled(shares, plan)

    vector(action, sum(entry["width"] for entry in plan), "policy action")
    shares = {}
    cursor = 0
    for entry in plan:
        shares[entry["name"]] = list(action[cursor : cursor + entry["width"]])
        cursor += entry["width"]
    return _scaled(shares, plan)


def _scaled(shares: dict[str, list], plan: list[dict]) -> dict[str, list]:
    for entry in plan:
        scale = entry["spec"].get("scale")
        if scale is not None:
            factors = vector(scale, entry["width"], f"scale for {entry['name']}")
            shares[entry["name"]] = [
                value * factor
                for value, factor in zip(shares[entry["name"]], factors)
            ]
        clip = entry["spec"].get("clip")
        if clip is not None:
            lower, upper = vector(clip, 2, f"clip for {entry['name']}")
            if lower > upper:
                raise ValueError(
                    f"clip lower bound exceeds upper bound for {entry['name']}"
                )
            shares[entry["name"]] = [
                max(lower, min(upper, value))
                for value in shares[entry["name"]]
            ]
    return shares


def control(request: dict) -> dict:
    model = request["model"]
    policy = profile(model)
    operations = [
        operation["name"]
        for operation in policy.get("intrinsics", {}).get("operations", [])
    ]
    if policy.get("type") != "model" or "infer" not in operations:
        raise ValueError("model must be a model device declaring infer")

    plan = [commanded(entry) for entry in request["command"]]
    if len({entry["name"] for entry in plan}) != len(plan):
        raise ValueError("command must name each device once")
    names = [entry["name"] for entry in plan]

    options = request.get("inference_args", {})
    if set(options) & {"obs", "task"}:
        raise ValueError("inference_args cannot override obs or task")
    inference = device(
        model,
        "infer",
        {
            **options,
            "obs": request["observation"],
            **({"task": request["task"]} if request.get("task") else {}),
        },
    )
    actions = inference.get("actions")
    if actions is None:
        single = inference.get("action")
        if single is None:
            raise ValueError("infer must return action or actions")
        actions = [single]
    if not isinstance(actions, list) or not actions:
        raise ValueError("infer actions must be a non-empty list")
    if request.get("max_actions") is not None:
        actions = actions[: request["max_actions"]]

    split = [share(action, plan) for action in actions]
    sequences = {}
    count = len(split)
    ik_failure = None
    for entry in plan:
        samples = [action[entry["name"]] for action in split[:count]]
        if entry["model"] is not None:
            sequence, failure = joint_sequence(entry, samples)
            if failure is not None:
                count = len(sequence)
                ik_failure = failure
                if count == 0:
                    return {
                        "actions_submitted": 0,
                        "duration_s": 0,
                        "execution_ids": {},
                        "ik_failure": ik_failure,
                    }
        else:
            sequence = [action[0] for action in samples]
        sequences[entry["name"]] = sequence

    duration = count / request.get("action_hz", 10)
    for entry in plan:
        name = entry["name"]
        sequences[name] = sequences[name][:count]
        command = {"actions": sequences[name], "duration_s": duration}
        if entry["model"] is not None:
            joint_command(entry["profile"], command, entry["model"])
        else:
            scalar_command(entry["profile"], command)

    with stop_on_error(names):
        results = devices(
            [
                {
                    "deviceID": name,
                    "action": "write",
                    "args": {
                        "actions": sequences[name],
                        "duration_s": duration,
                    },
                }
                for name in names
            ]
        )["results"]
        for result in results:
            if not result.get("ok"):
                raise RPCError(
                    result.get("error", "policy write failed"),
                    code=result.get("code"),
                    details=result.get("details"),
                )
        identities = {
            name: result["data"]["execution_id"]
            for name, result in zip(names, results)
        }
        return {
            "actions_submitted": count,
            "duration_s": duration,
            "execution_ids": identities,
            "ik_failure": ik_failure,
        }


if __name__ == "__main__":
    finish(control(args()))
