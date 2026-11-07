from __future__ import annotations

import json
import math
import os
import socket
from collections.abc import Callable
from typing import Any


class VLAToolError(RuntimeError):
    pass


def execute(
    request: dict[str, Any],
    *,
    device: Callable[..., dict[str, Any]],
    call: Callable[..., dict[str, Any]],
    stop_owned: Callable[[], Any],
    profile: Callable[[str], dict[str, Any]] | None = None,
    native_request: Callable[[dict], dict] | None = None,
) -> dict[str, Any]:
    instruction = request.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise VLAToolError("instruction must be a non-empty string")

    max_actions = int(request.get("max_actions", 10))
    if not 1 <= max_actions <= 50:
        raise VLAToolError("max_actions must be between 1 and 50")

    model = str(request.get("model", "smolvla"))
    if native_request is not None:
        return _execute_native(instruction, max_actions, model, device, native_request)

    action_hz = float(request.get("action_hz", 30))
    if action_hz <= 0:
        raise VLAToolError("action_hz must be positive")

    head_camera = str(request.get("head_camera", "camera"))
    wrist_camera = str(request.get("wrist_camera", "camera_wrist"))
    arm = str(request.get("arm", "arm"))
    gripper = str(request.get("gripper", "gripper"))
    world = str(request.get("world", "world"))

    before_head = _camera_file(device, head_camera)
    before_wrist = _camera_file(device, wrist_camera)
    observation_state = _smolvla_state(device, arm, gripper, profile)

    try:
        execution = call(
            "run_policy",
            model=model,
            observation={
                "camera1": before_head,
                "camera2": before_wrist,
                "state": observation_state,
            },
            task=instruction,
            command=[
                {
                    "device": arm,
                    "space": "pose_delta",
                    "scale": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                },
                {"device": gripper, "clip": [-1.0, 1.0]},
            ],
            action_hz=action_hz,
            max_actions=max_actions,
        )

        if execution.get("ik_failure"):
            raise VLAToolError(f"IK rejected the action chunk: {execution}")
        submitted = int(execution.get("actions_submitted", 0))
        if submitted <= 0:
            raise VLAToolError(f"no actions were submitted: {execution}")

        duration_s = float(execution.get("duration_s", 0.0))
        if duration_s <= 0:
            raise VLAToolError(f"invalid run_policy duration: {duration_s}")
        device(world, "wait", {"duration_s": duration_s})

        after_head = _camera_file(device, head_camera)
        after_wrist = _camera_file(device, wrist_camera)
        arm_state = device(arm)
        gripper_state = device(gripper)

        return {
            "execution_status": "completed",
            "instruction": instruction,
            "model": model,
            "actions_submitted": submitted,
            "duration_s": duration_s,
            "execution_ids": execution.get("execution_ids", []),
            "before_images": [before_head, before_wrist],
            "after_images": [after_head, after_wrist],
            "observation_state": observation_state,
            "arm_state": arm_state,
            "gripper_state": gripper_state,
        }
    finally:
        stop_owned()


def _execute_native(instruction, max_actions, model, device, request):
    before = request({"op": "observe"})
    if before["controller"] != "OSC_POSE" or before["control_hz"] != 20:
        raise VLAToolError("expected the native 20 Hz LIBERO OSC controller")
    if before["terminal"]:
        raise VLAToolError("the native episode already ended")
    actions = device(model, "infer", {
        "obs": {
            "camera1": before["raw_images"][0],
            "camera2": before["raw_images"][1],
            "state": before["state"],
        },
        "task": instruction,
    }).get("actions")
    if not isinstance(actions, list) or not actions:
        raise VLAToolError("SmolVLA returned no actions")
    after = request({"op": "step", "frame": before["frame"], "actions": actions[:max_actions]})
    return {
        "execution_status": "completed",
        "control_protocol": "libero_native_osc",
        "instruction": instruction,
        "model": model,
        "actions_submitted": after["executed"],
        "duration_s": after["sim_time"] - before["sim_time"],
        "frame": after["frame"],
        "before_images": before["images"],
        "after_images": after["images"],
        "observation_state": before["state"],
        "after_observation_state": after["state"],
        "terminal": after["terminal"],
    }


def _native_request(path: str, message: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(60)
        connection.connect(path)
        connection.sendall(json.dumps(message, allow_nan=False).encode() + b"\n")
        with connection.makefile("rb") as reply:
            line = reply.readline(65_537)
    if not line:
        raise VLAToolError("native OSC session closed without a response")
    result = json.loads(line)
    if "error" in result:
        raise VLAToolError(result["error"])
    return result


def _camera_file(
    device: Callable[..., dict[str, Any]],
    camera: str,
) -> str:
    result = device(camera, args={"output": "file"})
    path = result.get("file")
    if not isinstance(path, str) or not path:
        raise VLAToolError(f"camera {camera} did not return a file path")
    return path


def _smolvla_state(
    device: Callable[..., dict[str, Any]],
    arm: str,
    gripper: str,
    profile: Callable[[str], dict[str, Any]] | None,
) -> list[float]:
    arm_state = device(arm)
    positions = arm_state.get("positions")
    if not isinstance(positions, list) or not positions:
        raise VLAToolError("arm read did not return joint positions")

    fk = device(
        arm,
        "fk",
        {"joints": positions},
    )
    pose = fk.get("pose")
    if not isinstance(pose, list) or len(pose) != 7:
        raise VLAToolError("arm FK must return a 7-value pose")
    pose = [float(value) for value in pose]
    if profile is not None:
        mount = profile(arm).get("extrinsics", {}).get("mount")
        if isinstance(mount, dict):
            pose = _compose_mount(pose, mount)

    gripper_state = device(gripper)
    aperture = gripper_state.get("aperture_m")
    if not isinstance(aperture, (int, float)):
        raise VLAToolError("gripper read must return aperture_m")

    axis_angle = _quaternion_xyzw_to_axis_angle(pose[3:7])
    half_aperture = float(aperture) / 2.0
    state = [
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        *axis_angle,
        -half_aperture,
        half_aperture,
    ]
    return state


def _compose_mount(
    pose_base_flange: list[float],
    mount: dict[str, Any],
) -> list[float]:
    translation = mount.get("translation_m")
    rotation = mount.get("rotation_xyzw")
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or not isinstance(rotation, list)
        or len(rotation) != 4
    ):
        raise VLAToolError("arm mount must contain translation_m and rotation_xyzw")

    try:
        mount_translation = [float(value) for value in translation]
        mount_rotation = _normalize_quaternion(
            [float(value) for value in rotation]
        )
        flange_translation = [float(value) for value in pose_base_flange[:3]]
        flange_rotation = _normalize_quaternion(pose_base_flange[3:7])
    except (TypeError, ValueError) as exc:
        raise VLAToolError(f"invalid arm mount or FK pose: {exc}") from exc

    rotated = _rotate_vector(flange_translation, mount_rotation)
    world_translation = [
        mount_translation[index] + rotated[index]
        for index in range(3)
    ]
    world_rotation = _quaternion_multiply(mount_rotation, flange_rotation)
    return [*world_translation, *world_rotation]


def _normalize_quaternion(quat: list[float]) -> list[float]:
    if len(quat) != 4:
        raise ValueError("quaternion must contain four values")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return [value / norm for value in quat]


def _quaternion_multiply(
    left: list[float],
    right: list[float],
) -> list[float]:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return _normalize_quaternion(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _rotate_vector(
    vector: list[float],
    quat: list[float],
) -> list[float]:
    x, y, z, w = quat
    q_vector = [x, y, z]
    twice_cross = [
        2.0 * (q_vector[1] * vector[2] - q_vector[2] * vector[1]),
        2.0 * (q_vector[2] * vector[0] - q_vector[0] * vector[2]),
        2.0 * (q_vector[0] * vector[1] - q_vector[1] * vector[0]),
    ]
    return [
        vector[0] + w * twice_cross[0]
        + q_vector[1] * twice_cross[2]
        - q_vector[2] * twice_cross[1],
        vector[1] + w * twice_cross[1]
        + q_vector[2] * twice_cross[0]
        - q_vector[0] * twice_cross[2],
        vector[2] + w * twice_cross[2]
        + q_vector[0] * twice_cross[1]
        - q_vector[1] * twice_cross[0],
    ]


def _quaternion_xyzw_to_axis_angle(
    quat_xyzw: list[float],
) -> list[float]:
    if len(quat_xyzw) != 4:
        raise VLAToolError("quaternion must contain four values")
    x, y, z, w = quat_xyzw
    w = max(-1.0, min(1.0, w))
    denominator = math.sqrt(max(0.0, 1.0 - w * w))
    if denominator <= 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.acos(w)
    return [
        x / denominator * angle,
        y / denominator * angle,
        z / denominator * angle,
    ]


def main() -> None:
    from kinex import args, call, device, fail, finish, profile, stop_owned

    native_socket = os.environ.get("TABLECLEAR_NATIVE_OSC_SOCKET")
    try:
        result = execute(
            dict(args()),
            device=device,
            call=call,
            stop_owned=stop_owned,
            profile=profile,
            native_request=(lambda message: _native_request(native_socket, message)) if native_socket else None,
        )
    except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
        fail(str(exc))
    else:
        finish(result)


if __name__ == "__main__":
    main()

