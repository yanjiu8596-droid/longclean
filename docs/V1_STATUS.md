# V1 Status

## Completed

- Core object state machine and bounded retry behavior.
- One-episode runner that requests the simulator verdict at the end.
- Kinex `table-clear` skill.
- Durable `cleanup_state` tool.
- `run_vla_subtask` execution bridge with camera/state feedback.
- Pinned SmolVLA LIBERO model-device profile and persistent LeRobot driver.
- Eight-dimensional SmolVLA state assembly from FK, quaternion and gripper data.
- 135-episode V1 matrix and one-episode smoke matrix.
- JSONL logging and per-method aggregate CSV.
- Linux bootstrap, cache redirection and disk guards.
- Remote Kinex 1.0.0 installation on the data disk.
- Remote RoboEnv 1.0.0 development environment.
- LIBERO source pinned to the RoboEnv expected commit and marked `ready`.
- LIBERO Panda workspace exported with devices:
  `arm`, `gripper`, `camera`, `camera_wrist`, `smolvla`, `world`.

- LIBERO-specific Python 3.10 simulator environment installed and registered.
- Kinex 1.0.0 and Node 22 installed under the persistent data disk.
- RoboEnv 1.0.0 development environment installed; static and source gates pass.
- RoboEnv full CPU suite: 1558 passed, 130 skipped; privacy audit and wheel
  build/import checks pass.
- LIBERO source, import probes and bundled 130-task assets verified.
- SmolVLA Python environment installed against the image's CUDA PyTorch build.
- SmolVLA and SmolVLM2 base weights downloaded at pinned revisions.
- Kinex workspace exports all six device profiles and both project tool overrides.

## Current Result

The new host is currently in no-GPU mode, so simulator launch, EGL camera replay,
CUDA model loading and motion are intentionally not claimed. Static model checks
confirm the 8D state, two-camera input, 7D action output and 50-step action chunk.

On the previous GPU host, the direct-VLA smoke completed operationally but
`libero-10-01` returned `success=false`. RoboEnv reported that Kinex executed
absolute joint positions through `JOINT_POSITION/set_qpos`, not native LIBERO OSC:

```text
control_protocol: absolute_joint_position
native_osc_equivalence: false
```

This is an execution-equivalence limitation, not a model-loading or plumbing
failure. The next performance task is to expose or implement a native OSC action
path before running the 135-episode matrix.

## Remaining

- Implement native OSC-equivalent action execution.
- Configure the Kinex high-level model provider for visual inventory and
  closed-loop object-state updates.
- Run and aggregate the 135-episode V1 matrix.

## First GPU Acceptance Gate

1. Verify `nvidia-smi` and `torch.cuda.is_available()` after switching modes.
2. `roboenv launch libero/libero-10-01 --port 8123` starts.
3. `kinex -p "/ping all"` sees all six devices.
4. Load the pinned SmolVLA checkpoint on CUDA.
5. `run_vla_subtask` completes one bounded action execution.
6. `tableclear-01-smoke` produces one complete `episode_end` record.

