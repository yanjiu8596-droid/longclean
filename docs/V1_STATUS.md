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
  `arm`, `gripper`, `camera`, `camera_wrist`, `world`.

- LIBERO-specific Python 3.10 simulator environment installed and registered.
- MuJoCo EGL rendering and `roboenv launch` validated on RTX 4090D.
- All Kinex devices load and ping successfully.
- SmolVLA and its pinned SmolVLM2 base model load fully offline.
- A 50x7 action chunk reaches IK and executes through arm/gripper devices.
- A 600-action direct-VLA smoke produced preserved before/after images, a
  sealed RoboEnv terminal record, an `episode_end` JSONL record and CSV summary.

## Current Result

The direct-VLA smoke completed operationally but `libero-10-01` returned
`success=false`. RoboEnv reports that Kinex executed absolute joint positions
through `JOINT_POSITION/set_qpos`, not native LIBERO OSC:

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

1. `roboenv doctor libero` reports a registered reusable interpreter.
2. `roboenv launch libero/libero-10-01 --port 8123` starts.
3. `kinex -p "/ping all"` sees all five devices.
4. `run_vla_subtask` completes one bounded action execution.
5. `tableclear-01-smoke` produces one complete `episode_end` record.

