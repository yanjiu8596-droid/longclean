# V1 Status

## Historical Completed Work

The preparation results in this section describe the earlier host. The GPU
clone results have their own dated entry below.

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

Before the GPU clone, static model checks confirmed the 8D state, two-camera
input, 7D action output and 50-step action chunk. The GPU results below are
from the new instance.

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

## Local Handoff Check — 2026-11-04 (Asia/Shanghai)

- Starting local `main` HEAD and live GitHub `main` both matched
  `10a103adbd195692dc2922e5a0ee2622de482495`; working tree was clean before
  this documentation update.
- This session reran the platform-independent suite with Windows Python 3.13:
  **18 tests passed**.
- This session expanded `v1_smoke.toml` to **1 job** and `v1_eval.toml` to
  **135 jobs**. No simulation episodes were run.
- Inspected the locked RoboEnv, Kinex, LIBERO and LeRobot sources. The current
  bridge and backend are not native OSC equivalent; see
  [`NATIVE_OSC_AUDIT.md`](NATIVE_OSC_AUDIT.md).
- No AutoDL connection, server inspection, GPU start, environment installation
  or automation creation occurred during that November 4 handoff check.

## GPU Clone Check — 2026-11-07 (Asia/Shanghai)

- Verified one RTX 4090 (24564 MiB), CUDA availability from model Python
  (`torch 2.8.0+cu128`), and 144 GiB available on the data disk. Reused the
  cloned environments and local model weights.
- At `0692093`, `libero-10-01` launched in stepped mode, EGL produced both
  camera views, and all six Kinex devices loaded. SmolVLA executed one action
  through the joint-position bridge. The owner stopped the simulator after
  the plumbing check. Its final verdict was `success=false`.
- At `80ccee35177148dd359f440a20fff675ee844f1d`, the standalone native OSC
  direct-VLA episode used official task 0, seed 0, init-state index 0, and
  10 settling frames. It logged 520 policy actions in 11 model calls at 20 Hz.
  The 520 action frames advanced 26 seconds of simulation time. The step
  budget ended the episode and LIBERO reported `success=false`.
- The two paths used different initial settling procedures. The captured
  states also differ in end-effector height and gripper qpos. A paired parity
  test with identical initialization is needed to isolate those differences.
- Results are on the data disk under `tableclear-workspace/results/` in
  `gpu-device-20261107T014118Z` and `native-osc-01-20261107T014440Z`.
  The native run retained its episode metadata, raw actions and both camera
  views before and after the episode. The first run retained the owner verdict.
- The local suite passed 18 tests after adding the native episode entry point.

## Remaining

- Expose the native OSC path to Kinex and verify action, time and observation
  parity against the standalone reference under the same initialization.
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

Steps 5–6 on the existing joint bridge are plumbing checks only. Native OSC
action/observation parity must be accepted separately before any benchmark
interpretation; a completed episode record is not a successful task.

