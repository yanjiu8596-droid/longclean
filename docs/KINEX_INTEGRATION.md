# Kinex Integration Contract

The V1 Kinex workspace uses these logical devices:

```text
camera
camera_wrist
arm
gripper
world
smolvla
```

## SmolVLA model device

Install the pinned model-device template into a prepared workspace:

```bash
bash scripts/bootstrap_smolvla_model.sh \
  /root/autodl-tmp/tableclear-workspace
bash scripts/install_smolvla_device.sh \
  /root/autodl-tmp/tableclear-workspace
source /etc/network_turbo
/root/autodl-tmp/tableclear-workspace/.kinex/model-python/bin/python \
  scripts/download_smolvla.py \
  --output /root/autodl-tmp/tableclear-workspace/models/smolvla_libero
```

Pinned model: `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`.
The base VLM is pinned to
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
Both are copied into workspace-local model directories for offline loading.

The model device must be persistent. Each inference receives:

- head image;
- wrist image;
- robot state;
- one short subtask instruction;
- the live action schema.

It returns an action chunk. Kinex `max_actions` applies the execution horizon.
The model action contract is:

```text
6D end-effector delta + 1D gripper
```

With `TABLECLEAR_NATIVE_OSC_SOCKET` configured, Kinex passes each postprocessed
7D action directly to LIBERO's relative `OSC_POSE` controller at 20 Hz. The
session owns action validation, time, and the terminal verdict.

The legacy backend converts `pose_delta` through IK into absolute joint
targets. Its scheduler uses a 30 Hz default and applies the legacy scaling and
gripper clipping. The original audit records the protocol differences.
[`NATIVE_OSC_AUDIT.md`](NATIVE_OSC_AUDIT.md)

## `run_vla_subtask`

Inputs:

```json
{
  "instruction": "Pick up the red mug and place it into the basket.",
  "max_actions": 10
}
```

Selected fields from a native result:

```json
{
  "execution_status": "completed",
  "control_protocol": "libero_native_osc",
  "actions_submitted": 10,
  "frame": 10,
  "duration_s": 0.5,
  "terminal": false
}
```

The native tool reads the current frame, calls SmolVLA with both camera images
and the model state, and submits the bounded action batch against that frame.
It returns before and after images and model states. `completed` describes the
action batch. The independent simulator records task success.

## Acceptance commands

```bash
# Native OSC smoke with independent replay
bash scripts/run_kinex_native_smoke.sh

# Legacy model/device/action smoke
node scripts/run_kinex_smoke.mjs WORKSPACE OUTPUT KINEX_CORE_DIST

# Legacy direct-VLA episode without an LLM provider
node scripts/run_direct_episode.mjs WORKSPACE OUTPUT KINEX_CORE_DIST \
  "put both the alphabet soup and the tomato sauce in the basket" 20 10
```

Stop the RoboEnv run by run ID so the owner socket seals the simulator verdict:

```bash
roboenv stop RUN_ID --json
```

Then convert the episode output and `evidence/terminal.json` into the project's
`episode_end` JSONL format with `scripts/finalize_roboenv_episode.py`.

## Safety and correctness

- Never hardcode scene ground truth into prompts.
- Never allow the Agent to modify the simulator verdict.
- Never treat model confidence as task success.
- Stop owned motion on every error path.
- Keep episode sessions isolated.
- Save every VLA call and its image references in JSONL.

