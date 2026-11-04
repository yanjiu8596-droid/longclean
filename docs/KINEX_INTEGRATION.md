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

The dimensions and ordering must be checked against the native controller and
the running RoboEnv task. The existing bridge sends `space="pose_delta"` and
uses IK followed by absolute joint writes. This is the historical integration
path, not a native OSC-equivalent executor. See
[`NATIVE_OSC_AUDIT.md`](NATIVE_OSC_AUDIT.md) before changing or evaluating it.

The existing bridge applies translation scale `0.05 m`, rotation scale
`0.5 rad`, and clips the gripper command to `[-1, 1]`. These constants do not
make IK/joint-position execution equivalent to OSC. A native path must pass the
postprocessed 7D policy action to LIBERO's `OSC_POSE` controller without this
external physical-unit scaling or IK, and preserve native stepping semantics.
The current tool/script default is 30 Hz while the pinned RoboEnv backend uses
20 Hz; scheduling and executed frame counts still need to be aligned.

## `run_vla_subtask`

Inputs:

```json
{
  "instruction": "Pick up the red mug and place it into the basket.",
  "max_actions": 10
}
```

Output:

```json
{
  "execution_status": "completed",
  "action_space": "pose_delta",
  "actions_submitted": 10,
  "ik_failure": false,
  "images": ["head.png", "wrist.png"],
  "robot_state": [0.0]
}
```

The existing tool is intended to:

1. read fresh observations;
2. call the SmolVLA model device;
3. validate the action chunk;
4. send it through `run_policy`;
5. wait on the simulator clock;
6. read fresh feedback;
7. return images and structured execution evidence.

This interface's `completed` status means execution completed, not task success.
The native OSC replacement has not been implemented or accepted.

## Acceptance commands

```bash
# One model/device/action smoke
node scripts/run_kinex_smoke.mjs WORKSPACE OUTPUT KINEX_CORE_DIST

# Direct-VLA episode without an LLM provider
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

