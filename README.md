# TableClear

Fixed-arm tabletop cleaning in LIBERO, with SmolVLA as the low-level action executor and Kinex as the robot-agent harness.

## Current status

This repository currently implements the V1 foundation and a validated direct
SmolVLA LIBERO path:

- object inventory and cleanup state machine;
- next-object subtask selection;
- bounded retry and unreachable-object handling;
- TOML experiment configuration and matrix expansion;
- JSONL event logging;
- episode-result CSV aggregation;
- storage and Linux runtime checks;
- replaceable environment, VLA and progress-verifier contracts;
- one-episode controller that keeps the simulator verdict authoritative;
- pinned SmolVLA LIBERO Kinex model-device definition and driver;
- an offline-loadable SmolVLA + SmolVLM2 base-model stack;
- a one-action device smoke and a deterministic direct-VLA episode runner;
- Kinex `table-clear` skill template.

The current AutoDL host has completed the no-GPU preparation: TableClear,
Kinex, RoboEnv, LIBERO dependencies/assets, the exported device workspace,
SmolVLA's model environment and both pinned model artifacts are ready. A GPU is
still required to launch LIBERO and load the policy on CUDA.

On the previous GPU host, the direct path executed real LIBERO motion and sealed
a native simulator verdict. The current Kinex arm path uses absolute joint-position
execution rather than LIBERO's native OSC controller, so benchmark performance
is not yet validated. The first 600-action direct-VLA smoke completed
operationally but did not solve `libero-10-01`.

## V1 task

A fixed Franka Panda arm must move every object from a bounded tabletop region into one basket:

```text
observe table
-> maintain remaining-object inventory
-> choose the next unfinished object
-> ask SmolVLA for a short action chunk
-> execute through Kinex/RoboEnv
-> verify the object is in the basket
-> repeat until all objects are cleared
-> let LIBERO score the final state
```

The V1 experiment compares:

- `direct_vla`: one full instruction for the whole cleanup;
- `fixed_checklist_vla`: a fixed list without feedback-driven updates;
- `closed_loop_vla`: dynamic state, visual checks and bounded retries.

Fixed method prompts live in `configs/methods/` so behavior cannot drift between
episodes.

The three V1 scenes use official `libero_10` tasks whose success condition is
"put both specified objects in the basket". V1 does not change LIBERO's scene
files or success rules. Larger 3-4 object layouts are a later upgrade.

## Local checks

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m tableclear.cli doctor
python -m tableclear.cli matrix --config configs/v1_smoke.toml
python -m tableclear.cli matrix --config configs/v1_eval.toml
python -m tableclear.cli demo
```

## Linux target

CPU-only hosts can install and statically validate all software and model files.
Running the simulator and CUDA policy requires Ubuntu and an NVIDIA GPU:

```bash
bash scripts/setup_cache.sh
bash scripts/check_storage.sh
bash scripts/bootstrap_linux.sh
bash scripts/bootstrap_smolvla_model.sh \
  /root/autodl-tmp/tableclear-workspace
bash scripts/install_smolvla_device.sh \
  /root/autodl-tmp/tableclear-workspace

# One-action device/model acceptance
node scripts/run_kinex_smoke.mjs \
  /root/autodl-tmp/tableclear-workspace \
  /root/autodl-tmp/tableclear-workspace/results/device-smoke \
  /root/autodl-tmp/kinex-data/install/node_modules/@kinex/core/dist

# Deterministic direct-VLA episode smoke
node scripts/run_direct_episode.mjs \
  /root/autodl-tmp/tableclear-workspace \
  /root/autodl-tmp/tableclear-workspace/results/direct-episode \
  /root/autodl-tmp/kinex-data/install/node_modules/@kinex/core/dist \
  "put both the alphabet soup and the tomato sauce in the basket" \
  20 10
```

Known-working upstream versions are recorded in `versions.lock`.

Do not install Kinex, MuJoCo or LIBERO into the Windows Python environment.

## AutoDL GPU availability

Check whether the AutoDL instance is online and ready for CUDA work:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_autodl_gpu.ps1
```

Exit code `0` means the instance changed from not-ready to GPU/CUDA-ready.
Codes `1` and `2` mean the instance is unreachable or has no GPU; code `3`
means it was already ready and has already been reported. Override `-SshHost`,
`-Port` and `-IdentityFile` when AutoDL returns a new connection address.

## Repository layout

```text
src/tableclear/      core state, controller, config, logging and CLI
configs/             V1 scene and evaluation definitions
.kinex/              Kinex tools and skills
scripts/             Linux bootstrap and storage checks
tests/               platform-independent tests
docs/                integration and smoke-test documentation
results/             generated experiment artifacts
```

## Boundaries

- No real robot.
- No mobile base or navigation.
- No drawer/container manipulation in V1.
- No VLA foundation-model training.
- Final success must come from the simulator, not from the language model.

