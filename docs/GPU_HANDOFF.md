# GPU Handoff

The no-GPU instance is suitable only for source and CPU dependency preparation.
Before the first simulator launch:

1. Wait for the user to manually start a GPU instance using the prepared data
   disk. The latest reported package was RTX 4090 24GB; verify the actual device
   rather than assuming the earlier 4090D configuration. Do not rent/start an
   instance or restore the cancelled GPU monitor automatically.
2. Verify the GPU:

   ```bash
   nvidia-smi
   /root/miniconda3/bin/python -c \
     "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

3. Re-activate the cache environment in every shell:

   ```bash
   source /root/autodl-tmp/longclean/scripts/setup_cache.sh
   export PATH="/root/autodl-tmp/longclean/.venv/bin:/root/autodl-tmp/kinex-data/node/current/bin:${PATH}"
   ```

4. Verify RoboEnv:

   ```bash
   cd /root/autodl-tmp/roboenv
   ./.venv/bin/roboenv doctor libero
   ```

5. Start one task and let RoboEnv keep running:

   ```bash
   ./.venv/bin/roboenv launch libero/libero-10-01 --port 8123
   ```

6. In the Kinex workspace, load and ping devices:

   ```bash
   cd /root/autodl-tmp/tableclear-workspace
   export ROBOENV_ENDPOINT=http://127.0.0.1:8123
   /root/.local/bin/kinex -p "/load all"
   /root/.local/bin/kinex -p "/ping all"
   ```

These checks establish device connectivity for the legacy IK and absolute joint
target path. The native OSC smoke below exercises the new Kinex backend. The
high-level provider and a complete closed-loop smoke episode remain prerequisites
for the 135-episode matrix.

The SSH host or port may change after starting a GPU instance. Update the local
SSH config accordingly. The dedicated public key may need to be installed again
if AutoDL recreates the system disk; files under `/root/autodl-tmp` should remain
on the data disk.

## Standalone native OSC reference

Run this from the project checkout on the prepared GPU host. The model Python
loads the pinned SmolVLA weights offline. The separate LIBERO Python owns the
official simulator, its native 7D relative OSC controller and its verdict.

```bash
cd /root/autodl-tmp/longclean
source scripts/setup_cache.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYOPENGL_PLATFORM=egl
workspace=/root/autodl-tmp/tableclear-workspace
out="${workspace}/results/native-osc-01-$(date -u +%Y%m%dT%H%M%SZ)"
"${workspace}/.kinex/model-python/bin/python" -u scripts/run_native_osc.py \
  --workspace "${workspace}" \
  --roboenv /root/autodl-tmp/roboenv \
  --sim-python /root/venvs/libero-cpu2/bin/python \
  --output "${out}"
```

The default run uses seed 0, init-state index 0 and a 520-action budget.
Check `episode.json` for the terminal reason and LIBERO verdict and
`actions.jsonl` for the actions actually stepped. This is the direct-VLA
reference for the planned Kinex parity test.

For detached RoboEnv plumbing checks launched with a custom `--run-dir`,
pass the resulting run directory's `run.json` to `roboenv stop` to use the
owner socket and retain the terminal evidence.

## Kinex native OSC smoke and replay

```bash
cd /root/autodl-tmp/longclean
bash scripts/run_kinex_native_smoke.sh
```

The script creates a run-local Kinex workspace with the existing robot manifest
and pinned SmolVLA weights. The tool sends up to ten 7D policy actions through
the local OSC socket. An independent simulator process then replays the recorded
actions from the same seed and LIBERO init-state index. `parity.json` reports
the comparison. Both runs close their simulator processes when finished.

`TABLECLEAR_NATIVE_OSC_SOCKET` selects this backend for `run_vla_subtask`.
The session advances only on accepted action batches and uses the observation
frame to reject stale requests. The legacy scheduler remains the default in
workspaces without that environment variable.

