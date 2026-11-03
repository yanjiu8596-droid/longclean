# Linux Bootstrap

The full runtime is supported on Ubuntu 22.04 with an NVIDIA CUDA GPU. Run the
project from the data disk, not the 30 GB system disk.

## 1. Prepare storage

```bash
cd /root/autodl-tmp/longclean
source scripts/setup_cache.sh
bash scripts/check_storage.sh
bash scripts/bootstrap_linux.sh
```

Do not continue when less than 8 GiB is free.

## 2. Install the TableClear core

```bash
source .venv/bin/activate
tableclear doctor
tableclear matrix --config configs/v1_eval.toml
tableclear demo
```

## 3. Install Kinex and RoboEnv

Install Kinex using its official installer and pin the version used for the
experiment. Clone RoboEnv at a pinned commit under a separate work directory.

```bash
git clone https://github.com/stacx-lab/RoboEnv.git \
  /root/autodl-tmp/roboenv
cd /root/autodl-tmp/roboenv
git checkout <PINNED_COMMIT>
bash tools/bootstrap.sh
source .venv/bin/activate
roboenv prepare libero
```

Record the exact Kinex and RoboEnv commits in `versions.lock`.

If the installer reaches its 30-minute subprocess cap while wheels are still
being written, preserve the cache and use a new prefix with the audited extended
runner from this project:

```bash
PATH="/root/autodl-tmp/longclean/.venv/bin:${PATH}" \
ROBOENV_ASSET_CACHE=/root/autodl-tmp/roboenv-assets \
UV_CACHE_DIR=/root/autodl-tmp/cache/uv \
/root/autodl-tmp/roboenv/.venv/bin/python \
  /root/autodl-tmp/longclean/scripts/prepare_libero_extended.py \
  --prefix /root/autodl-tmp/roboenv/.venv-sim/libero-extended
```

If uv's cache extraction itself repeats for more than an hour without
registering an interpreter, use the direct-pip fallback. It still performs
RoboEnv's lock audit, import probes and registration:

```bash
PATH="/root/autodl-tmp/longclean/.venv/bin:${PATH}" \
ROBOENV_ASSET_CACHE=/root/autodl-tmp/roboenv-assets \
/root/autodl-tmp/roboenv/.venv/bin/python \
  /root/autodl-tmp/longclean/scripts/prepare_libero_pip.py \
  --prefix /root/venvs/libero-pip
```

The official Kinex installer stores its managed Node runtime under
`/root/autodl-tmp/kinex-data` when run with:

```bash
export KINEX_DATA_DIR=/root/autodl-tmp/kinex-data
bash install.sh
```

If direct GitHub access stalls, enable AutoDL's session-local network helper and
retry. The final LIBERO checkout must still match the commit pinned by RoboEnv.

```bash
source /etc/network_turbo
roboenv sources sync --backends libero
roboenv sources status --backends libero
```

## 4. First simulator task

```bash
roboenv tasks list --simulator libero | grep libero-10-01
roboenv launch libero/libero-10-01 --port 8123
```

Keep the launch terminal alive. In another terminal:

```bash
source /root/autodl-tmp/roboenv/.venv/bin/activate
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
mkdir -p /root/autodl-tmp/tableclear-workspace
cd /root/autodl-tmp/tableclear-workspace
roboenv setup libero/libero-panda-living-room
roboenv describe --endpoint "${ROBOENV_ENDPOINT}"
kinex
```

Do not start SmolVLA integration until the native simulator and Kinex device
connection both pass.

## 5. Native SmolVLA validation

Create a separate LeRobot environment if required by its current installation
instructions. Run a single LIBERO-Long task before adding the Kinex adapter:

```bash
lerobot-eval \
  --policy.path=<PINNED_SMOLVLA_LIBERO_CHECKPOINT> \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

Save the command, model revision, dataset metadata, action shapes and output log.

