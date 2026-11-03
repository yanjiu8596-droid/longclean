#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_TMP:-/root/autodl-tmp}"

mkdir -p \
  "${DATA_ROOT}/cache/huggingface" \
  "${DATA_ROOT}/cache/uv" \
  "${DATA_ROOT}/cache/pip" \
  "${DATA_ROOT}/cache/torch" \
  "${DATA_ROOT}/cache/xdg" \
  "${DATA_ROOT}/roboenv-assets" \
  "${DATA_ROOT}/tmp"

export ROBOENV_ASSET_CACHE="${DATA_ROOT}/roboenv-assets"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export UV_CACHE_DIR="${DATA_ROOT}/cache/uv"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/xdg"
export TMPDIR="${DATA_ROOT}/tmp"
export MUJOCO_GL=egl

printf 'Data root: %s\n' "${DATA_ROOT}"
printf 'Source this file in each shell: source scripts/setup_cache.sh\n'

