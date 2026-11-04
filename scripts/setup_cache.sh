#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_TMP:-/root/autodl-tmp}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KINEX_DATA_ROOT="${KINEX_DATA_DIR:-${DATA_ROOT}/kinex-data}"

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

for path in \
  "${PROJECT_ROOT}/.venv/bin" \
  "${KINEX_DATA_ROOT}/node/current/bin" \
  "${HOME}/.local/bin"; do
  if [[ -d "${path}" && ":${PATH}:" != *":${path}:"* ]]; then
    PATH="${path}:${PATH}"
  fi
done
export PATH

printf 'Data root: %s\n' "${DATA_ROOT}"
printf 'Kinex data: %s\n' "${KINEX_DATA_ROOT}"
printf 'Source this file in each shell: source scripts/setup_cache.sh\n'

