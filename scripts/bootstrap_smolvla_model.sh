#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/root/autodl-tmp/tableclear-workspace}"
lerobot_root="${LEROBOT_ROOT:-/root/autodl-tmp/lerobot}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
model_python="${workspace}/.kinex/model-python"

if [[ ! -f "${lerobot_root}/pyproject.toml" ]]; then
  printf 'LeRobot checkout is missing: %s\n' "${lerobot_root}" >&2
  exit 1
fi

mkdir -p "${workspace}/.kinex"
if [[ ! -f "${model_python}/pyvenv.cfg" ]]; then
  "${python_bin}" -m venv --system-site-packages "${model_python}"
fi

"${model_python}/bin/python" -m pip install --upgrade pip setuptools wheel
"${model_python}/bin/python" -m pip install \
  -e "${lerobot_root}[smolvla]" \
  'Brotli==1.2.0' \
  PyYAML \
  --no-build-isolation

"${model_python}/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print(f"torch={torch.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"smolvla_policy={SmolVLAPolicy.__name__}")
PY

