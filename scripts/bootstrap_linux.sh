#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'error: full TableClear runtime requires Linux\n' >&2
  exit 1
fi

for command in git nvidia-smi; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'error: missing required command: %s\n' "${command}" >&2
    exit 1
  fi
done

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x /root/miniconda3/bin/python ]]; then
    python_bin=/root/miniconda3/bin/python
  elif command -v python3.12 >/dev/null 2>&1; then
    python_bin=$(command -v python3.12)
  elif command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
  else
    printf 'error: Python >=3.11 is required (set PYTHON_BIN to override).\n' >&2
    exit 1
  fi
fi

source "$(dirname "$0")/setup_cache.sh"
bash "$(dirname "$0")/check_storage.sh"

"${python_bin}" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install uv
./.venv/bin/python -m pip install -e . --no-build-isolation

PATH="$(pwd)/.venv/bin:${PATH}"
export PATH

printf '\nTableClear core installed.\n'
printf 'Python: %s\n' "$(./.venv/bin/python --version)"
printf 'uv: %s\n' "$(./.venv/bin/uv --version)"
printf 'Next: source .venv/bin/activate and install Kinex/RoboEnv via docs/LINUX_BOOTSTRAP.md.\n'

