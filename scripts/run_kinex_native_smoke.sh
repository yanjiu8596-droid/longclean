#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-/root/autodl-tmp/tableclear-workspace}"
roboenv="${ROBOENV_ROOT:-/root/autodl-tmp/roboenv}"
sim_python="${LIBERO_PYTHON:-/root/venvs/libero-cpu2/bin/python}"
core_dist="${KINEX_CORE_DIST:-/root/autodl-tmp/kinex-data/install/node_modules/@kinex/core/dist}"
run="${2:-${workspace}/results/kinex-native-$(date -u +%Y%m%dT%H%M%SZ)}"
model_python="${workspace}/.kinex/model-python/bin/python"

source "${project}/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYOPENGL_PLATFORM=egl
mkdir -p -- "$(dirname -- "${run}")"
mkdir -- "${run}"
run="$(cd -- "${run}" && pwd)"
echo "Run directory ${run}"
work="${run}/workspace"
mkdir -p -- "${run}/kinex-osc" "${work}/.kinex/devices" "${work}/.kinex/tools"
ln -s -- "${workspace}/models" "${work}/models"
ln -s -- "${workspace}/.kinex/model-python" "${work}/.kinex/model-python"
ln -s -- "${workspace}/.kinex/python" "${work}/.kinex/python"
cp -r -- "${project}/kinex_devices/smolvla" "${work}/.kinex/devices/"
cp -r -- "${project}/.kinex/tools/run_vla_subtask" "${work}/.kinex/tools/"

socket_dir="$(mktemp -d /tmp/tableclear-osc.XXXXXX)"
export TABLECLEAR_NATIVE_OSC_SOCKET="${socket_dir}/session.sock"
"${sim_python}" "${project}/scripts/run_native_osc.py" --sim-worker \
  --socket "${TABLECLEAR_NATIVE_OSC_SOCKET}" --roboenv "${roboenv}" \
  --output "${run}/kinex-osc" >"${run}/kinex-osc/simulator.log" 2>&1 &
sim_pid=$!
cleanup() {
  if [[ -n "${sim_pid}" ]]; then
    kill "${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  fi
  if [[ -S "${TABLECLEAR_NATIVE_OSC_SOCKET}" ]]; then
    unlink -- "${TABLECLEAR_NATIVE_OSC_SOCKET}"
  fi
  rmdir -- "${socket_dir}"
}
trap cleanup EXIT

for ((attempt=0; attempt<300; attempt++)); do
  [[ -S "${TABLECLEAR_NATIVE_OSC_SOCKET}" ]] && break
  kill -0 "${sim_pid}" 2>/dev/null || break
  sleep 0.1
done
if [[ ! -S "${TABLECLEAR_NATIVE_OSC_SOCKET}" ]]; then
  cat "${run}/kinex-osc/simulator.log" >&2
  exit 1
fi
node "${project}/scripts/run_kinex_smoke.mjs" "${work}" "${run}" "${core_dist}"
kill "${sim_pid}"
wait "${sim_pid}"
sim_pid=""
"${model_python}" "${project}/scripts/check_native_osc_parity.py" "${run}" \
  --sim-python "${sim_python}" --roboenv "${roboenv}"
