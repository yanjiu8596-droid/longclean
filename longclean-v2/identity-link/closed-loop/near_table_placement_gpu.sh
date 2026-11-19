#!/usr/bin/env bash
set -euo pipefail

lab=${1:?absolute personal experiment root required}
mode=${2:?check or run required}
card=${3:-}
gpu=${4:-}
rendering=${5:-strict}
refined_screen=${6:-}
refined_sha=${7:-}
cd -- "$(dirname -- "$0")"
[[ "$mode" == check || "$mode" == run ]]
[[ "$rendering" == strict || "$rendering" == different-renderer ]]
export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$lab/cache/xdg/matplotlib"
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config"

python="$lab/roboenv/.venv-sim/libero/bin/python"
if [[ -x "$lab/usr/bin/python3.10" ]]; then
    python="$lab/usr/bin/python3.10"
    export PYTHONHOME="$lab/usr"
    packages="$lab/roboenv/.venv-sim/libero/lib/python3.10/site-packages"
    export PYTHONPATH="$packages:$lab/roboenv/third_party/LIBERO"
fi

runner=near_table_placement_pair.py
protocol=near_table_placement_protocol.md
sha256sum -c near_table_placement_sources.sha256
source_args=(--study ../../study4 --source stage4-near-table-grasp)
stem=stage4-near-table-placement
if [[ -n "$refined_screen" || -n "$refined_sha" ]]; then
    [[ -n "$refined_screen" && -n "$refined_sha" && "$rendering" == strict ]]
    source_args+=(--refined-screen "$refined_screen" --refined-screen-sha256 "$refined_sha")
    stem=stage4-near-table-refined-placement
fi
if [[ "$mode" == check ]]; then
    "$python" -B -u "$runner" check "${source_args[@]}"
    "$python" -B - <<'PY'
import sys
from importlib.metadata import version

from actual_physics_simulator import load_native

native = load_native()
native.ensure_libero_config()
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
from robosuite.utils import camera_utils
print(sys.version.split()[0], {name: version(name) for name in ('mujoco', 'robosuite', 'numpy', 'scipy', 'PyOpenGL')})
PY
    exit
fi

[[ ! -e "$stem" && ! -e "$stem.exit" ]]
(set -o noclobber; : > "$stem.log")
exec >> "$stem.log" 2>&1
finish() {
    rc=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits || true
    printf '%s\n' "$rc" > "$stem.exit"
}
trap finish EXIT

idle() {
    local selected compute util
    selected=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
    compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ "$selected" == "$gpu" && -z "$compute" ]] || return 1
    util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    [[ "$util" == 0 ]] || return 1
    # Idle memory usage varies with the driver. Check the reported processes and utilization.
    nvidia-smi -i "$gpu" | awk '
        /No running processes found/ { seen = 1 }
        $1 == "|" && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ {
            seen = 1
            if ($6 != "G" || $7 != "/usr/lib/xorg/Xorg" || $8 + 0 > 8) busy = 1
        }
        END { exit !seen || busy }'
}
idle
sleep 3
idle
nvidia-smi -i "$gpu" --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
renderer_args=()
if [[ "$rendering" == different-renderer ]]; then
    renderer_args=(--different-renderer)
fi
sha256sum "$0" "$protocol" "$runner"
timeout -k 30s 45m "$python" -B -u "$runner" run "${source_args[@]}" --card "$card" --output "$stem" "${renderer_args[@]}"
"$python" -B -u "$runner" audit "${source_args[@]}" --output "$stem"
