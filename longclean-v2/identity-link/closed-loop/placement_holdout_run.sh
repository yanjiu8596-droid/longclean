#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
lab=${1:?absolute personal data directory required}
stage=${2:?collect, inspect, screen, pair or audit required}
card=${3:-}
gpu=${4:-}
stem=stage4-placement-holdout-935600
[[ "$stage" =~ ^(collect|inspect|screen|pair|audit)$ ]]
log="$stem.$stage.log"
(set -C; : > "$log")
exec >> "$log" 2>&1
finish() {
    local rc=$?
    if [[ -n "$gpu" ]]; then
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits || true
    fi
    printf '%s\n' "$rc" > "$stem.$stage.exit"
}
trap finish EXIT
date -Is
hostname
findmnt -T "$PWD" -n -o TARGET,SOURCE
df -h .

export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config" MPLCONFIGDIR="$lab/cache/xdg/matplotlib"
sha256sum -c placement_holdout_sources.sha256
if [[ "$stage" != collect ]]; then
    sha256sum -c placement_holdout_replay.sha256
fi

idle() {
    local actual util compute processes
    actual=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
    util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i "$gpu")
    [[ "$actual" == "$gpu" && "$util" == 0 && -z "$compute" && "$processes" == *"No running processes found"* ]]
}

if [[ "$stage" == collect || "$stage" == inspect || "$stage" == pair ]]; then
    [[ -n "$card" && "$gpu" == GPU-* ]]
    idle
    sleep 3
    idle
    nvidia-smi -i "$gpu" --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
else
    export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable
fi
args=(--study ../../study4 --output "$stem")
if [[ "$stage" == collect ]]; then
    timeout -k 30s 45m "$lab/workspace/.kinex/model-python/bin/python" -B -u placement_holdout_collect.py \
        "${args[@]}" --card "$card" --gpu "$gpu"
else
    timeout -k 30s 20m "$lab/roboenv/.venv-sim/libero/bin/python" -B -u placement_holdout.py "$stage" "${args[@]}"
fi
