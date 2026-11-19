#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
lab=${1:?absolute personal data directory required}
card=${2:?GPU index required}
gpu=${3:?GPU UUID required}
primary=stage4-placement-holdout-935600
development=stage4-placement-holdout-supplement-935600
stem=stage4-placement-preopen-935632
(set -C; : > "$stem.log")
exec >> "$stem.log" 2>&1
finish() {
    local rc=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits || true
    printf '%s\n' "$rc" > "$stem.exit"
}
trap finish EXIT
date -Is
hostname
findmnt -T "$PWD" -n -o TARGET,SOURCE
df -h .
sha256sum -c placement_holdout_sources.sha256
sha256sum -c placement_holdout_replay.sha256
sha256sum -c placement_holdout_supplement.sha256
sha256sum -c placement_preopen_holdout.sha256
export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config" MPLCONFIGDIR="$lab/cache/xdg/matplotlib"

idle() {
    [[ "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" == "$gpu" ]]
    [[ "$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" == 0 ]]
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
    local processes
    processes=$(nvidia-smi -i "$gpu")
    [[ "$processes" == *"No running processes found"* ]]
}

sim="$lab/roboenv/.venv-sim/libero/bin/python"
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable timeout 2m "$sim" -B -u placement_contact_clearance_probe.py \
    --study ../../study4 --source "$development" --output "$stem.cutoffs.json"
sha256sum "$stem.cutoffs.json"
for stage in collect inspect screen pair audit; do
    printf '%s stage=%s\n' "$(date -Is)" "$stage"
    if [[ "$stage" == screen || "$stage" == audit ]]; then
        export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable
    else
        idle
        sleep 3
        idle
        nvidia-smi -i "$gpu" --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
        export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    fi
    python="$sim"
    if [[ "$stage" == collect ]]; then
        python="$lab/workspace/.kinex/model-python/bin/python"
    fi
    timeout -k 30s 16m "$python" -B -u placement_preopen_holdout.py "$stage" \
        --primary "$primary" --development "$development" --study ../../study4 --output "$stem"
done
