#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
lab=${1:?absolute personal data directory required}
card=${2:?GPU index required}
gpu=${3:?GPU UUID required}
primary=stage4-placement-holdout-935600
supplement=stage4-placement-holdout-supplement-935600
previous=stage4-placement-preopen-935632
stem=stage4-placement-orientation-935636
dev=$stem-dev
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
for lock in placement_holdout_sources placement_holdout_replay placement_holdout_supplement placement_preopen_holdout placement_orientation_feedback; do
    sha256sum -c "$lock.sha256"
done
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
for stage in dev dev-audit collect inspect screen pair audit; do
    printf '%s stage=%s\n' "$(date -Is)" "$stage"
    case "$stage" in
        dev-audit|screen|audit) export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable ;;
        *)
            idle
            sleep 3
            idle
            nvidia-smi -i "$gpu" --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
            export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
            ;;
    esac
    python="$sim"
    if [[ "$stage" == collect ]]; then
        python="$lab/workspace/.kinex/model-python/bin/python"
    fi
    output="$stem"
    if [[ "$stage" == dev || "$stage" == dev-audit ]]; then
        output="$dev"
    fi
    timeout -k 30s 16m "$python" -B -u placement_orientation_feedback.py "$stage" \
        --primary "$primary" --supplement "$supplement" --previous "$previous" --study ../../study4 --output "$output"
    if [[ "$stage" == dev-audit ]]; then
        if ! "$sim" -B -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1]))["development_gate"])' "$dev/summary.json"; then
            printf 'DEVELOPMENT_GATE_FAILED\n'
            exit 0
        fi
    fi
done
