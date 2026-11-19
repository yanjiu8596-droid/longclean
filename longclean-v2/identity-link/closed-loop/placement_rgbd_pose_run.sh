#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
lab=${1:?absolute personal data directory required}
mode=${2:?development or holdout required}
trial=${3:?unique run name required}
[[ "$trial" =~ ^stage4-rgbd-pose-(dev-)?935640(-[0-9]+)?$ ]]
[[ "$mode" == development || "$mode" == holdout ]]
(set -C; : > "$trial.log")
exec >> "$trial.log" 2>&1
finish() {
    local rc=$?
    date -Is
    if [[ "$mode" == holdout ]]; then
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader || true
    fi
    printf '%s\n' "$rc" > "$trial.exit"
}
date -Is
hostname
findmnt -T "$PWD" -n -o TARGET,SOURCE
df -h .
if [[ "$mode" == holdout ]]; then
    card=${4:?GPU index required}
    gpu=${5:?GPU UUID required}
    calibration=${6:?frozen calibration JSON required}
    cal_sha=${7:?calibration SHA-256 required}
fi
trap finish EXIT
for lock in placement_holdout_sources placement_holdout_replay placement_holdout_supplement placement_preopen_holdout placement_orientation_feedback placement_rgbd_pose; do
    sha256sum -c "$lock.sha256"
done
export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config" MPLCONFIGDIR="$lab/cache/xdg/matplotlib"
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable
sim="$lab/roboenv/.venv-sim/libero/bin/python"
entry=placement_rgbd_pose_holdout.py
if [[ "$mode" == development ]]; then
    mkdir -- "$trial"
    calibration="$trial/calibration.json"
    timeout -k 10s 3m "$sim" -B -u "$entry" calibrate --calibration "$calibration" --output "$trial"
    cal_sha=$(sha256sum "$calibration" | cut -d ' ' -f1)
    timeout -k 10s 5m "$sim" -B -u "$entry" dev-score --calibration "$calibration" \
        --calibration-sha "$cal_sha" --output "$trial/development-score.json"
    exit
fi

idle() {
    [[ "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" == "$gpu" ]]
    [[ "$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" == 0 ]]
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
    local processes
    processes=$(nvidia-smi -i "$gpu")
    [[ "$processes" == *"No running processes found"* ]]
}

for stage in collect inspect screen pair audit score; do
    printf '%s stage=%s\n' "$(date -Is)" "$stage"
    case "$stage" in
        collect|inspect|pair)
            idle
            sleep 3
            idle
            nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
            export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
            ;;
        *) export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable ;;
    esac
    python="$sim"
    if [[ "$stage" == collect ]]; then
        python="$lab/workspace/.kinex/model-python/bin/python"
    fi
    timeout -k 30s 14m "$python" -B -u "$entry" "$stage" --calibration "$calibration" \
        --calibration-sha "$cal_sha" --study ../../study4 --output "$trial"
done
