#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
lab=${1:?absolute personal data directory required}
mode=${2:?regression or holdout required}
trial=${3:?unique trial required}
[[ "$trial" =~ ^stage4-joint-pose-(reg-)?935614(-[0-9]+)?$ ]]
[[ "$mode" == regression || "$mode" == holdout ]]
(set -C; : > "$trial.log")
exec >> "$trial.log" 2>&1
if [[ "$mode" == holdout ]]; then
    card=${4:?GPU index required}
    gpu=${5:?GPU UUID required}
    reg=${6:?signed regression report required}
    reg_sha=${7:?regression SHA-256 required}
fi
finish() {
    local rc=$?
    date -Is
    if [[ "$mode" == holdout ]]; then
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader || true
    fi
    printf '%s\n' "$rc" > "$trial.exit"
}
trap finish EXIT
date -Is
hostname
findmnt -T "$PWD" -n -o TARGET,SOURCE
df -h .
for lock in placement_holdout_sources placement_holdout_replay placement_holdout_supplement placement_preopen_holdout placement_orientation_feedback placement_rgbd_pose placement_joint_pose; do
    sha256sum -c "$lock.sha256"
done
export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config" MPLCONFIGDIR="$lab/cache/xdg/matplotlib"
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable
sim="$lab/roboenv/.venv-sim/libero/bin/python"
entry=placement_joint_pose_holdout.py
calibration=stage4-rgbd-pose-dev-935640-2/calibration.json
if [[ "$mode" == regression ]]; then
    mkdir -- "$trial"
    timeout -k 10s 6m "$sim" -B -u "$entry" regression --calibration "$calibration" \
        --regression-report "$trial/regression.json" --output "$trial"
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
    timeout -k 30s 14m "$python" -B -u "$entry" "$stage" --calibration "$calibration" --regression-report "$reg" \
        --regression-sha "$reg_sha" --study ../../study4 --output "$trial"
done
