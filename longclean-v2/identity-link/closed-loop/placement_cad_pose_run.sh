#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
lab=${1:?absolute personal data directory required}
card=${2:?GPU index required}
gpu=${3:?GPU UUID required}
trial=${4:?unique run name required}
[[ "$trial" =~ ^stage4-cad-pose-935632(-[0-9]+)?$ ]]
source=stage4-placement-orientation-935636-dev
(set -C; : > "$trial.log")
exec >> "$trial.log" 2>&1
finish() {
    local status=$?
    date -Is
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader || true
    printf '%s\n' "$status" > "$trial.exit"
}
trap finish EXIT
date -Is
hostname
findmnt -T "$PWD" -n -o TARGET,SOURCE
df -h .
sha256sum -c placement_cad_pose.sha256
sha256sum -c placement_orientation_feedback.sha256
export TABLECLEAR_LAB="$lab" AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$lab/cache/xdg/libero" XDG_CONFIG_HOME="$lab/cache/xdg/config" MPLCONFIGDIR="$lab/cache/xdg/matplotlib"
sim="$lab/roboenv/.venv-sim/libero/bin/python"
idle() {
    [[ "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" == "$gpu" ]]
    [[ "$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" == 0 ]]
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
    local processes
    processes=$(nvidia-smi -i "$gpu")
    [[ "$processes" == *"No running processes found"* ]]
}
idle
sleep 3
idle
nvidia-smi -i "$gpu" --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
(
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    timeout -k 30s 8m "$sim" -B -u placement_cad_pose_probe.py capture "$source" "$trial" --study ../../study4
)
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable
timeout -k 10s 3m "$sim" -B -u placement_cad_pose_probe.py score "$trial" "$trial/score.json"
