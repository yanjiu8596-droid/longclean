#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
arm=${1:?A or B}
case "$arm" in A|B) ;; *) exit 2 ;; esac
review=$(dirname "$PWD")
source "$review/scene01-rgbd-grasp-recovery/env.sh"
export TABLECLEAR_LAB="$lab" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$lab/longclean/scripts:$lab/longclean/src:$review/scene01-early-night-20261123-b01"
uuid=$(sed -n '2p' "$review/scene01-warm75-20261124-sync/gpu.binding")
index=$(head -1 "$review/scene01-warm75-20261124-sync/gpu.binding")
idle() {
    local snapshot apps used util
    snapshot=$(nvidia-smi -i "$uuid" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || return 1
    apps=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) || return 1
    IFS=, read -r used util <<<"$snapshot"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && "$util" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && (( used < 128 && util == 0 ))
}
safe_egl() { idle && sleep 2 && idle; }
if [[ "$arm" == A ]]; then
    if safe_egl; then backend=egl; else backend=osmesa; fi
    printf '%s\n' "$backend" > backend.txt
else
    backend=$(<backend.txt)
    if [[ "$backend" == egl ]]; then
        until safe_egl; do
            printf '%s waiting for original GPU\n' "$(date -Is)"
            sleep 30
        done
    fi
fi
export MUJOCO_GL="$backend" PYOPENGL_PLATFORM="$backend"
if [[ "$backend" == egl ]]; then
    export CUDA_VISIBLE_DEVICES="$uuid" MUJOCO_EGL_DEVICE_ID="$index"
else
    export CUDA_VISIBLE_DEVICES=''
    unset MUJOCO_EGL_DEVICE_ID
fi
echo "$(date -Is) START $arm $backend"
finish() {
    local code=$?
    printf '%s\n' "$code" > "$arm.exit"
    echo "$(date -Is) EXIT $arm $code"
    if [[ "$backend" == egl ]]; then
        nvidia-smi -i "$uuid" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader
    fi
}
trap finish EXIT
for item in run_pair.py protocol.json launch.sh; do sha256sum "$item"; done
if [[ "$arm" == B ]]; then sha256sum slot.py; fi
"$sim" -B -u run_pair.py "$arm"
