#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
card=$1
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
case "$card" in
    2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2}; layout=931301 ;;
    3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3}; layout=931303 ;;
    *) exit 2 ;;
esac
stem="stage4-soup-execution-$card"
[[ ! -e "$stem.log" && ! -e "$stem.exit" ]] || exit 2
exec > "$stem.log" 2>&1
finish() {
    rc=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    printf '%s\n' "$rc" > "$stem.exit"
}
trap finish EXIT
export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
printf '%s\n' '13dbd5a239f96be7d2fffaaedb9eec72ec331395819c2439c82a0b9501d656b5  soup_execution_replay.py' | sha256sum -c -
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ "$usage" =~ ^[012],\ 0$ ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found'
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout -k 30s 30m "$lab/workspace/.kinex/model-python/bin/python" -B -u soup_execution_replay.py \
    --card "$card" --layout "$layout" --study ../../study4 --run stage4-dense-online \
    --audit stage4-dense-online/audit.json --output stage4-soup-execution
