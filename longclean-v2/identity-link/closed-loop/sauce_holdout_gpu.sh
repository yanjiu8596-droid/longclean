#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
card=2
gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2}
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
stem=stage4-sauce-holdout-replay
[[ ! -e "$stem.log" && ! -e "$stem.exit" && ! -e "$stem" ]] || exit 2
exec > "$stem.log" 2>&1
finish() {
    rc=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    printf '%s\n' "$rc" > "$stem.exit"
}
trap finish EXIT
export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
printf '%s\n' '1034b797ef0679cfa3bf3b7f5bb3cf2113d39d3ec8ac7567ef9f11e131106f57  sauce_holdout_replay.py' | sha256sum -c -
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ "$usage" =~ ^[0-8],\ 0$ ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found'
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout -k 30s 18m "$lab/workspace/.kinex/model-python/bin/python" -B -u sauce_holdout_replay.py \
    --card "$card" --study ../../study4 --source stage4-basket-retreat-holdout --output "$stem"
