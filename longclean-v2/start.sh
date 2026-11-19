#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd -P)
phase=$1
card=$2
gpu=$3
run=$4
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
exec >> "$run/$phase-$card.log" 2>&1
finish() {
    status=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    printf '%s\n' "$status" > "$run/$phase-$card.exit.pending"
    mv "$run/$phase-$card.exit.pending" "$run/$phase-$card.exit"
}
trap finish EXIT
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if [ "$usage" != '0, 0' ]; then
    [[ "$usage" == [12]", 0" ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found' || exit 75
fi
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model="$lab/workspace/.kinex/model-python/bin/python"
timeout -k 30s 95m "$model" -B -u "$src/run.py" "$phase" "$card" --run "$run" < /dev/null
