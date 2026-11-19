#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
card=$1
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
case "$card" in
    2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} ;;
    3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} ;;
    *) exit 2 ;;
esac
stem="stage4-dense-online-$card"
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
printf '%s\n' '02e0bb18130d7f19ab65df2a85b3b859ca11b873a69b117c82f22fc729113119  stage4-release-dense/schedule-continuation.json' | sha256sum -c -
printf '%s\n' '6de3c9241ad9f592a5c04639e022bcb623f3ee98d5368d7670813b2cd7a2654f  dense_online.py' | sha256sum -c -
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ "$usage" =~ ^[012],\ 0$ ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found'
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout -k 30s 3h "$lab/workspace/.kinex/model-python/bin/python" -B -u dense_online.py \
    --card "$card" --study ../../study4 --output stage4-dense-online \
    --regression stage4-release-dense/regression.json \
    --regression-sha 024410dda19a86fd30788cc1ab1040cb7e7a94f875eee534aa3360b2fa62ac6a \
    --controls stage4-sauce-controls/audit.json \
    --controls-sha 8866d8b3627b7afe04fde0914237e39d3e8c64fc7b5b6684144fe3ecbce8ea82
