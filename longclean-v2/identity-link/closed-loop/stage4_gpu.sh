#!/usr/bin/env bash
set -euo pipefail
mode=$1
card=$2
phase=$3
out=$4
[[ "$phase" =~ ^(smoke|pilot)$ && "$out" =~ ^stage4-[a-z0-9-]+$ ]]
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
source "$lab/longclean/scripts/setup_cache.sh"
case "$card" in
    2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} ;;
    3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} ;;
    *) exit 2 ;;
esac
stamp="$out-$mode-$card.exit"
test ! -e "$stamp"
finish() {
    rc=$?
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    printf '%s\n' "$rc" > "$stamp"
}
trap finish EXIT
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if [[ "$usage" != "0, 0" ]]; then
    [[ "$usage" == [12]", 0" ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found' || exit 75
fi
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$mode" in
    screen) cmd=(stage4_fixture.py screen --phase "$phase" --card "$card" --study ../../study4 --output "$out") ;;
    run) cmd=(run_closed_loop.py "$phase" --card "$card" --study ../../study4 --output "$out") ;;
    legacy) cmd=(legacy_parity.py replay --card "$card" --study ../../study4 --output "$out") ;;
    *) exit 2 ;;
esac
timeout -k 25s 40m "$lab/workspace/.kinex/model-python/bin/python" -B -u "${cmd[@]}"
