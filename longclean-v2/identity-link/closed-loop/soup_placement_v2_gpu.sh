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
stem="stage4-soup-placement-v2-$card"
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
printf '%s\n' 'd8ef0f1f5df2525cdbb152ea1b62644bb533a6bd2d453e98e65e58c5d979fe7e  soup_placement_feasibility_v2.py' | sha256sum -c -
test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ "$usage" =~ ^[012],\ 0$ ]] && nvidia-smi -i "$gpu" | grep -q 'No running processes found'
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout -k 30s 30m "$lab/roboenv/.venv-sim/libero/bin/python" -B -u soup_placement_feasibility_v2.py \
    --card "$card" --study ../../study4 --run stage4-dense-online --audit stage4-dense-online/audit.json \
    --source-report stage4-soup-execution/931301/report.json --output stage4-soup-placement-feasibility-v2
