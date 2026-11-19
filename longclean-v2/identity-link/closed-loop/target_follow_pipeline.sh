#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
mdl="$lab/workspace/.kinex/model-python/bin/python"
out=stage4-target-follow

if [[ "${1:-}" == worker ]]; then
    card=$2
    case "$card" in
        2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} ;;
        3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} ;;
        *) exit 2 ;;
    esac
    stem="$out-$card"
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
    test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
    ready=0
    for _ in $(seq 1 60); do
        usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || exit 74
        pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) || exit 74
        if [[ -z "$pids" && "$usage" =~ ^[012],\ 0$ ]] &&
            nvidia-smi -i "$gpu" | grep -q 'No running processes found'; then
            ready=1
            break
        fi
        sleep 5
    done
    [[ $ready == 1 ]] || exit 75
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    timeout -k 20s 35m "$mdl" -B -u target_follow.py --card "$card" --study ../../study4 \
        --smoke stage4-smoke --output "$out"
    exit
fi

[[ ! -e "$out" && ! -e "$out-pipeline.exit" && ! -e stage4-pilot ]] || exit 2
trap 'rc=$?; printf "%s\n" "$rc" > "$out-pipeline.exit"' EXIT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
printf '%s\n' TARGET_FOLLOW_STARTED
sha256sum target_follow.py target_follow_audit.py target_follow_pipeline.sh stage4-smoke/frozen.json ../../study4/frozen.json
for card in 2 3; do
    tmux new-session -d -s "lc4-target-follow-$card" "bash target_follow_pipeline.sh worker $card"
done
for _ in $(seq 1 492); do
    if [[ -s "$out-2.exit" && -s "$out-3.exit" ]]; then
        printf 'WORKERS: GPU2=%s GPU3=%s\n' "$(cat "$out-2.exit")" "$(cat "$out-3.exit")"
        [[ $(cat "$out-2.exit") == 0 && $(cat "$out-3.exit") == 0 ]]
        "$mdl" -B -u target_follow_audit.py --run "$out" --out "$out/audit.json"
        printf '%s\n' TARGET_FOLLOW_SCORED
        exit 0
    fi
    sleep 5
done
exit 124
