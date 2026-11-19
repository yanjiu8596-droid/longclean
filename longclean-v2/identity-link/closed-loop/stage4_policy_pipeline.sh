#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
mdl="$lab/workspace/.kinex/model-python/bin/python"

idle() {
    local gpu=$1 usage pids
    for _ in $(seq 1 60); do
        usage=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || return 74
        pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) || return 74
        if [[ -z "$pids" && "$usage" =~ ^[012],\ 0$ ]] &&
            nvidia-smi -i "$gpu" | grep -q 'No running processes found'; then
            return 0
        fi
        sleep 5
    done
    return 75
}

if [[ "${1:-}" == worker ]]; then
    mode=$2 card=$3
    case "$card" in
        2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} ;;
        3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} ;;
        *) exit 2 ;;
    esac
    case "$mode" in
        diagnostic) prefix="stage4-policy-smoke-$card" ;;
        screen|run) prefix="stage4-pilot-$mode-$card" ;;
        *) exit 2 ;;
    esac
    test ! -e "$prefix.exit" && test ! -e "$prefix.log" || exit 2
    exec > "$prefix.log" 2>&1
    finish_worker() {
        rc=$?
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
        [[ -e "$prefix.exit" ]] || printf '%s\n' "$rc" > "$prefix.exit"
    }
    trap finish_worker EXIT
    test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
    idle "$gpu"
    if [[ "$mode" != diagnostic ]]; then
        bash stage4_gpu.sh "$mode" "$card" pilot stage4-pilot
        exit
    fi
    export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
    source "$lab/longclean/scripts/setup_cache.sh"
    idle "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    timeout -k 20s 18m "$mdl" -B -u progress_policy.py --card "$card" --study ../../study4 \
        --smoke stage4-smoke --output stage4-policy-smoke
    exit
fi

test ! -e stage4-policy-pipeline.exit && test ! -e stage4-policy-smoke && test ! -e stage4-pilot || exit 2
trap 'rc=$?; printf "%s\n" "$rc" > stage4-policy-pipeline.exit' EXIT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_pair() {
    local mode=$1 limit=$2 prefix=$3 card
    for card in 2 3; do
        tmux new-session -d -s "lc4-policy-$mode-$card" "bash stage4_policy_pipeline.sh worker $mode $card"
    done
    for _ in $(seq 1 "$limit"); do
        if [[ -s "$prefix-2.exit" && -s "$prefix-3.exit" ]]; then
            printf 'WORKERS %s: GPU2=%s GPU3=%s\n' "$mode" "$(cat "$prefix-2.exit")" "$(cat "$prefix-3.exit")"
            [[ $(cat "$prefix-2.exit") == 0 && $(cat "$prefix-3.exit") == 0 ]]
            return
        fi
        sleep 5
    done
    return 124
}

"$mdl" -B stage4_policy_gate.py preflight --smoke stage4-smoke --output stage4-policy-smoke
printf '%s\n' POLICY_DIAGNOSTIC_STARTED
run_pair diagnostic 290 stage4-policy-smoke
if "$mdl" -B stage4_policy_gate.py score --smoke stage4-smoke --output stage4-policy-smoke; then
    printf '%s\n' POLICY_DIAGNOSTIC_PASS
else
    rc=$?
    printf 'POLICY_DIAGNOSTIC_STOP %s\n' "$rc"
    exit "$rc"
fi

printf '%s\n' PILOT_SCREEN_STARTED
run_pair screen 552 stage4-pilot-screen
"$mdl" -B -u stage4_fixture.py freeze --phase pilot --study ../../study4 --output stage4-pilot
printf '%s\n' PILOT_EXECUTION_STARTED
run_pair run 552 stage4-pilot-run
"$mdl" -B -u audit4_native.py --run stage4-pilot --out stage4-pilot/audit4-native.json
printf '%s\n' PILOT_SCORED
