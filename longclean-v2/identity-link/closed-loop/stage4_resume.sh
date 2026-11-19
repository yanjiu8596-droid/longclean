#!/usr/bin/env bash
set -euo pipefail
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
mdl="$lab/workspace/.kinex/model-python/bin/python"

if [[ "${1:-}" == exercise ]]; then
    card=$2
    export AUTODL_TMP="$lab" PYTHONDONTWRITEBYTECODE=1
    source "$lab/longclean/scripts/setup_cache.sh"
    case "$card" in
        2) gpu=${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} ;;
        3) gpu=${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} ;;
        *) exit 2 ;;
    esac
    trap 'rc=$?; printf "%s\n" "$rc" > "stage4-smoke-exercise-$card.exit"' EXIT
    test "$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)" = "$gpu"
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
    nvidia-smi -i "$gpu" | grep -q 'No running processes found'
    read -r mem util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    test "$util" -eq 0 && test "$mem" -le 4
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$card" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    timeout -k 20s 12m "$mdl" -B -u progress_exercise.py --card "$card" --study ../../study4 --output stage4-smoke
    exit
fi

trap 'rc=$?; printf "%s\n" "$rc" > stage4-resume.exit' EXIT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for _ in $(seq 1 250); do
    test ! -s stage4-pipeline.exit || break
    sleep 5
done
test "$(cat stage4-pipeline.exit)" != 0
grep -q '^PARITY ' stage4-pipeline.log
grep -q 'Nonfinite or invalid OSC action' stage4-pipeline.log
"$mdl" -B -u audit4_native.py --run stage4-smoke --out stage4-smoke/audit4-native.json
"$mdl" -B -c 'import json; from pathlib import Path; p=Path("stage4-smoke"); a=json.loads((p/"audit4-native.json").read_text()); b=json.loads((p/"legacy-parity.json").read_text()); assert b["gate"] and a["gate"]["smoke_release_pilot"] and a["gate"]["zero_wrong_candidate_observations_on_both_arms"] and a["freeze_sha256"]==b["smoke_freeze_sha256"]; assert not any(r["arms"]["candidate"]["soup_writes"] for r in a["cases"])'
printf '%s\n' SMOKE_PARITY_AND_SAFETY_PASS

wait_pair() {
    local phase=$1 mode=$2
    for _ in $(seq 1 552); do
        if test -s "stage4-$phase-$mode-2.exit" && test -s "stage4-$phase-$mode-3.exit"; then
            test "$(cat "stage4-$phase-$mode-2.exit")" = 0
            test "$(cat "stage4-$phase-$mode-3.exit")" = 0
            return
        fi
        sleep 5
    done
    return 124
}

start_pair() {
    for card in 2 3; do
        tmux new-session -d -s "lc4-$2-$1-$card" "bash stage4_pipeline.sh gpu $1 $card $2"
    done
}

for card in 2 3; do
    tmux new-session -d -s "lc4-smoke-exercise-$card" \
        "bash stage4_resume.sh exercise $card > stage4-smoke-exercise-$card.log 2>&1"
done
wait_pair smoke exercise
"$mdl" -B -c 'import json,hashlib; from pathlib import Path; p=Path("stage4-smoke"); h=hashlib.sha256((p/"frozen.json").read_bytes()).hexdigest(); a=[json.loads((p/f"progress-exercise-{i}.json").read_text()) for i in (2,3)]; assert all(x["freeze_sha256"]==h and x["wrong_writes"]==0 for x in a); assert any(x["valid_write"] for x in a), "Supplemental smoke did not exercise an audited, nonterminal soup write"'
printf '%s\n' SUPPLEMENTAL_SMOKE_PASS

start_pair screen pilot
wait_pair pilot screen
"$mdl" -B -u stage4_fixture.py freeze --phase pilot --study ../../study4 --output stage4-pilot
start_pair run pilot
wait_pair pilot run
"$mdl" -B -u audit4_native.py --run stage4-pilot --out stage4-pilot/audit4-native.json
printf '%s\n' PILOT_SCORED
