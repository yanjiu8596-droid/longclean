#!/usr/bin/env bash
set -euo pipefail
lab=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}
mdl="$lab/workspace/.kinex/model-python/bin/python"

idle() {
    for _ in $(seq 1 75); do
        if test "$(nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" = 0 &&
           test -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
            return 0
        fi
        sleep 4
    done
    return 75
}

gpu() {
    local mode=$1 card=$2 phase=$3 out="stage4-$3"
    idle "$card"
    bash stage4_gpu.sh "$mode" "$card" "$phase" "$out" > "$out-$mode-$card.log" 2>&1
}

if [[ "${1:-}" == gpu ]]; then
    shift
    gpu "$@"
    if [[ "$1" == run && "$3" == smoke ]]; then
        gpu legacy "$2" smoke
    fi
    exit
fi

trap 'rc=$?; printf "%s\n" "$rc" > stage4-pipeline.exit' EXIT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

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

pair() {
    for card in 2 3; do
        tmux new-session -d -s "lc4-$2-$1-$card" "bash stage4_pipeline.sh gpu $1 $card $2"
    done
}

wait_pair smoke screen
"$mdl" -B -u stage4_fixture.py freeze --phase smoke --study ../../study4 --output stage4-smoke
pair run smoke
wait_pair smoke run
wait_pair smoke legacy
"$mdl" -B -u legacy_parity.py compare --output stage4-smoke
"$mdl" -B -u audit4.py --run stage4-smoke
"$mdl" -B -c 'import json; from pathlib import Path; p=Path("stage4-smoke"); a=json.loads((p/"audit4.json").read_text()); b=json.loads((p/"legacy-parity.json").read_text()); assert b["gate"] and a["gate"]["smoke_release_pilot"] and a["gate"]["zero_wrong_candidate_observations_on_both_arms"]; assert a["freeze_sha256"]==b["smoke_freeze_sha256"]; assert any(r["arms"]["candidate"]["soup_writes"] for r in a["cases"]), "No applied soup commit in smoke; treatment not exercised"'
printf '%s\n' SMOKE_PASS

pair screen pilot
wait_pair pilot screen
"$mdl" -B -u stage4_fixture.py freeze --phase pilot --study ../../study4 --output stage4-pilot
pair run pilot
wait_pair pilot run
"$mdl" -B -u audit4.py --run stage4-pilot
printf '%s\n' PILOT_SCORED
