#!/usr/bin/env bash
set -euo pipefail
card=$1
profile=$2
case "$card" in 2|3) ;; *) exit 2 ;; esac
for _ in $(seq 1 400); do
    if test -s "stage3d-natural-$card.exit" &&
       test "$(nvidia-smi -i "$card" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" = 0 &&
       test -z "$(nvidia-smi -i "$card" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
        test "$(cat "stage3d-natural-$card.exit")" = 0
        exec bash stage3d-run.sh rim "$card" "rim-profiled-blind-$card" --phase blind \
            --profile "$profile" --descent-steps 220 --save-stride 5
    fi
    sleep 3
done
exit 75
