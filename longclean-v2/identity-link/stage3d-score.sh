#!/usr/bin/env bash
set -euo pipefail
result=stage3d-score.exit
mdl=${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to your personal data directory}/workspace/.kinex/model-python/bin/python
trap 'code=$?; printf "%s\n" "$code" > "$result"' EXIT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for _ in $(seq 1 410); do
    if test -s rim-profiled-blind-2.exit && test -s rim-profiled-blind-3.exit; then break; fi
    sleep 3
done
test "$(cat rim-profiled-blind-2.exit)" = 0
test "$(cat rim-profiled-blind-3.exit)" = 0
"$mdl" -B -u freeze_stage3d.py stage3d-plan.json stage3d-freeze.json --with-rim
"$mdl" -B -u stage3d_gate.py stage3d-freeze.json stage3d-preflight.json --preflight
if "$mdl" -B -c 'import json,sys; sys.exit(not json.load(open("stage3d-preflight.json"))["preflight"]["ready_for_scoring"])'; then
    timeout 8m "$mdl" -B -u stage3d_gate.py stage3d-freeze.json stage3d-independent.json
else
    printf '%s\n' 'Coverage not met; independent sensor and action data stay unopened.'
fi
