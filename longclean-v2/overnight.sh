#!/usr/bin/env bash
set -euo pipefail

src=$(cd "$(dirname "$0")" && pwd -P)
run=$(cd "$1" && pwd -P)
cd "$src"
exec >> "$run/overnight.log" 2>&1
stage=fixture_screening
finish() {
    status=$?
    printf '%s %s %s\n' "$(date -Is)" "$stage" "$status" > "$run/overnight.status.pending"
    mv "$run/overnight.status.pending" "$run/overnight.status"
    nvidia-smi -i 2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
}
trap finish EXIT

wait_workers() {
    local phase=$1 prefix=$2 card
    for card in 2 3; do
        while [ ! -s "$run/$phase-$card.exit" ]; do
            tmux has-session -t "$prefix$card"
            sleep 20
        done
    done
}

start_workers() {
    local phase=$1 prefix=$2
    tmux new-session -d -s "${prefix}2" "bash '$src/start.sh' '$phase' 2 ${TABLECLEAR_GPU_UUID_2:?Set TABLECLEAR_GPU_UUID_2} '$run'"
    tmux new-session -d -s "${prefix}3" "bash '$src/start.sh' '$phase' 3 ${TABLECLEAR_GPU_UUID_3:?Set TABLECLEAR_GPU_UUID_3} '$run'"
}

sha256sum -c "$run/pre-pilot-source.sha256"
wait_workers fixture lc-v2-fixture
if [ "$(cat "$run/fixture-2.exit")" != 0 ] || [ "$(cat "$run/fixture-3.exit")" != 0 ]; then
    stage=extended_fixture_screening
    python3 -B - "$run" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
p = root / 'plan.json'
old = p.read_bytes()
plan = json.loads(old)
end = max(row['layout_seed'] for row in plan['fixture']) + 1
assert end == 924132
(root / 'plan-before-night-extension.json').write_bytes(old)
plan['fixture'].extend({'layout_seed': seed, 'card': 2 + seed % 2} for seed in range(end, end + 64))
plan['screening_extension'] = 'Up to 64 additional prespecified native reset seeds, before any pilot arm.'
pending = p.with_suffix('.pending')
pending.write_text(json.dumps(plan, indent=2) + '\n')
pending.replace(p)
PY
    mkdir "$run/fixture-before-night-extension"
    for card in 2 3; do
        mv "$run/fixture-$card.log" "$run/fixture-$card.exit" "$run/fixture-before-night-extension/"
    done
    start_workers fixture lc-v2-night-fixture
    wait_workers fixture lc-v2-night-fixture
fi
test "$(cat "$run/fixture-2.exit")" = 0
test "$(cat "$run/fixture-3.exit")" = 0

stage=freeze
python3 -B audit.py freeze --run "$run"
python3 -B - "$run" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
cases = json.loads((root / 'frozen.json').read_text())['cases']
layouts = {row['layout_seed'] for row in cases}
assert len(cases) == 24 and len(layouts) == 8
raws = {json.loads((root / 'fixtures' / str(seed) / 'prefix.json').read_text())['initial']['raw_sha256']
        for seed in layouts}
assert len(raws) == 8
PY

stage=pilot
start_workers pilot lc-v2-night-pilot
wait_workers pilot lc-v2-night-pilot
test "$(cat "$run/pilot-2.exit")" = 0
test "$(cat "$run/pilot-3.exit")" = 0

stage=archive
python3 -B audit.py summarize --run "$run"
(cd "$run" && sha256sum -c evidence.sha256)
stage=complete
