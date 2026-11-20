#!/usr/bin/env bash
# Use the same pinned native simulator and policy as the original V1 LoRA evaluation.
set -Eeuo pipefail
umask 077

lab=$(realpath -e "$1")
primary=$(realpath -e "$2")
out=$(realpath -e "$3")
uuid=$4
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pinned="$primary/src-torch-state"
case "$primary" in "$lab"/*) ;; *) exit 2 ;; esac
case "$out" in "$lab"/*) ;; *) exit 2 ;; esac
[[ "$uuid" == GPU-* && "$primary" != "$out" ]]
mkdir -p "$out"/{cache,tmp,logs,transfer}
cd "$out"
exec 9>"$out/controller.lock"
flock -n 9
exec >> "$out/controller.log" 2>&1

finish() {
    local code=$? tag=development-evidence.tar.gz name
    trap - EXIT
    printf '%s\n' "$code" > "$out/controller.exit"
    printf '%s DEVELOPMENT_EXIT %s\n' "$(date -Is)" "$code"
    local facts=(controller.log controller.exit logs src)
    for name in geometry.json test-plan.json summary.json eval resets; do
        if [[ -e "$name" ]]; then facts+=("$name"); fi
    done
    tar -czf "transfer/$tag.pending" "${facts[@]}"
    mv -- "transfer/$tag.pending" "transfer/$tag"
    (cd transfer && sha256sum "$tag" > "$tag.sha256.pending" && mv -- "$tag.sha256.pending" "$tag.sha256")
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s DEVELOPMENT_STARTED %s\n' "$(date -Is)" "$(hostname)"
(cd "$pinned" && sha256sum -c "$primary/logs/source.sha256") > logs/pinned-source.log

model=${TABLECLEAR_MODEL_PYTHON:-"$lab/workspace/.kinex/model-python/bin/python"}
sim=${TABLECLEAR_SIM_PYTHON:-"$lab/roboenv/.venv-sim/libero/bin/python"}
export TABLECLEAR_LAB="$lab" TABLECLEAR_SIM_PYTHON="$sim"
export PYTHONPATH="$src:$pinned:$primary/python-extra"
export HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch"
export XDG_CACHE_HOME="$out/cache/xdg" TMPDIR="$out/tmp"
export XDG_CONFIG_HOME="$out/cache/config" LIBERO_CONFIG_PATH="$out/cache/libero"
export MPLCONFIGDIR="$out/cache/xdg/matplotlib" ROBOENV_ASSET_CACHE="${ROBOENV_ASSET_CACHE:-$lab/roboenv-assets}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

sample() {
    nvidia-smi -i "$uuid" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi -i "$uuid" --query-compute-apps=used_memory --format=csv,noheader
}

stage() {
    local label=$1 marker=$2 device=$3
    shift 3
    local index value code=0
    if [[ -e "$marker" && -f "logs/$label.exit" && $(<"logs/$label.exit") == 0 ]]; then return; fi
    printf '%s START %s\n' "$(date -Is)" "$label"
    index=$(CUDA_VISIBLE_DEVICES= "$model" -B -c '
import sys
from checkpoint_control import wait_until_idle
status = wait_until_idle(sys.argv[1])
print(status["index"])
' "$uuid" | tee -a "logs/$label.gpu-wait.log" | tail -n 1)
    [[ "$index" =~ ^[0-9]+$ ]]
    value=$uuid
    if [[ "$device" == ordinal ]]; then value=$index; fi
    {
        sample
        env CUDA_VISIBLE_DEVICES="$value" MUJOCO_EGL_DEVICE_ID="$index" TABLECLEAR_EGL_GPU_UUID="$uuid" \
            timeout --kill-after=30s 45m "$@" || code=$?
        sample
    } >> "logs/$label.log" 2>&1
    printf '%s\n' "$code" > "logs/$label.exit"
    printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$code"
    return "$code"
}

stage profile geometry.json ordinal "$sim" -B -u "$src/layout_difficulty.py" profile --run "$out" --lab "$lab"
stage freeze test-plan.json uuid "$model" -B -u "$src/layout_difficulty.py" freeze \
    --run "$out" --lab "$lab" --primary "$primary" --gpu-uuid "$uuid"
stage baseline done-baseline uuid "$model" -B -u "$pinned/evaluate.py" base --run "$out" --lab "$lab" --gpu-uuid "$uuid"
touch done-baseline
CUDA_VISIBLE_DEVICES= "$model" -B -u "$src/layout_difficulty.py" score --run "$out" | tee logs/summary.log
