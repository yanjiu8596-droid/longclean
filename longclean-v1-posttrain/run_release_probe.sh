#!/usr/bin/env bash
# Run under tmux on the personal data disk. Only active simulator/model stages use the GPU.
set -Eeuo pipefail
umask 077

lab=$(realpath -e "${1:?personal lab}")
primary=$(realpath -e "${2:?primary run}")
development=$(realpath -e "${3:?development run}")
out=$(realpath -e "${4:?overnight output}")
uuid=${5:?authorized GPU UUID}
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pinned="$primary/src-torch-state"
for path in "$primary" "$development" "$out"; do
    case "$path" in "$lab"/*) ;; *) exit 2 ;; esac
done
[[ "$uuid" == GPU-* && "$primary" != "$out" && "$development" != "$out" && "$src" == "$out/src" ]]
mkdir -p "$out"/{logs,tmp,cache,transfer}
cd "$out"
exec 9>"$out/controller.lock"
flock -n 9
exec >> "$out/controller.log" 2>&1

gpu_sample() {
    nvidia-smi -i "$uuid" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi -i "$uuid" --query-compute-apps=used_memory --format=csv,noheader
}

finish() {
    local code=$? name tag=release-probe-evidence.tar.gz
    trap - EXIT
    printf '%s\n' "$code" > "$out/controller.exit"
    printf '%s CONTROLLER_EXIT %s\n' "$(date -Is)" "$code"
    gpu_sample || true
    local files=(controller.log controller.exit logs src)
    for name in protocol.json probe-summary.json eval formal; do
        if [[ -e "$name" ]]; then files+=("$name"); fi
    done
    tar -czf "transfer/$tag.pending" "${files[@]}"
    mv -- "transfer/$tag.pending" "transfer/$tag"
    (cd transfer && sha256sum "$tag" > "$tag.sha256.pending" && mv -- "$tag.sha256.pending" "$tag.sha256")
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s RELEASE_PROBE_START %s %s\n' "$(date -Is)" "$(hostname)" "$uuid"
test "$(<"$primary/controller.exit")" = 0
test "$(<"$development/controller.exit")" = 0
(cd "$pinned" && sha256sum -c "$primary/logs/source.sha256") > logs/pinned-source.log
(cd "$src" && sha256sum -- *.py *.sh) > logs/runner-source.sha256

model=${TABLECLEAR_MODEL_PYTHON:-"$lab/workspace/.kinex/model-python/bin/python"}
export TABLECLEAR_LAB="$lab"
export PYTHONPATH="$src:$pinned:$primary/python-extra"
export HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch"
export XDG_CACHE_HOME="$out/cache/xdg" TMPDIR="$out/tmp"
export XDG_CONFIG_HOME="$out/cache/config" LIBERO_CONFIG_PATH="$out/cache/libero"
export MPLCONFIGDIR="$out/cache/xdg/matplotlib" ROBOENV_ASSET_CACHE="${ROBOENV_ASSET_CACHE:-$lab/roboenv-assets}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=

args=(--run "$out" --lab "$lab" --primary "$primary" --development "$development" --gpu-uuid "$uuid")
"$model" -B -u "$src/release_probe.py" freeze-protocol "${args[@]}"

stage() {
    local label=$1 limit=$2 index code=0
    shift 2
    if [[ -f "logs/$label.exit" && $(<"logs/$label.exit") == 0 ]]; then return; fi
    printf '%s START %s\n' "$(date -Is)" "$label"
    index=$("$model" -B -c '
import sys
from checkpoint_control import wait_until_idle
print(wait_until_idle(sys.argv[1])["index"])
' "$uuid" | tee -a "logs/$label.gpu-wait.log" | tail -n 1)
    [[ "$index" =~ ^[0-9]+$ ]]
    {
        gpu_sample
        env CUDA_VISIBLE_DEVICES="$uuid" MUJOCO_EGL_DEVICE_ID="$index" TABLECLEAR_EGL_GPU_UUID="$uuid" \
            timeout --kill-after=30s "$limit" "$@" || code=$?
        gpu_sample
    } >> "logs/$label.log" 2>&1
    printf '%s\n' "$code" > "logs/$label.exit"
    printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$code"
    return "$code"
}

stage probe 120m "$model" -B -u "$src/release_probe.py" probe "${args[@]}"
test -s probe-summary.json
if [[ ! -s formal/test-plan.json ]]; then
    printf '%s NO_FORMAL_SCREEN_GATE\n' "$(date -Is)"
    exit 0
fi

observer="$primary/v1-release-observer.py"
test -s "$observer"
cp -- "$observer" formal/v1-release-observer.py
for arm in base rule natural corrected; do
    stage "formal-$arm" 45m "$model" -B -u "$pinned/evaluate.py" "$arm" \
        --run "$out/formal" --lab "$lab" --gpu-uuid "$uuid"
done
"$model" -B -u "$src/release_probe.py" score --run "$out" > logs/formal-summary.log
printf '%s PAIRED_EVALUATION_COMPLETE\n' "$(date -Is)"
