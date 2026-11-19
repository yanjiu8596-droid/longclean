#!/usr/bin/env bash
# Run in tmux. Each GPU stage acquires an idle device after dependency checks.
set -Eeuo pipefail
umask 077

lab=$(realpath "${TABLECLEAR_LAB:?Set TABLECLEAR_LAB to the personal V1 installation}")
run=$(realpath -e "${1:?Supply an existing results directory under TABLECLEAR_LAB}")
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
through=${2:-score}
case "$through" in
    replay-smoke|replay|dataset|train-smoke|freeze|score) ;;
    *) printf 'Unknown stage %s\n' "$through" >&2; exit 2 ;;
esac
case "$run" in
    "$lab"/*) ;;
    *) printf 'The run directory must be on the personal data disk.\n' >&2; exit 2 ;;
esac

wait_seconds=${TABLECLEAR_GPU_WAIT_SECONDS:-28800}
replay_workers=${TABLECLEAR_REPLAY_WORKERS:-2}
sim_python="${TABLECLEAR_SIM_PYTHON:-$lab/roboenv/.venv-sim/libero/bin/python}"
model_python="${TABLECLEAR_MODEL_PYTHON:-$lab/workspace/.kinex/model-python/bin/python}"
archive="${TABLECLEAR_V1_ARCHIVE:-$lab/workspace/results/lab-backbone-v1/review/scene08-night-release-120/night-evidence.zip}"
logs="$run/logs"
mkdir -p "$logs" "$run/cache" "$run/tmp"
cd "$run"
exec 9>"$run/controller.lock"
flock -n 9 || { printf 'This run already has a controller.\n' >&2; exit 21; }
trap 'code=$?; printf "%s\n" "$code" > "$run/controller.exit"; date -Is' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$(git -C "$lab/longclean" rev-parse HEAD)" = 42a8fa2d01a60e82d3da8707b421f6b6d722aad1
test -x "$sim_python"
test -r "$archive"
export TABLECLEAR_LAB="$lab" TABLECLEAR_V1_ARCHIVE="$archive" TABLECLEAR_SIM_PYTHON="$sim_python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
export HF_HOME="$run/cache/huggingface" TORCH_HOME="$run/cache/torch"
export XDG_CACHE_HOME="$run/cache/xdg" TMPDIR="$run/tmp"
export XDG_CONFIG_HOME="$run/cache/config" LIBERO_CONFIG_PATH="$run/cache/libero"
export MPLCONFIGDIR="$run/cache/xdg/matplotlib"
export ROBOENV_ASSET_CACHE="${ROBOENV_ASSET_CACHE:-$lab/roboenv-assets}"
export PYTHONPATH="$run/python-extra${PYTHONPATH:+:$PYTHONPATH}"
printf '%s %s V1 run %s\n' "$(date -Is)" "$(hostname)" "$run"
df -Ph "$run"
free -h | head -2
(cd "$src" && sha256sum -- ./*.py ./*.sh) > "$logs/source.pending"
if [[ -f "$logs/source.sha256" ]]; then
    cmp "$logs/source.sha256" "$logs/source.pending"
else
    mv "$logs/source.pending" "$logs/source.sha256"
fi

# Query devices individually so a failed device cannot block healthy idle cards.
gpu_ready() {
    local index=$1 required_uuid=$2 row apps uuid memory utilization
    row=$(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || return 1
    IFS=',' read -r uuid memory utilization <<< "$row"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    [[ "$uuid" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" == 0 ]] || return 1
    (( memory <= 64 )) || return 1
    [[ -z "$required_uuid" || "$uuid" == "$required_uuid" ]] || return 1
    apps=$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
    [[ -z "$apps" ]] || return 1
    nvidia-smi -i "$index" --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || return 1
    nvidia-smi -i "$index" | awk '$1 == "|" && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ &&
        ($6 != "G" || $7 != "/usr/lib/xorg/Xorg") { busy = 1 } END { exit busy }' || return 1
    device_index=$index
    device_uuid=$uuid
}

wait_for_gpu() {
    local stage=$1 required_uuid=$2 index started=$SECONDS
    while :; do
        local indexes=${TABLECLEAR_ALLOWED_GPU_INDICES:-}
        if [[ -z "$indexes" ]]; then
            indexes=$(nvidia-smi -L | sed -nE 's/^GPU ([0-9]+):.*/\1/p')
        fi
        for index in $indexes; do
            if gpu_ready "$index" "$required_uuid"; then
                sleep 2
                if ! gpu_ready "$index" "$required_uuid"; then continue; fi
                printf '%s %s GPU %s %s\n' "$(date -Is)" "$stage" "$device_index" "$device_uuid"
                return
            fi
        done
        if (( SECONDS - started >= wait_seconds )); then
            printf '%s %s no authorized idle GPU within the wait window\n' "$(date -Is)" "$stage" >&2
            return 20
        fi
        printf '%s %s waiting for an idle GPU\n' "$(date -Is)" "$stage"
        sleep 60
    done
}

gpu_sample() {
    printf '%s GPU %s %s\n' "$(date -Is)" "$device_index" "$device_uuid"
    nvidia-smi -i "$device_uuid" --query-gpu=name,driver_version,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
    nvidia-smi -i "$device_uuid" --query-compute-apps=used_memory --format=csv,noheader
}

stage() {
    local name=$1 device=$2 code=0
    shift 2
    if [[ -f "$logs/$name.exit" ]] && [[ $(<"$logs/$name.exit") == 0 ]]; then
        return
    fi
    if [[ "$device" == cpu ]]; then
        env CUDA_VISIBLE_DEVICES= "$@" >> "$logs/$name.log" 2>&1 || code=$?
    else
        wait_for_gpu "$name" "${device#gpu}" || return "$?"
        gpu_sample >> "$logs/$name.log" 2>&1 || true
        env CUDA_VISIBLE_DEVICES="$device_uuid" MUJOCO_EGL_DEVICE_ID="$device_index" \
            MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$@" >> "$logs/$name.log" 2>&1 || code=$?
        gpu_sample >> "$logs/$name.log" 2>&1 || true
    fi
    printf '%s\n' "$code" > "$logs/$name.exit"
    printf '%s %s exit %s\n' "$(date -Is)" "$name" "$code"
    return "$code"
}

if [[ "$through" != replay-smoke && "$through" != replay ]]; then
    test -x "$model_python"
    stage training-runtime cpu "$model_python" -B "$src/check_runtime.py"
fi
stage replay-smoke gpu "$sim_python" -B "$src/prepare.py" --run "$run" --archive "$archive" --case init44-seed08 --workers 1
if [[ "$through" == replay-smoke ]]; then exit 0; fi
stage replay gpu "$sim_python" -B "$src/prepare.py" --run "$run" --archive "$archive" --workers "$replay_workers"
if [[ "$through" == replay ]]; then exit 0; fi

stage dataset cpu "$model_python" -B "$src/export_dataset.py" --run "$run"
if [[ "$through" == dataset ]]; then exit 0; fi

stage gradients gpu "$model_python" -B "$src/train.py" --run "$run" --arm smoke
stage adapter-rollout gpu "$model_python" -B "$src/evaluate.py" smoke --run "$run"
if [[ "$through" == train-smoke ]]; then exit 0; fi

locked_gpu=
if [[ -f "$run/test-plan.json" ]]; then
    locked_gpu=$("$sim_python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_uuid"])' "$run/test-plan.json")
fi
stage freeze "gpu$locked_gpu" "$model_python" -B "$src/evaluate.py" freeze --run "$run"
if [[ "$through" == freeze ]]; then exit 0; fi

for arm in natural corrected; do
    stage "train-$arm" gpu "$model_python" -B "$src/train.py" --run "$run" --arm "$arm"
done
locked_gpu=$("$sim_python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_uuid"])' "$run/test-plan.json")
for arm in base rule natural corrected; do
    stage "eval-$arm" "gpu$locked_gpu" "$model_python" -B "$src/evaluate.py" "$arm" --run "$run"
done
stage score cpu "$model_python" -B "$src/score.py" --run "$run"
