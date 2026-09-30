#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

config="${1:-config.example.json}"

if (( $# > 0 )); then
    shift
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
    num_gpus="$NUM_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "-1" ]]; then
    visible="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    IFS=',' read -r -a gpu_ids <<< "$visible"
    num_gpus="${#gpu_ids[@]}"
else
    num_gpus="$(nvidia-smi -L | wc -l)"
fi

if (( num_gpus < 1 )); then
    echo "No CUDA GPU detected" >&2
    exit 1
fi

launch_args=(--num_processes "$num_gpus")

if (( num_gpus > 1 )); then
    launch_args+=(--multi_gpu)
fi

accelerate launch \
    "${launch_args[@]}" \
    train.py \
    --config "$config" \
    "$@"
