#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1
RUN=$2
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$RUN/source/"
export PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
$ROOT/envs/vllm-moss-20260914/bin/python "$RUN/source/performance.py" --run "$RUN/check" --sample smoke --window 128 --engineering-tokens 300
