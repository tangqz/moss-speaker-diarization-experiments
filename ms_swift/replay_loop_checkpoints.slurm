#!/usr/bin/env bash
#SBATCH --job-name=moss-loop-onset
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail

RUN=/work/qt28/moss/results/ms-swift-63643
OUT="$RUN/evaluations/loop-onset-R8005_M8009"
SRC="$RUN/source/evaluation"
VLLM_PY=/work/qt28/moss/envs/vllm-moss-20260914/bin/python
CONTROL="$RUN/source-loop-onset/replay_loop_checkpoints.py"

export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:/work/qt28/moss/dkucc/ms_swift_20260914:$SRC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p "$OUT/logs"

"$VLLM_PY" "$CONTROL" prepare

steps=(30 60 90 120)
pids=()
for gpu in 0 1 2 3; do
  step=${steps[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu "$VLLM_PY" "$SRC/vllm_generate.py" \
    --run "$OUT/step-$step" \
    --checkpoint "$RUN/vllm_exports/checkpoint-$step" \
    --rank 0 --model-label sft \
    >"$OUT/logs/step-$step.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  printf '{"complete":false,"reason":"one_or_more_workers_failed"}\n' > "$OUT/complete.json"
  exit 1
fi

"$VLLM_PY" "$CONTROL" summarize > "$OUT/summary.log"
printf '{"complete":true,"target":"R8005_M8009","steps":[30,60,90,120]}\n' > "$OUT/complete.json"
