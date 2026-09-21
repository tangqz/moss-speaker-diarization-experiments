#!/usr/bin/env bash
#SBATCH --job-name=moss-loop-step30
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail

RUN=/work/qt28/moss/results/ms-swift-63643
OUT="$RUN/evaluations/loop-onset-R8005_M8009"
SRC="$RUN/source/evaluation"
VLLM_PY=/work/qt28/moss/envs/vllm-moss-20260914/bin/python
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:/work/qt28/moss/dkucc/ms_swift_20260914:$SRC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4

rm -f "$OUT/step-30/predictions/sft/alimeeting/test/R8005_M8009.json"
CUDA_VISIBLE_DEVICES=1 "$VLLM_PY" "$SRC/vllm_generate.py" \
  --run "$OUT/step-30" \
  --checkpoint "$RUN/vllm_exports/checkpoint-30" \
  --rank 0 --model-label sft \
  >"$OUT/logs/step-30-gpu1.log" 2>&1

"$VLLM_PY" "$RUN/source-loop-onset/replay_loop_checkpoints.py" summarize > "$OUT/summary.log"
printf '{"complete":true,"target":"R8005_M8009","steps":[30,60,90,120],"step30_gpu_index_within_allocation":1}\n' > "$OUT/complete.json"
