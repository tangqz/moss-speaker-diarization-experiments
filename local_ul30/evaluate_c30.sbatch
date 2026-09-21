#!/usr/bin/env bash
#SBATCH --job-name=ul30-c-eval
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/dkucc/local_ul30/eval-c30-%j.out
#SBATCH --error=/work/qt28/moss/dkucc/local_ul30/eval-c30-%j.err
set -euo pipefail

DKUCC=/work/qt28/moss/dkucc
ENV=/work/qt28/moss/envs/vllm-moss-20260914
TRAIN=/work/qt28/moss/results/local-ul30-ul30-c-seed1/ul/seed-1
CHECKPOINT="$TRAIN/checkpoint-30"
if [[ ! -f "$CHECKPOINT/recovery_state.json" ]]; then
  echo "C checkpoint-30 is missing: $CHECKPOINT" >&2
  exit 2
fi
source "$DKUCC/env.sh"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:$DKUCC/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:$DKUCC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$DKUCC"
"$ENV/bin/python" "$DKUCC/repeated_error_recovery/evaluate_milestone.py" \
  --training-root "$TRAIN" --step 30
