#!/usr/bin/env bash

# DKUCC smoke SFT job for MOSS-Transcribe-Diarize (official finetune.py).
#
# IMPORTANT (measured 2026-09-11):
#   - Do NOT use --exclusive: the qt28 association caps GRES at gpu=4 per job;
#     --exclusive requests the whole node (8 GPUs) and stays pending with
#     reason AssocMaxGRESPerJob.
#   - MaxJobs=1 for this account: only ONE job may run at a time.
#   - Env comes from dkucc/env.sh (module + conda moss312 + proxies + paths).
#
# Submit:  sbatch /work/qt28/moss/dkucc/smoke_job.slurm

#SBATCH --job-name=moss-smoke
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err

set -euo pipefail

source /work/qt28/moss/dkucc/env.sh

TRAIN_JSONL="/work/qt28/moss/data/smoke_train.jsonl"
OUTPUT_DIR="/work/qt28/moss/checkpoints/smoke-${SLURM_JOB_ID}"

cd "$MOSS_ROOT"
echo "job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi -L

python finetune.py \
  --train_jsonl "$TRAIN_JSONL" \
  --model_name_or_path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --max_steps 2 \
  --learning_rate 1e-5 \
  --max_length 8192 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --dataloader_num_workers 0
