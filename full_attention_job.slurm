#!/usr/bin/env bash

# DKUCC full-attention SFT baseline for MOSS-Transcribe-Diarize (official finetune.py).
# Run only after the smoke job succeeds and the unified train.jsonl exists.
#
# IMPORTANT (measured 2026-09-11):
#   - No --exclusive (account limit: gres/gpu=4 per job; exclusive = whole node = 8 GPUs -> pending).
#   - MaxJobs=1 for this account: only ONE job may run at a time.
#   - For multi-GPU, first extend --gres=gpu:N (up to 4) and use
#       python -m torch.distributed.run --nproc_per_node=N finetune.py ...
#     or MS-Swift --sequence_parallel_size N after validating NCCL on the node.
#
# Submit:  sbatch /work/qt28/moss/dkucc/full_attention_job.slurm

#SBATCH --job-name=moss-full-attn
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err

set -euo pipefail

source /work/qt28/moss/dkucc/env.sh

TRAIN_JSONL="/work/qt28/moss/data/train.jsonl"
OUTPUT_DIR="/work/qt28/moss/checkpoints/full-attention-baseline-${SLURM_JOB_ID}"
MAX_LENGTH="32768"

cd "$MOSS_ROOT"
echo "job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi -L

python finetune.py \
  --train_jsonl "$TRAIN_JSONL" \
  --model_name_or_path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --max_length "$MAX_LENGTH" \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --report_to none \
  --dataloader_num_workers 0
