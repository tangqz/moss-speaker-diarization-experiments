#!/usr/bin/env bash
#SBATCH --job-name=moss-swift-resume
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
RUN=/work/qt28/moss/results/ms-swift-resume-${SLURM_JOB_ID}
mkdir -p "$RUN"
export PATH="$ENV/bin:$PATH"
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
export MOSS_IMPLICIT_CAUSAL=1 CELOSS_PARALLEL_SIZE=512
MOSS_STOP_STEP=1 NPROC_PER_NODE=4 bash "$TASK/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp4" 4 4 \
    --val_dataset "$TASK/data/smoke.jsonl" --eval_strategy steps --eval_steps 1 \
    --fsdp "$TASK/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false \
    > "$RUN/sp4.log" 2>&1
/dkucc/home/qt28/envs/moss312/bin/python "$TASK/verify_saved.py" --run "$RUN" --mode production > "$RUN/production_loading_check.log" 2>&1
MOSS_STOP_STEP=2 NPROC_PER_NODE=4 bash "$TASK/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/resume-sp4" 4 4 \
    --resume_from_checkpoint "$RUN/sp4/checkpoint-1" --val_dataset "$TASK/data/smoke.jsonl" \
    --eval_strategy steps --eval_steps 1 --eval_on_start true \
    --fsdp "$TASK/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false \
    > "$RUN/resume-sp4.log" 2>&1
python "$TASK/verify_resume.py" "$RUN" > "$RUN/resume_check.log" 2>&1
