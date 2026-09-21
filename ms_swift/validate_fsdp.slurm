#!/usr/bin/env bash
#SBATCH --job-name=moss-swift-fsdp
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH" PYTHONUNBUFFERED=1
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/work/qt28/moss/results/ms-swift-fsdp-check-${SLURM_JOB_ID}
mkdir -p "$RUN"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RUN/gpus.csv"
export NPROC_PER_NODE=4 MOSS_STOP_STEP=1
bash "$TASK/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp4" 4 4 \
  --fsdp "$TASK/fsdp2_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false \
  > "$RUN/sp4.log" 2>&1
bash "$TASK/native_sft.sh" "$TASK/data/longest.jsonl" "$RUN/longest-sp4" 4 4 \
  --fsdp "$TASK/fsdp2_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false \
  > "$RUN/longest-sp4.log" 2>&1
printf '{"status":"completed_native_swift_smoke_and_longest","fsdp2":true,"activation_cpu_offload":true}\n' > "$RUN/outcome.json"
