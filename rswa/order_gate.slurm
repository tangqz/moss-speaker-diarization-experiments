#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-order-gate
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-order-gate-${SLURM_JOB_ID}
ENV=$ROOT/envs/ms-swift-20260914
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$TASK"/*.sh "$RUN/source/"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
export MOSS_RSWA_WINDOW=128 NPROC_PER_NODE=4 MOSS_STOP_STEP=1
unset MOSS_RESUME_AUDIT_CHECKPOINT MOSS_SAVE_STEP_ONE
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
FSDP=(--fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false)
bash "$RUN/source/native_sft.sh" "$TASK/data/varied.jsonl" "$RUN/training" 4 4 "${FSDP[@]}"
python "$RUN/source/order_audit.py" --training "$RUN/training" --step 1
export MOSS_STOP_STEP=2
bash "$RUN/source/native_sft.sh" "$TASK/data/varied.jsonl" "$RUN/training" 4 4 "${FSDP[@]}" --resume_from_checkpoint "$RUN/training/checkpoint-1"
python "$RUN/source/order_audit.py" --training "$RUN/training" --step 2
