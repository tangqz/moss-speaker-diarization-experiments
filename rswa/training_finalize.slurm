#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-resume-gate
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
ENV=$ROOT/envs/ms-swift-20260914
RUN=$ROOT/results/rswa-resume-gate-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.sh "$TASK"/*.json "$RUN/source/"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MOSS_RSWA_WINDOW=128 NPROC_PER_NODE=4 MOSS_STOP_STEP=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp4-ddp" 4 4
FSDP=(--fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false)
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/resumed" 4 4 "${FSDP[@]}"
export MOSS_STOP_STEP=2
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/resumed" 4 4 "${FSDP[@]}" --resume_from_checkpoint "$RUN/resumed/checkpoint-1"
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/continuous" 4 4 "${FSDP[@]}"
python "$RUN/source/compare_resume.py" "$RUN/continuous/checkpoint-2" "$RUN/resumed/checkpoint-2" --output "$RUN/resume_contract.json"
printf '{"passed":true}\n' > "$RUN/stage.json"
