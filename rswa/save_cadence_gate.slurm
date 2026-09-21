#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-save-cadence
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
ENV=$ROOT/envs/ms-swift-20260914
RUN=$ROOT/results/rswa-save-cadence-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.sh "$TASK"/*.json "$RUN/source/"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MOSS_RSWA_WINDOW=128 NPROC_PER_NODE=4 MOSS_STOP_STEP=2 MOSS_SAVE_STEP_ONE=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/continuous" 4 4 \
 --fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false
python "$RUN/source/compare_resume.py" "$RUN/continuous/checkpoint-2" \
 "$ROOT/results/rswa-resume-gate-64089/resumed/checkpoint-2" --output "$RUN/resume_same_save_cadence.json"
