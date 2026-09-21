#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-pilot-finish
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-pilot-finish-${SLURM_JOB_ID}
ENV=$ROOT/envs/ms-swift-20260914
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$TASK"/*.sh "$RUN/source/"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
python "$RUN/source/finish_pilot.py" --run "$RUN" --pilot "$ROOT/results/rswa-eval-pilot-64279"
