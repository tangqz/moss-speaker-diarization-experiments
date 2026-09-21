#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-core
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
ENV=$ROOT/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
RUN=$ROOT/results/rswa-core-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$RUN/source/"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
python "$RUN/source/contract_checks.py" --output "$RUN/contracts.json"
