#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa2
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=18:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
CAMPAIGN=/work/qt28/moss/results/moss-rswa-r2-20260921
CODE=$CAMPAIGN/source
export PYTHONPATH="$CODE:$ROOT/MOSS-Transcribe-Diarize"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$ROOT/dkucc/rswa_20260920/compiler_cache"
exec > >(tee -a "$CAMPAIGN/job-${SLURM_JOB_ID}.log" "$ROOT/logs/rswa-20260921-console.log" "$ROOT/logs/rswa-20260920-console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$code" "$SLURM_JOB_ID" > "$CAMPAIGN/job-${SLURM_JOB_ID}-exit.json"' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
"$ROOT/envs/ms-swift-20260914/bin/python" "$CODE/round2.py" --window gate
"$ROOT/envs/ms-swift-20260914/bin/python" "$CODE/round2.py" --window 256
"$ROOT/envs/ms-swift-20260914/bin/python" "$CODE/round2.py" --window 512
