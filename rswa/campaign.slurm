#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-formal
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
CAMPAIGN=${MOSS_CAMPAIGN:?A qualified immutable campaign is required}
CODE=$CAMPAIGN/source
test -f "$CAMPAIGN/qualification.json"
export PYTHONPATH="$CODE:$ROOT/MOSS-Transcribe-Diarize"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$ROOT/dkucc/rswa_20260920/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$CAMPAIGN/job-${SLURM_JOB_ID}.log") 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
"$ROOT/envs/ms-swift-20260914/bin/python" "$CODE/campaign.py" --root "$CAMPAIGN" --chain
