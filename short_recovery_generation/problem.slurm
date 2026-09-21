#!/usr/bin/env bash
#SBATCH --job-name=moss-recovery-problem
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/short-recovery-problem-%j.out
#SBATCH --error=/work/qt28/moss/logs/short-recovery-problem-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/short_recovery_generation
ENV=/work/qt28/moss/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914:/work/qt28/moss/dkucc"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
cd "$TASK"
exec srun "$ENV/bin/python" -u "$TASK/problem.py"
