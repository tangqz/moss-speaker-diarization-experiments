#!/usr/bin/env bash
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=/work/qt28/moss/dkucc/local_ul30/slurm-%j.out
#SBATCH --error=/work/qt28/moss/dkucc/local_ul30/slurm-%j.err
set -euo pipefail

arm="$1"
port="$2"
run_id="$3"
source /work/qt28/moss/dkucc/env.sh
export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:$PATH
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /work/qt28/moss/dkucc
/work/qt28/moss/envs/ms-swift-20260914/bin/torchrun --nproc_per_node=4 --master_port="$port" \
  -m local_ul30.train \
  --config "local_ul30/config.${arm}.seed1.json" \
  --manifest-summary "local_ul30/plan.${arm}.seed1/manifest_summary.json" \
  --run-id "$run_id" --stop-at 30 --authorized-pending-gates --launch
