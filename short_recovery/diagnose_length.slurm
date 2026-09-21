#!/usr/bin/env bash
#SBATCH --job-name=moss-sr-len-diag
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=/work/qt28/moss/logs/short-recovery-length-diag-%j.out
#SBATCH --error=/work/qt28/moss/logs/short-recovery-length-diag-%j.err
set -euo pipefail
DKUCC=/work/qt28/moss/dkucc
ENV=/work/qt28/moss/envs/ms-swift-20260914
SWIFT=$DKUCC/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:$SWIFT:$DKUCC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DKUCC"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV/bin/torchrun" --nproc_per_node=4 \
  -m debug_short_recovery_length --config "$DKUCC/short_recovery/config.train150.json" \
  --manifest-summary "$DKUCC/short_recovery/artifacts/plan.train150.sub/manifest_summary.json" \
  --run-id sr-v2-sub-150-20260915-r2diag --engineering-stage G2 --stop-at 1 --launch
