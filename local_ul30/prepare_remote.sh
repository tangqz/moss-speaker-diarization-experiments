#!/usr/bin/env bash
set -euo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:$PATH
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc
mkdir -p /work/qt28/moss/dkucc/local_ul30/plan.recovery.seed1
srun --job-name=ul30-prep --partition=common-gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
  /work/qt28/moss/envs/ms-swift-20260914/bin/python -m local_ul30.prepare manifest \
  --config /work/qt28/moss/dkucc/local_ul30/config.recovery.seed1.json \
  --output-dir /work/qt28/moss/dkucc/local_ul30/plan.recovery.seed1
