#!/usr/bin/env bash
set -euo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:$PATH
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc
srun --job-name=ul30-c-smoke --partition=common-gpu --gres=gpu:4 --cpus-per-task=32 --mem=384G --time=01:00:00 \
  bash -lc 'source /work/qt28/moss/dkucc/env.sh; export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:$PATH; export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True; cd /work/qt28/moss/dkucc; /work/qt28/moss/envs/ms-swift-20260914/bin/torchrun --nproc_per_node=4 --master_port=29631 -m local_ul30.train --config local_ul30/config.ul.seed1.json --manifest-summary local_ul30/plan.ul.seed1/manifest_summary.json --run-id c-smoke-seed1 --engineering-stage G2 --stop-at 1 --launch'
