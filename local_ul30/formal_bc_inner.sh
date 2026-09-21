#!/usr/bin/env bash
set -euo pipefail

source /work/qt28/moss/dkucc/env.sh
export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:$PATH
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc

run_arm() {
  local arm="$1"
  local port="$2"
  local run_id="$3"
  local log="/work/qt28/moss/dkucc/local_ul30/${run_id}.log"
  srun --exclusive --ntasks=1 --gres=gpu:4 --cpus-per-task=32 \
    bash -lc "source /work/qt28/moss/dkucc/env.sh; export PATH=/work/qt28/moss/envs/ms-swift-20260914/bin:\$PATH; export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076:/work/qt28/moss/dkucc; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True; cd /work/qt28/moss/dkucc; /work/qt28/moss/envs/ms-swift-20260914/bin/torchrun --nproc_per_node=4 --master_port=${port} -m local_ul30.train --config local_ul30/config.${arm}.seed1.json --manifest-summary local_ul30/plan.${arm}.seed1/manifest_summary.json --run-id ${run_id} --stop-at 30 --authorized-pending-gates --launch > ${log} 2>&1"
}

run_arm recovery 29641 ul30-b-seed1 &
pid_b=$!
run_arm ul 29642 ul30-c-seed1 &
pid_c=$!
status=0
wait "$pid_b" || status=1
wait "$pid_c" || status=1
exit "$status"
