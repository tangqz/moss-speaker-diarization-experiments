#!/usr/bin/env bash
set -euo pipefail
echo QUEUE
squeue -u qt28 -o '%i|%j|%T|%M|%l|%R'
echo GATES
for d in \
  /work/qt28/moss/results/ms-swift-fsdp-check-63506 \
  /work/qt28/moss/results/ms-swift-resume-63512 \
  /work/qt28/moss/results/ms-swift-vllm-check-63520
do
  if test -d "$d"; then
    echo "present|$d"
  else
    echo "missing|$d"
  fi
done
echo SOURCE_HASHES
sha256sum \
  /work/qt28/moss/dkucc/ms_swift_20260914/native_sft.sh \
  /work/qt28/moss/dkucc/ms_swift_20260914/pipeline.py \
  /work/qt28/moss/dkucc/ms_swift_20260914/train.slurm
echo SPACE
df -h /work/qt28/moss | tail -1
echo RUN_STATUS
cat /work/qt28/moss/results/ms-swift-63521/outcome.json
