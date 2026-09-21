#!/usr/bin/env bash
while squeue -h -j 63686 | grep -q .; do
  date -Is
  squeue -h -j 63686 -o '%i %T %M %R'
  for f in /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/logs/step-30-retry.log; do
    if [[ -f "$f" ]]; then
      wc -c "$f"
    fi
  done
  sleep 15
done
echo JOB_DONE
sacct -j 63686 --format=JobID,State,Elapsed,ExitCode -n -P
if [[ -f /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/summary.json ]]; then
  cat /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/summary.json
fi
