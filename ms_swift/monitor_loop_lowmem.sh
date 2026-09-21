#!/usr/bin/env bash
while squeue -h -j 63687 | grep -q .; do
  date -Is
  squeue -h -j 63687 -o '%i %T %M %R'
  f=/work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/logs/step-30-lowmem.log
  if [[ -f "$f" ]]; then
    wc -c "$f"
  fi
  sleep 15
done
echo JOB_DONE
sacct -j 63687 --format=JobID,State,Elapsed,ExitCode -n -P
if [[ -f /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/summary.json ]]; then
  cat /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/summary.json
fi
