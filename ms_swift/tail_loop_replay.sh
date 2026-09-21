#!/usr/bin/env bash
for f in /work/qt28/moss/results/ms-swift-63643/evaluations/loop-onset-R8005_M8009/logs/step-*.log; do
  echo "FILE:$f"
  tail -20 "$f"
done
