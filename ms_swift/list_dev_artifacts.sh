#!/usr/bin/env bash
set -u
run=/work/qt28/moss/results/ms-swift-63521
for step in 15 20 30 40; do
  echo "STEP:$step"
  find "$run/evaluations/dev-$step" -maxdepth 2 -type f -printf '%P\n' 2>/dev/null | sort | head -100
done
