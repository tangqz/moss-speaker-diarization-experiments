#!/usr/bin/env bash
set -euo pipefail
for file in /work/qt28/moss/results/ms-swift-63643/logs/dev-0-rank-*.log
do
  echo "FILE:$file"
  tail -n 8 "$file"
done
