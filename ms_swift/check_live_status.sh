#!/usr/bin/env bash
set -u

run=/work/qt28/moss/results/ms-swift-63521

echo DEV40_STATUS
find "$run/evaluations/dev-40" -type f -name '*status*.json' -print -exec cat {} \; 2>/dev/null
echo COUNTS
for model in base sft; do
  printf '%s=' "$model"
  find "$run/evaluations/dev-40/predictions/$model" -type f 2>/dev/null | wc -l
done
echo ERRORS
grep -iE 'error|traceback|oom|killed|failed' "$run"/logs/dev-40-rank-*.log 2>/dev/null | tail -20 || true
