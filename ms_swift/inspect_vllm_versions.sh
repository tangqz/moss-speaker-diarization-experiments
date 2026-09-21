#!/usr/bin/env bash
set -euo pipefail
RUN=/work/qt28/moss/results/ms-swift-63643
OLD=/work/qt28/moss/results/ms-swift-63521

echo '=== installed ==='
/work/qt28/moss/envs/vllm-moss-20260914/bin/python - <<'PY'
import vllm
print(vllm.__version__)
PY

echo '=== current aborted base engines ==='
find "$RUN/evaluations/test-150" -maxdepth 1 -name 'base-worker-*-engine.json' -type f -print -exec cat {} \; 2>/dev/null || true

echo '=== current dev150 engines ==='
find "$RUN/evaluations/dev-150" -maxdepth 1 -name '*engine.json' -type f -print -exec cat {} \; 2>/dev/null || true

echo '=== previous test engines ==='
find "$OLD/evaluations/test-20" -maxdepth 1 -name '*engine.json' -type f -print -exec cat {} \; 2>/dev/null || true
