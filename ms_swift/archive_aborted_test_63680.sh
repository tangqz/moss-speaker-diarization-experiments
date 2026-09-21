#!/usr/bin/env bash
set -euo pipefail

RUN=/work/qt28/moss/results/ms-swift-63643
ARCHIVE=/work/qt28/moss/dkucc/ms_swift_20260914/aborted-test-63680
test "$RUN" = '/work/qt28/moss/results/ms-swift-63643'
test "$ARCHIVE" = '/work/qt28/moss/dkucc/ms_swift_20260914/aborted-test-63680'
mkdir -p "$ARCHIVE/logs"

[[ ! -e "$RUN/evaluations/test-150" ]] || mv "$RUN/evaluations/test-150" "$ARCHIVE/evaluations-test-150"
[[ ! -e "$RUN/test_events" ]] || mv "$RUN/test_events" "$ARCHIVE/test_events"
[[ ! -e "$RUN/test_progress.json" ]] || mv "$RUN/test_progress.json" "$ARCHIVE/test_progress.json"
[[ ! -e "$RUN/final_test_job_exit.json" ]] || mv "$RUN/final_test_job_exit.json" "$ARCHIVE/final_test_job_exit.json"
for file in "$RUN"/logs/test-base-rank-*.log "$RUN"/logs/test-sft-rank-*.log "$RUN"/logs/final-test-score.log; do
  [[ ! -e "$file" ]] || mv "$file" "$ARCHIVE/logs/"
done
printf '{"job":63680,"reason":"cancelled_after_base_reuse_protocol_check","preserved":true}\n' > "$ARCHIVE/archive.json"
echo "$ARCHIVE"
