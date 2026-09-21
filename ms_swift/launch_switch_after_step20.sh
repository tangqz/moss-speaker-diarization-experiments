#!/usr/bin/env bash
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
RUN=/work/qt28/moss/results/ms-swift-63521
nohup bash "$TASK/switch_after_step20.sh" 63521 "$RUN" \
  > "$RUN/logs/switch-after-step20.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN/switch-after-step20.pid"
printf 'WATCHER_PID=%s\n' "$!"
