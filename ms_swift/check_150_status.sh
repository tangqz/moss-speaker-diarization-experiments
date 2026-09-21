#!/usr/bin/env bash
set -u

RUN=/work/qt28/moss/results/ms-swift-63643
TASK=/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915_cadence30

echo '=== remote time ==='
date -Is

echo '=== Slurm queue ==='
squeue -j 63643,63645 -o '%.18i %.12T %.10M %.10l %.30R' || true

echo '=== Slurm accounting ==='
sacct -j 63643,63645 --format=JobID,State,Elapsed,Start,End,ExitCode,AllocTRES%50 -X -n -P || true

echo '=== task state ==='
cat "$TASK/current_run.json" 2>/dev/null || true

echo '=== run status ==='
cat "$RUN/status.json" 2>/dev/null || true

echo '=== cadence handoff ==='
cat "$RUN/cadence30_dispatch.json" 2>/dev/null || true
cat "$RUN/cadence30_resume_receipt.json" 2>/dev/null || true

echo '=== checkpoints ==='
find "$RUN/training" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V
du -sh "$RUN"/training/checkpoint-* 2>/dev/null | sort -V || true

echo '=== evaluation receipts ==='
find "$RUN/evaluations" -maxdepth 3 -type f \( -name 'scoring_complete.json' -o -name 'generation_complete.json' -o -name 'status.json' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -30 | cut -d' ' -f2-

echo '=== latest logs ==='
find "$RUN" -maxdepth 4 -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' -o -name 'logging.jsonl' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -12 | cut -d' ' -f2- | while IFS= read -r file; do
  echo "--- $file"
  tail -8 "$file" 2>/dev/null || true
done

echo '=== recent files ==='
find "$RUN" -maxdepth 4 -type f -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -n | tail -25 | cut -d' ' -f2-
