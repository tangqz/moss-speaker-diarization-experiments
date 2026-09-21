#!/usr/bin/env bash
set -u

echo '=== remote time ==='
date -Is
echo '=== current common-gpu queue ==='
squeue -p common-gpu -o '%.18i %.9P %.12j %.10u %.2t %.10M %.10l %.4D %.24R' | head -40
echo '=== partition nodes ==='
sinfo -p common-gpu -o '%P %a %l %D %t %G'
echo '=== recent user jobs ==='
sacct -u qt28 -S 2026-09-14 --format=JobID,JobName%28,Partition,State,Elapsed,Submit,Start,End -X -n -P | tail -30
