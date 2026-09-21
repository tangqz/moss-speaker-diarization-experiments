#!/usr/bin/env bash
set -euo pipefail

# Keep both arms in one eight-GPU allocation.  The account limit is 384 GiB,
# so the two four-GPU child steps share that allocation's memory.
srun --job-name=ul30-formal --partition=common-gpu --nodes=1 --ntasks=1 \
  --gres=gpu:8 --cpus-per-task=64 --mem=384G --time=04:00:00 \
  bash /work/qt28/moss/dkucc/local_ul30/formal_bc_inner.sh
