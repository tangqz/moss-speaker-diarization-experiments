#!/usr/bin/env bash
#SBATCH --job-name=moss-swift-check
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH"
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
RUN=/work/qt28/moss/results/ms-swift-check-${SLURM_JOB_ID}
mkdir -p "$RUN"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RUN/gpus.csv"
nvidia-smi topo -m > "$RUN/topology.txt"
python "$TASK/adapter_check.py" > "$RUN/adapter_check.log" 2>&1
export MOSS_STOP_STEP=1
CUDA_VISIBLE_DEVICES=0 bash "$TASK/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/single" 1 4 > "$RUN/single.log" 2>&1
NPROC_PER_NODE=4 bash "$TASK/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp4" 4 4 > "$RUN/sp4.log" 2>&1
NPROC_PER_NODE=4 bash "$TASK/native_sft.sh" "$TASK/data/longest.jsonl" "$RUN/longest-sp4" 4 4 > "$RUN/longest-sp4.log" 2>&1
printf '{"status":"completed_native_swift_smoke_and_longest"}\n' > "$RUN/outcome.json"
