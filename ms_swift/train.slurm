#!/usr/bin/env bash
#SBATCH --job-name=moss-native-swift
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
RUN=/work/qt28/moss/results/ms-swift-${SLURM_JOB_ID}
export PATH="$ENV/bin:$PATH"
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p "$RUN/source"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
cp "$TASK"/*.py "$TASK"/*.sh "$TASK"/*.json "$TASK"/*.slurm "$RUN/source/"
cp -r "$TASK/evaluation" "$RUN/source/evaluation"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RUN/gpus.csv"
nvidia-smi topo -m > "$RUN/topology.txt"
python "$RUN/source/pipeline.py" --run "$RUN" --gate-run "${MOSS_GATE_RUN:?Provide completed native gate run}" \
    --resume-gate "${MOSS_RESUME_GATE:?Provide successful native resume gate}" \
    --vllm-gate "${MOSS_VLLM_GATE:?Provide successful vLLM loading and long-audio gate}"
