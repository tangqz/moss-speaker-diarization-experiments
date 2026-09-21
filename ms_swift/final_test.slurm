#!/usr/bin/env bash
#SBATCH --job-name=moss-ms-swift-final-test
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
RUN=${MOSS_RUN:?Provide completed MS-Swift run}
export PATH="$ENV/bin:$PATH"
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p "$RUN/logs" "$RUN/source-final-test"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/final_test_job_exit.json"' EXIT
cp "$TASK/final_test.py" "$TASK/final_test.slurm" "$RUN/source-final-test/"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RUN/final_test_gpus.csv"
python "$TASK/final_test.py"
