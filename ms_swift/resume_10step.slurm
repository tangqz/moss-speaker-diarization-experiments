#!/usr/bin/env bash
#SBATCH --job-name=moss-swift-resume10
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
RUN=${MOSS_RESUME_RUN:?Provide the existing formal run}
START=${MOSS_RESUME_STEP:-20}
SOURCE="$RUN/source-resume-${SLURM_JOB_ID}"
export PATH="$ENV/bin:$PATH"
export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p "$SOURCE"
trap 'code=$?; printf "{\"job_id\":%s,\"exit_code\":%s}\n" "$SLURM_JOB_ID" "$code" > "$RUN/resume_job_exit_${SLURM_JOB_ID}.json"' EXIT
cp "$TASK"/*.py "$TASK"/*.sh "$TASK"/*.json "$TASK"/*.slurm "$SOURCE/"
cp -r "$TASK/evaluation" "$SOURCE/evaluation"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RUN/resume_gpus_${SLURM_JOB_ID}.csv"
nvidia-smi topo -m > "$RUN/resume_topology_${SLURM_JOB_ID}.txt"
python "$SOURCE/resume_10step.py" --run "$RUN" --start-step "$START" --interval 10 --stop-step 50
