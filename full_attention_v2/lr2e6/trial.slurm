#!/usr/bin/env bash
#SBATCH --job-name=moss-v2-lr2e6
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -eo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK=/work/qt28/moss/dkucc/full_attention_v2_lr2e6_20260913
RUN=/work/qt28/moss/results/full-attention-v2-${SLURM_JOB_ID}
CHECKPOINTS=/work/qt28/moss/checkpoints/full-attention-v2-${SLURM_JOB_ID}
mkdir -p "$RUN/source" "$RUN/logs"
finish() { code=$?; printf '{"exit_code":%s,"finished_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RUN/job_exit.json"; }
trap finish EXIT
cp "$TASK/bundle/"*.py "$TASK/bundle/"*.json "$TASK/bundle/"*.slurm "$TASK/bundle/"*.diff "$RUN/source/"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv > "$RUN/gpu_start.csv"
python -m pip freeze > "$RUN/packages.txt"
cd "$MOSS_ROOT"
python "$RUN/source/trial_run.py" --run "$RUN" --checkpoints "$CHECKPOINTS"
