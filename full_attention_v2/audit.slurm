#!/usr/bin/env bash
#SBATCH --job-name=moss-v2-audit
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -eo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK=/work/qt28/moss/dkucc/full_attention_v2
RUN=/work/qt28/moss/results/full-attention-v2-audit-${SLURM_JOB_ID}
mkdir -p "$RUN/source" "$RUN/logs"
cp "$TASK"/*.py "$TASK"/*.json "$TASK"/*.slurm "$RUN/source/"
finish() { code=$?; printf '{"exit_code":%s,"finished_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RUN/job_exit.json"; }
trap finish EXIT
cd "$MOSS_ROOT"
python "$RUN/source/qualify.py" --run "$RUN" --previous /work/qt28/moss/results/full-attention-v2-gates-63406 --repeat-run "$1" --audit-only
