#!/usr/bin/env bash
#SBATCH --job-name=moss-sft-post
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN=/work/qt28/moss/results/sft-design-${SLURM_JOB_ID}
TASK=/work/qt28/moss/dkucc/next_sft
mkdir -p "$RUN/source"
cp "$TASK/post_pilot.py" "$TASK/post_pilot.slurm" "$TASK/probes.py" "$RUN/source/"
python "$TASK/post_pilot.py" --pilot-run /work/qt28/moss/results/sft-design-63401 --run "$RUN" > "$RUN/post-pilot.out" 2> "$RUN/post-pilot.err"
