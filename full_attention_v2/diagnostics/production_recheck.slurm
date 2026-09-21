#!/usr/bin/env bash
#SBATCH --job-name=moss-v2-prod
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -eo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN=/work/qt28/moss/results/full-attention-v2-diagnostic-${SLURM_JOB_ID}
mkdir -p "$RUN"
finish() { code=$?; printf '{"exit_code":%s,"finished_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RUN/job_exit.json"; }
trap finish EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv > "$RUN/gpu_start.csv"
python -c 'import torch; assert torch.cuda.device_count()==4; assert all(torch.cuda.mem_get_info(i)[0]>16*1024**3 for i in range(4)), "Insufficient free GPU memory for full generation"'
cd "$MOSS_ROOT"
python /work/qt28/moss/dkucc/full_attention_v2_production_recheck/production_recheck.py
