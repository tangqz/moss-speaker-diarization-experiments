#!/usr/bin/env bash
#SBATCH --job-name=moss-sft-design
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:45:00
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
cp "$TASK/probes.py" "$TASK/probes.slurm" "$RUN/source/"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$RUN/gpu-before.csv"
nvidia-smi topo -m > "$RUN/topology.txt"
python "$TASK/probes.py" contracts --run "$RUN" --device 0 > "$RUN/contracts.out" 2> "$RUN/contracts.err" &
P0=$!
python "$TASK/probes.py" replay --run "$RUN" --device 1 > "$RUN/replay.out" 2> "$RUN/replay.err" &
P1=$!
python "$TASK/probes.py" pilot --precision bf16 --run "$RUN" --device 2 > "$RUN/pilot-bf16.out" 2> "$RUN/pilot-bf16.err" &
P2=$!
python "$TASK/probes.py" pilot --precision fp32 --run "$RUN" --device 3 > "$RUN/pilot-fp32.out" 2> "$RUN/pilot-fp32.err" &
P3=$!
STATUS=0
for PID in "$P0" "$P1" "$P2" "$P3"; do
  wait "$PID" || STATUS=1
done
printf '%s\n' "$STATUS" > "$RUN/exit-status.txt"
exit "$STATUS"
