#!/usr/bin/env bash
#SBATCH --job-name=moss-eval
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -eo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK=/work/qt28/moss/dkucc/evaluation
RUN=/work/qt28/moss/results/eval-62994-${SLURM_JOB_ID}
METRIC_PY="$TASK/metrics-env/bin/python"
mkdir -p "$RUN/source" "$RUN/logs"
cp "$TASK"/*.py "$TASK"/*.slurm "$TASK"/PROTOCOL.md "$RUN/source/"
finish() { code=$?; printf '{"exit_code":%s,"finished_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RUN/job_exit.json"; }
trap finish EXIT
date -Is
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv | tee "$RUN/gpu_start.csv"
python -c 'import torch,transformers; print(torch.__version__,transformers.__version__); assert torch.cuda.device_count()==4'
python -m pip freeze > "$RUN/inference-packages.txt"
/dkucc/home/qt28/.local/bin/uv pip freeze --python "$METRIC_PY" > "$RUN/metric-packages.txt"
git -C "$MOSS_ROOT" rev-parse HEAD > "$RUN/repo-commit.txt"
cd "$MOSS_ROOT"
"$METRIC_PY" "$TASK/test_metrics.py" | tee "$RUN/logs/metric-tests.log"
python "$TASK/audit_inputs.py" --run "$RUN" | tee "$RUN/logs/input-audit.log"
pids=()
for rank in 0 1 2 3; do
  python "$TASK/infer_worker.py" --run "$RUN" --rank "$rank" > "$RUN/logs/worker-${rank}.out" 2> "$RUN/logs/worker-${rank}.err" &
  pids+=("$!")
done
worker_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_status=1
done
"$METRIC_PY" "$TASK/score.py" --run "$RUN" | tee "$RUN/logs/scoring.log"
test "$worker_status" -eq 0
echo "EVALUATION_COMPLETE run=$RUN"
