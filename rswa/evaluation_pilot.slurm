#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-eval-pilot
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-eval-pilot-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$TASK"/*.sh "$RUN/source/"
export PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
/work/qt28/moss/envs/ms-swift-20260914/bin/python "$RUN/source/freeze_inputs.py"
/work/qt28/moss/envs/ms-swift-20260914/bin/python "$RUN/source/protocol_contract.py"
/work/qt28/moss/envs/ms-swift-20260914/bin/python "$RUN/source/evaluation_pilot.py" --run "$RUN"
/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python "$RUN/source/analysis_contract.py"
# A tiny forced-history smoke is segregated from every quality prediction.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} /work/qt28/moss/envs/vllm-moss-20260914/bin/python \
 "$RUN/source/performance.py" --run "$RUN/performance-smoke" --window 128 --sample smoke --engineering-tokens 300
