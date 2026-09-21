#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-full-equivalence
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-full-equivalence-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$RUN/source/"
export PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
/work/qt28/moss/envs/ms-swift-20260914/bin/python "$RUN/source/check_dev_lengths.py"
for SAMPLE in smoke typical; do
 /work/qt28/moss/envs/ms-swift-20260914/bin/python "$RUN/source/full_equivalence.py" --sample "$SAMPLE" --run "$RUN/$SAMPLE"
done
