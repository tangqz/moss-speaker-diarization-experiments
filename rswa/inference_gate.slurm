#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-infer-gate
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-infer-gate-${SLURM_JOB_ID}
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$RUN/source/"
export PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used --format=csv
VENV=$ROOT/envs/vllm-moss-20260914
HENV=$ROOT/envs/ms-swift-20260914
if [[ "${RSWA_SKIP_INFERENCE_CONTRACTS:-0}" != 1 ]]; then
 "$VENV/bin/python" "$RUN/source/vllm_contract.py" --output "$RUN/vllm_contract.json"
fi
for WINDOW in full 128 256; do
 "$VENV/bin/python" "$RUN/source/inference_gate.py" --backend vllm --window "$WINDOW" --run "$RUN/$WINDOW"
 if [[ "$WINDOW" == full ]]; then
  "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window "$WINDOW" --run "$RUN/$WINDOW" --native-full --record-only
 fi
 "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window "$WINDOW" --run "$RUN/$WINDOW" --record-only
done
"$HENV/bin/python" "$RUN/source/alignment_calibration.py" --run "$RUN"
printf '{"status":"alignment_recorded","passed":false,"requires_numerical_review":true}\n' > "$RUN/stage.json"
