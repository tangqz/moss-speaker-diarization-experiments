#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-triton-gate
#SBATCH --partition=common-gpu
#SBATCH --nodelist=dkucc-core-gpu-dkurc-d-it09-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
RUN=$ROOT/results/rswa-triton-gate-${SLURM_JOB_ID}
HENV=$ROOT/envs/ms-swift-20260914
VENV=$ROOT/envs/vllm-moss-20260914
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$RUN/source/"
export PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
for PHASE in 0 1; do
 pids=()
 for INDEX in 0 1 2 3; do
  if (( PHASE == 0 )); then
   SAMPLES=(smoke smoke smoke longest); WINDOWS=(full 128 256 256)
  else
   SAMPLES=(longest longest typical typical); WINDOWS=(full 128 128 256)
  fi
  SAMPLE=${SAMPLES[$INDEX]}; WINDOW=${WINDOWS[$INDEX]}; TOKENS=2050
  if [[ "$SAMPLE" == smoke ]]; then TOKENS=512; fi
  (
   export CUDA_VISIBLE_DEVICES=${GPUS[$INDEX]}
   CASE="$RUN/$SAMPLE-$WINDOW"
   "$VENV/bin/python" "$RUN/source/inference_gate.py" --backend vllm --window "$WINDOW" --sample "$SAMPLE" --tokens "$TOKENS" --run "$CASE" --attention-backend TRITON_ATTN
   if [[ "$WINDOW" == full ]]; then
    "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window "$WINDOW" --sample "$SAMPLE" --tokens "$TOKENS" --run "$CASE" --native-full --record-only
   fi
   "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window "$WINDOW" --sample "$SAMPLE" --tokens "$TOKENS" --run "$CASE" --record-only
  ) > "$RUN/case-$SAMPLE-$WINDOW.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 (( failed == 0 ))
done
printf '{"status":"triton_traces_recorded","requires_numerical_review":true}\n' > "$RUN/stage.json"
