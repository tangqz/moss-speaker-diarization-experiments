#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-long-gate
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
RUN=$ROOT/results/rswa-long-gate-${SLURM_JOB_ID}
HENV=$ROOT/envs/ms-swift-20260914
VENV=$ROOT/envs/vllm-moss-20260914
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.json "$TASK"/*.sh "$RUN/source/"
export PATH="$HENV/bin:$PATH" PYTHONPATH="$RUN/source:$ROOT/MOSS-Transcribe-Diarize"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
python "$RUN/source/prepare_gate_samples.py"
export MOSS_RSWA_WINDOW=256 NPROC_PER_NODE=4 MOSS_STOP_STEP=1
unset MOSS_SAVE_STEP_ONE MOSS_RESUME_AUDIT_CHECKPOINT
bash "$RUN/source/native_sft.sh" "$TASK/data/typical.jsonl" "$RUN/typical-sp4" 4 4 \
 --fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false
# Four independent one-GPU cases, with disjoint GPU ownership inside this job.
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
for INDEX in 0 1 2 3; do
 WINDOW=128; SAMPLE=typical
 if (( INDEX % 2 == 1 )); then WINDOW=256; fi
 if (( INDEX >= 2 )); then SAMPLE=longest; fi
 (
  export CUDA_VISIBLE_DEVICES=${GPUS[$INDEX]}
  CASE="$RUN/$SAMPLE-$WINDOW"
  "$VENV/bin/python" "$RUN/source/inference_gate.py" --backend vllm --window "$WINDOW" --sample "$SAMPLE" --tokens 2050 --run "$CASE"
  "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window "$WINDOW" --sample "$SAMPLE" --tokens 2050 --run "$CASE" --record-only
 ) > "$RUN/case-$SAMPLE-$WINDOW.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))
# Calibrate numerical noise at these same reference lengths and output spans.
pids=()
for INDEX in 0 1; do
 SAMPLE=typical
 if (( INDEX == 1 )); then SAMPLE=longest; fi
 (
  export CUDA_VISIBLE_DEVICES=${GPUS[$INDEX]}
  CASE="$RUN/$SAMPLE-full"
  "$VENV/bin/python" "$RUN/source/inference_gate.py" --backend vllm --window full --sample "$SAMPLE" --tokens 2050 --run "$CASE"
  "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window full --sample "$SAMPLE" --tokens 2050 --run "$CASE" --native-full --record-only
  "$HENV/bin/python" "$RUN/source/inference_gate.py" --backend hf --window full --sample "$SAMPLE" --tokens 2050 --run "$CASE" --record-only
 ) > "$RUN/case-$SAMPLE-full.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))
printf '{"status":"long_traces_recorded","requires_numerical_review":true}\n' > "$RUN/stage.json"
