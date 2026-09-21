#!/usr/bin/env bash
#SBATCH --job-name=moss-recovery-d0d1
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -uo pipefail

RUN=/work/qt28/moss/results/ms-swift-63643
SRC=/work/qt28/moss/dkucc/recovery_diagnostics_20260915
EXP="$RUN/evaluations/recovery-diagnostics-d0-d1"
HF_PY=/work/qt28/moss/envs/ms-swift-20260914/bin/python
VLLM_PY=/work/qt28/moss/envs/vllm-moss-20260914/bin/python

export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$RUN/source/evaluation:$SRC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p "$EXP/logs" "$EXP/source"
cp "$SRC"/*.py "$SRC"/*.slurm "$SRC"/*.md "$EXP/source/"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv > "$EXP/gpus.csv"
"$HF_PY" "$SRC/prepare_cases.py" --run "$RUN" --out "$EXP" > "$EXP/logs/prepare.log" 2>&1

printf '{"stage":"hf_bf16","complete":false}\n' > "$EXP/status.json"
models=(base step60 step90 step150)
pids=()
for gpu in 0 1 2 3; do
  label=${models[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu "$HF_PY" "$SRC/hf_replay.py" \
    --experiment "$EXP" --model-label "$label" --dtype bf16 --scope d1 --gpu 0 \
    >"$EXP/logs/hf-bf16-$label.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

printf '{"stage":"hf_fp32","complete":false,"prior_failure":%s}\n' "$failed" > "$EXP/status.json"
pids=()
for pair in 0:base 1:step90; do
  gpu=${pair%%:*}
  label=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu "$HF_PY" "$SRC/hf_replay.py" \
    --experiment "$EXP" --model-label "$label" --dtype fp32 --scope d0 --gpu 0 \
    >"$EXP/logs/hf-fp32-$label.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

printf '{"stage":"vllm_bf16","complete":false,"prior_failure":%s}\n' "$failed" > "$EXP/status.json"
pids=()
for gpu in 0 1 2 3; do
  label=${models[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu "$VLLM_PY" "$SRC/vllm_replay.py" \
    --experiment "$EXP" --model-label "$label" --gpu 0 \
    >"$EXP/logs/vllm-bf16-$label.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

"$HF_PY" "$SRC/summarize.py" --experiment "$EXP" > "$EXP/logs/summarize.log" 2>&1 || failed=1
if [[ $failed -eq 0 ]]; then
  printf '{"stage":"complete","complete":true,"failed":false}\n' > "$EXP/status.json"
else
  printf '{"stage":"complete_with_errors","complete":true,"failed":true}\n' > "$EXP/status.json"
fi
tar -czf "$RUN/recovery-diagnostics-d0-d1.tar.gz" -C "$RUN/evaluations" recovery-diagnostics-d0-d1
exit "$failed"
