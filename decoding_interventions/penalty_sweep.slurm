#!/usr/bin/env bash
#SBATCH --job-name=moss-penalty-sweep
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -uo pipefail

SRC=/work/qt28/moss/dkucc/decoding_interventions_20260915
OUT=/work/qt28/moss/results/ms-swift-63643/decoding-interventions-20260915
VLLM_PY=/work/qt28/moss/envs/vllm-moss-20260914/bin/python
export PATH=/work/qt28/moss/envs/vllm-moss-20260914/bin:$PATH
HF_PY=/work/qt28/moss/envs/ms-swift-20260914/bin/python
METRIC_PY=/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python
EVAL=/work/qt28/moss/results/ms-swift-63643/source/evaluation

export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$EVAL:$SRC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$OUT/logs" "$OUT/source"
cp "$SRC"/*.py "$SRC"/*.slurm "$SRC"/*.md "$OUT/source/"
if ! "$HF_PY" "$SRC/prepare_sweep.py" --out "$OUT" >"$OUT/logs/prepare.log" 2>&1; then
  printf '{"stage":"prepare_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi
printf '{"stage":"penalty_single_sample","complete":false}\n' > "$OUT/status.json"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ${#GPUS[@]} -lt 2 ]]; then
  printf '{"stage":"gpu_binding_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$VLLM_PY" "$SRC/sweep_worker.py" \
  --out "$OUT" --model-label base --family penalty --gpu 0 \
  >"$OUT/logs/penalty-base.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES="${GPUS[1]}" "$VLLM_PY" "$SRC/sweep_worker.py" \
  --out "$OUT" --model-label sft --family penalty --gpu 0 \
  >"$OUT/logs/penalty-sft.log" 2>&1 &
p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
if [[ $failed -ne 0 ]]; then
  printf '{"stage":"penalty_worker_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi

for name in A0_seed0 A1_seed0 A2_seed0 A3_seed0; do
  "$METRIC_PY" "$EVAL/selection.py" --run "$OUT/runs/$name" \
    >"$OUT/logs/score-$name.log" 2>&1 || failed=1
done
if [[ $failed -ne 0 ]]; then
  printf '{"stage":"penalty_score_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi

"$HF_PY" "$SRC/gate_sweep.py" --out "$OUT" --groups A1 A2 A3 \
  >"$OUT/logs/penalty-gate.log" 2>&1
gate_code=$?
if [[ $gate_code -eq 0 ]]; then
  if ! full_job=$(sbatch --parsable "$SRC/full_test.slurm"); then
    printf '{"stage":"full_test_submit_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
    exit 1
  fi
  printf '%s\n' "$full_job" > "$OUT/full_test_job_id.txt"
  printf '{"stage":"penalty_selected_full_test_submitted","complete":true,"failed":false,"full_test_job":"%s"}\n' "$full_job" > "$OUT/status.json"
  exit 0
fi
if [[ $gate_code -eq 3 ]]; then
  if ! fallback_job=$(sbatch --parsable "$SRC/temperature_sweep.slurm"); then
    printf '{"stage":"temperature_submit_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
    exit 1
  fi
  printf '%s\n' "$fallback_job" > "$OUT/temperature_fallback_job_id.txt"
  printf '{"stage":"penalty_failed_temperature_submitted","complete":true,"failed":false,"temperature_job":"%s"}\n' "$fallback_job" > "$OUT/status.json"
  exit 0
fi
printf '{"stage":"penalty_gate_error","complete":true,"failed":true,"gate_code":%s}\n' "$gate_code" > "$OUT/status.json"
exit "$gate_code"
