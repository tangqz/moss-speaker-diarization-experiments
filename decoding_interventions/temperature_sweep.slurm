#!/usr/bin/env bash
#SBATCH --job-name=moss-temp-fallback
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=03:00:00
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
printf '{"stage":"temperature_single_sample","complete":false}\n' > "$OUT/status.json"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ${#GPUS[@]} -lt 2 ]]; then
  printf '{"stage":"temperature_gpu_binding_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$VLLM_PY" "$SRC/sweep_worker.py" \
  --out "$OUT" --model-label base --family sampling --gpu 0 \
  >"$OUT/logs/temperature-base.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES="${GPUS[1]}" "$VLLM_PY" "$SRC/sweep_worker.py" \
  --out "$OUT" --model-label sft --family sampling --gpu 0 \
  >"$OUT/logs/temperature-sft.log" 2>&1 &
p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
if [[ $failed -ne 0 ]]; then
  printf '{"stage":"temperature_worker_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi
for name in B1_seed0 B1_seed1 B1_seed2 B2_seed0 B2_seed1 B2_seed2; do
  "$METRIC_PY" "$EVAL/selection.py" --run "$OUT/runs/$name" \
    >"$OUT/logs/score-$name.log" 2>&1 || failed=1
done
if [[ $failed -ne 0 ]]; then
  printf '{"stage":"temperature_score_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
  exit 1
fi
"$HF_PY" "$SRC/gate_sweep.py" --out "$OUT" --groups B1 B2 \
  >"$OUT/logs/temperature-gate.log" 2>&1
gate_code=$?
if [[ $gate_code -eq 0 ]]; then
  if ! full_job=$(sbatch --parsable "$SRC/full_test.slurm"); then
    printf '{"stage":"temperature_full_test_submit_failure","complete":true,"failed":true}\n' > "$OUT/status.json"
    exit 1
  fi
  printf '%s\n' "$full_job" > "$OUT/full_test_job_id.txt"
  printf '{"stage":"temperature_selected_full_test_submitted","complete":true,"failed":false,"full_test_job":"%s"}\n' "$full_job" > "$OUT/status.json"
  exit 0
fi
printf '{"stage":"no_decoding_config_passed","complete":true,"failed":false,"gate_code":%s}\n' "$gate_code" > "$OUT/status.json"
exit 0
