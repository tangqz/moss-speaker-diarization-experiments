#!/usr/bin/env bash
#SBATCH --job-name=moss-decode-full-test
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -uo pipefail

SRC=/work/qt28/moss/dkucc/decoding_interventions_20260915
SWEEP=/work/qt28/moss/results/ms-swift-63643/decoding-interventions-20260915
RUN="$SWEEP/full-test"
VLLM_PY=/work/qt28/moss/envs/vllm-moss-20260914/bin/python
export PATH=/work/qt28/moss/envs/vllm-moss-20260914/bin:$PATH
HF_PY=/work/qt28/moss/envs/ms-swift-20260914/bin/python
METRIC_PY=/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python
EVAL=/work/qt28/moss/results/ms-swift-63643/source/evaluation

export MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$EVAL:$SRC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$RUN/logs" "$RUN/source"
cp "$SRC"/*.py "$SRC"/*.slurm "$RUN/source/"
if ! "$HF_PY" "$SRC/prepare_full_test.py" --sweep "$SWEEP" --run "$RUN" >"$RUN/logs/prepare.log" 2>&1; then
  printf '{"stage":"prepare_failure","complete":true,"failed":true}\n' > "$RUN/status.json"
  exit 1
fi
printf '{"stage":"base","complete":false}\n' > "$RUN/status.json"

failed=0
for label in base sft; do
  printf '{"stage":"%s","complete":false}\n' "$label" > "$RUN/status.json"
  pids=()
  for rank in 0 1 2 3; do
    "$VLLM_PY" "$SRC/full_test_worker.py" --run "$RUN" --model-label "$label" --rank "$rank" \
      >"$RUN/logs/$label-rank-$rank.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ $failed -ne 0 ]]; then
    printf '{"stage":"%s_worker_failure","complete":true,"failed":true}\n' "$label" > "$RUN/status.json"
    exit 1
  fi
done

printf '{"stage":"scoring","complete":false}\n' > "$RUN/status.json"
"$METRIC_PY" "$EVAL/selection.py" --run "$RUN" >"$RUN/logs/score.log" 2>&1 || failed=1
"$HF_PY" "$SRC/summarize_full.py" --run "$RUN" >"$RUN/logs/summarize.log" 2>&1 || failed=1
if [[ $failed -eq 0 ]]; then
  printf '{"stage":"complete","complete":true,"failed":false}\n' > "$RUN/status.json"
else
  printf '{"stage":"summary_failure","complete":true,"failed":true}\n' > "$RUN/status.json"
fi
tar -czf "$SWEEP/full-test.tar.gz" -C "$SWEEP" full-test
exit "$failed"
