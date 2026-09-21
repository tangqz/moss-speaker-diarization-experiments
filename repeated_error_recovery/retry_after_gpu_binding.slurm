#!/usr/bin/env bash
#SBATCH --job-name=moss-error-repeat-seed1-r2
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/work/qt28/moss/logs/error-repeat-seed1-r2-%j.out
#SBATCH --error=/work/qt28/moss/logs/error-repeat-seed1-r2-%j.err
set -euo pipefail

DKUCC=/work/qt28/moss/dkucc
PACKAGE=$DKUCC/short_recovery
EVALUATOR=$DKUCC/repeated_error_recovery/evaluate_milestone.py
STATUS_WRITER=$DKUCC/repeated_error_recovery/status.py
ENV=/work/qt28/moss/envs/ms-swift-20260914
SWIFT=$DKUCC/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076
CALIBRATION=$PACKAGE/artifacts/dev_calibration.error-repeat.seed1.json
CONFIG=$PACKAGE/artifacts/config.error-repeat.seed1.json
PLAN=$PACKAGE/artifacts/plan.error-repeat.seed1
GATE=$PACKAGE/artifacts/gate.error-repeat.seed1.json
RUN_ID=sr-v3-error-repeat-seed1-20260916
TRAIN=/work/qt28/moss/results/short-recovery-$RUN_ID/sub_repeat/seed-1
CONTROL=/work/qt28/moss/results/error-repeat-controller-seed1-20260916

cd "$DKUCC"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:$SWIFT:$DKUCC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
ALLOCATED_CUDA=${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}
FIRST_GPU=${ALLOCATED_CUDA%%,*}
if [[ ! -f "$CALIBRATION" || ! -f "$CONFIG" || ! -f "$PLAN/manifest_summary.json" || ! -f "$GATE" ]]; then
  echo "Prepared and source-bound seed-1 artifacts are incomplete" >&2
  exit 2
fi
if [[ -e "$TRAIN" ]]; then
  echo "Refusing to overwrite an existing formal training run" >&2
  exit 2
fi

echo "SLURM_ALLOCATED_CUDA=$ALLOCATED_CUDA"
"$ENV/bin/python" "$STATUS_WRITER" --control "$CONTROL" --stage smoke --step 0
echo STAGE=smoke-r2
CUDA_VISIBLE_DEVICES="$ALLOCATED_CUDA" "$ENV/bin/torchrun" --nproc_per_node=4 \
  -m short_recovery.train --config "$CONFIG" --manifest-summary "$PLAN/manifest_summary.json" \
  --run-id "$RUN_ID-smoke-r2" --engineering-stage G2 --stop-at 1 --launch

previous=0
for step in 30 60 90 120 150; do
  echo "STAGE=train-to-$step"
  "$ENV/bin/python" "$STATUS_WRITER" --control "$CONTROL" --stage training --step "$step"
  resume_args=()
  if [[ "$previous" -gt 0 ]]; then
    resume_args=(--resume "$TRAIN/checkpoint-$previous")
  fi
  CUDA_VISIBLE_DEVICES="$ALLOCATED_CUDA" "$ENV/bin/torchrun" --nproc_per_node=4 \
    -m short_recovery.train --config "$CONFIG" --manifest-summary "$PLAN/manifest_summary.json" \
    --run-id "$RUN_ID" --stop-at "$step" --authorized-pending-gates --launch "${resume_args[@]}"
  echo "STAGE=test-step-$step"
  "$ENV/bin/python" "$STATUS_WRITER" --control "$CONTROL" --stage testing --step "$step"
  set +e
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$ENV/bin/python" "$EVALUATOR" \
    --training-root "$TRAIN" --step "$step"
  gate_rc=$?
  set -e
  if [[ "$gate_rc" -eq 42 ]]; then
    "$ENV/bin/python" "$STATUS_WRITER" --control "$CONTROL" --stage early_stop --step "$step"
    echo "EARLY_STOP=problem sample failed at step $step; later training was not started"
    exit 0
  elif [[ "$gate_rc" -ne 0 ]]; then
    echo "Evaluation infrastructure failed at step $step with exit $gate_rc" >&2
    exit "$gate_rc"
  fi
  previous=$step
done
"$ENV/bin/python" "$STATUS_WRITER" --control "$CONTROL" --stage complete --step 150
echo STAGE=complete-150
