#!/usr/bin/env bash
#SBATCH --job-name=moss-short-recovery
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/work/qt28/moss/logs/short-recovery-%j.out
#SBATCH --error=/work/qt28/moss/logs/short-recovery-%j.err
set -euo pipefail

DKUCC=/work/qt28/moss/dkucc
ROOT=$DKUCC/short_recovery
SWIFT=/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076
ENV=/work/qt28/moss/envs/ms-swift-20260914
CONFIG=${RECOVERY_CONFIG:?set RECOVERY_CONFIG to an arm-specific frozen JSON file}
SUMMARY=${RECOVERY_MANIFEST_SUMMARY:?set RECOVERY_MANIFEST_SUMMARY}
RUN_ID=${RECOVERY_RUN_ID:?set RECOVERY_RUN_ID}
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:$SWIFT:$DKUCC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DKUCC"

if [[ "${RECOVERY_LAUNCH:-0}" != 1 ]]; then
  python -m short_recovery.preflight --config "$CONFIG" dry-run --output "$ROOT/artifacts/slurm-${SLURM_JOB_ID:-manual}-dry-run.json"
  exit 0
fi

# No sbatch is called here. A human explicitly submits this file after G0-G5.
EXTRA_ARGS=()
if [[ -n "${RECOVERY_RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RECOVERY_RESUME")
fi
EXTRA_ARGS+=(--stop-at "${RECOVERY_STOP_AT:-150}")
if [[ "${RECOVERY_AUTHORIZED_PENDING_GATES:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--authorized-pending-gates)
fi
srun --ntasks=1 --kill-on-bad-exit=1 "$ENV/bin/torchrun" --nproc_per_node=4 --master_port="${MASTER_PORT:-29615}" \
  -m short_recovery.train --config "$CONFIG" --manifest-summary "$SUMMARY" --run-id "$RUN_ID" --launch "${EXTRA_ARGS[@]}"
