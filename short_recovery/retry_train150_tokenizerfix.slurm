#!/usr/bin/env bash
#SBATCH --job-name=moss-sr-sub-150-r3
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/work/qt28/moss/logs/short-recovery-sub150-r3-%j.out
#SBATCH --error=/work/qt28/moss/logs/short-recovery-sub150-r3-%j.err
set -euo pipefail
DKUCC=/work/qt28/moss/dkucc
PACKAGE=$DKUCC/short_recovery
ENV=/work/qt28/moss/envs/ms-swift-20260914
SWIFT=$DKUCC/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076
PROPOSED=$PACKAGE/config.train150.proposed.json
CALIBRATION=$PACKAGE/artifacts/dev_calibration.train150.r3.json
CONFIG=$PACKAGE/config.train150.r3.json
PLAN=$PACKAGE/artifacts/plan.train150.sub.r3
GATE=$PACKAGE/artifacts/gate.train150.r3.json
RUN_ID=sr-v2-sub-150-20260916-r3
cd "$DKUCC"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="/work/qt28/moss/MOSS-Transcribe-Diarize:$SWIFT:$DKUCC"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
echo STAGE=calibration
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" -m short_recovery.calibrate --config "$PROPOSED" --output "$CALIBRATION"
"$ENV/bin/python" -m short_recovery.finalize_calibration --proposed "$PROPOSED" --calibration "$CALIBRATION" --output "$CONFIG"
echo STAGE=manifest
"$ENV/bin/python" -m short_recovery.prepare manifest --config "$CONFIG" --output-dir "$PLAN"
"$ENV/bin/python" -m short_recovery.preflight --config "$CONFIG" init-gate --manifest-summary "$PLAN/manifest_summary.json" --output "$GATE"
echo STAGE=smoke
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV/bin/torchrun" --nproc_per_node=4 -m short_recovery.train --config "$CONFIG" --manifest-summary "$PLAN/manifest_summary.json" --run-id "$RUN_ID-smoke" --engineering-stage G2 --stop-at 1 --launch
echo STAGE=formal150
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV/bin/torchrun" --nproc_per_node=4 -m short_recovery.train --config "$CONFIG" --manifest-summary "$PLAN/manifest_summary.json" --run-id "$RUN_ID" --stop-at 150 --authorized-pending-gates --launch
