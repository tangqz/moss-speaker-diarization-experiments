#!/usr/bin/env bash
#SBATCH --job-name=moss-rswa-train-gate
#SBATCH --partition=common-gpu
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
ENV=$ROOT/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MOSS_ROOT="$ROOT/MOSS-Transcribe-Diarize" MOSS_IMPLICIT_CAUSAL=0 MOSS_RSWA_WINDOW=128
export TORCHINDUCTOR_CACHE_DIR="$TASK/compiler_cache"
RUN=$ROOT/results/rswa-train-gate-${SLURM_JOB_ID}
export RSWA_GATE_RUN="$RUN"
mkdir -p "$RUN/source"
cp "$TASK"/*.py "$TASK"/*.sh "$TASK"/*.json "$RUN/source/"
export PYTHONPATH="$RUN/source:$MOSS_ROOT"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$RUN/console.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$RUN/job_exit.json"' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
if [[ "${RSWA_SKIP_VERIFIED_CONTRACTS:-0}" != 1 ]]; then
 torchrun --standalone --nproc_per_node=2 "$RUN/source/sp_contract.py" --output "$RUN/sp2_contract.json"
 torchrun --standalone --nproc_per_node=4 "$RUN/source/sp_contract.py" --output "$RUN/sp4_contract.json"
 python "$RUN/source/adapter_check.py"
fi
export MOSS_STOP_STEP=1 NPROC_PER_NODE=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp1" 1 4
export NPROC_PER_NODE=4
bash "$RUN/source/native_sft.sh" "$TASK/data/smoke.jsonl" "$RUN/sp4" 4 4 \
 --fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false
bash "$RUN/source/native_sft.sh" "$TASK/data/longest.jsonl" "$RUN/longest-sp4" 4 4 \
 --fsdp "$RUN/source/fsdp1_offload.json" --gradient_checkpointing false --vit_gradient_checkpointing false
printf '{"status":"native_short_and_long_steps_completed","full_gate_passed":false}\n' > "$RUN/stage.json"
