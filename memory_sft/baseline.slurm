#!/usr/bin/env bash
# Full-attention DDP only: no sequence parallelism, no data slicing/truncation.
#SBATCH --job-name=moss-fa-ddp4
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -eo pipefail
source /work/qt28/moss/dkucc/env.sh
export PATH="/dkucc/home/qt28/envs/moss312/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MOSS_PROJECTION_CHUNK=512
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK=/work/qt28/moss/dkucc/memory_sft
RUN=/work/qt28/moss/checkpoints/full-attention-ddp4-${SLURM_JOB_ID}
SMOKE=/work/qt28/moss/checkpoints/memory-smoke-${SLURM_JOB_ID}
cd "$MOSS_ROOT"
nvidia-smi -L
nvidia-smi topo -m
python -c 'import torch,transformers,accelerate; print(torch.__version__,transformers.__version__,accelerate.__version__); assert torch.cuda.device_count()==4'
python -m torch.distributed.run --standalone --nproc_per_node=4 "$MOSS_HOME/dkucc/dist_test.py"
python "$TASK/test_projected_loss.py"
python -m torch.distributed.run --standalone --nproc_per_node=4 "$TASK/test_projected_loss.py"
python "$TASK/verify_model.py"
python "$TASK/preflight_data.py"
COMMON=(--model_name_or_path "$MOSS_MODELS/MOSS-Transcribe-Diarize"
  --per_device_train_batch_size 1 --learning_rate 1e-5 --max_length 131072
  --bf16 true --gradient_checkpointing true --attn_implementation sdpa
  --ddp_find_unused_parameters false --average_tokens_across_devices true
  --logging_steps 1 --report_to none --dataloader_num_workers 0 --seed 0
  --save_total_limit 2)
python -m torch.distributed.run --standalone --nproc_per_node=4 "$TASK/memory_sft.py" \
  --train_jsonl "$MOSS_DATA/moss_jsonl/codex_smoke_${SLURM_JOB_ID}.jsonl" \
  --output_dir "$SMOKE" --max_steps 2 --save_strategy steps --save_steps 1 "${COMMON[@]}"
python -m torch.distributed.run --standalone --nproc_per_node=4 "$TASK/memory_sft.py" \
  --train_jsonl "$MOSS_DATA/moss_jsonl/codex_smoke_${SLURM_JOB_ID}.jsonl" \
  --output_dir "$SMOKE" --max_steps 3 --resume_from_checkpoint "$SMOKE/checkpoint-2" \
  --save_strategy steps --save_steps 1 "${COMMON[@]}"
python -c 'import json,os; from pathlib import Path; p=Path("/work/qt28/moss/checkpoints/memory-smoke-"+os.environ["SLURM_JOB_ID"])/"trainer_state.json"; s=json.loads(p.read_text()); assert s["global_step"]==3; print("CHECKPOINT_RESUME_PASS step=3")'
printf '%s\n' "BASELINE_START output=$RUN records=536 epochs=3 DP=4 SP=1"
python -m torch.distributed.run --standalone --nproc_per_node=4 "$TASK/memory_sft.py" \
  --train_jsonl "$MOSS_DATA/moss_jsonl/train_mix.jsonl" --output_dir "$RUN" \
  --num_train_epochs 3 --save_strategy steps --save_steps 25 "${COMMON[@]}"
printf '%s\n' "BASELINE_COMPLETE output=$RUN"
