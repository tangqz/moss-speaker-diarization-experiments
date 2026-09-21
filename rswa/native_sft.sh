#!/usr/bin/env bash
set -euo pipefail
TASK=$(cd "$(dirname "$0")" && pwd)
ENV=/work/qt28/moss/envs/ms-swift-20260914
export PATH="$ENV/bin:$PATH" MOSS_ROOT=/work/qt28/moss/MOSS-Transcribe-Diarize
export PYTHONPATH="$MOSS_ROOT:$TASK${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
export MOSS_IMPLICIT_CAUSAL=0 CELOSS_PARALLEL_SIZE=512
export MOSS_RSWA_STRICT_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
DATA=$1
OUT=$2
SP=$3
ACC=$4
shift 4
mkdir -p "$OUT"
export MOSS_RSWA_ORDER_LOG="$OUT/sample_order.jsonl"
export MOSS_RSWA_INPUT_RECEIPTS=/work/qt28/moss/dkucc/rswa_20260920/evidence/training_inputs
swift sft \
 --model /work/qt28/moss/models/MOSS-Transcribe-Diarize \
 --model_type moss_rswa --template moss_rswa \
 --external_plugins "$TASK/moss_rswa_plugin.py" "$TASK/runtime_callback.py" \
 --tuner_type full --freeze_llm false --freeze_vit false --freeze_aligner false \
 --dataset "$DATA" --split_dataset_ratio 0 --load_from_cache_file false \
 --lazy_tokenize true --strict true --max_length 131072 --truncation_strategy delete \
 --packing false --padding_free false --sequence_parallel_size "$SP" \
 --torch_dtype float32 --bf16 true --fp16 false --attn_impl sdpa \
 --learning_rate 1e-7 --lr_scheduler_type linear --warmup_ratio 0 --weight_decay 0 \
 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 1 \
 --optim adamw_torch_fused --max_steps 402 \
 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps "$ACC" --gradient_checkpointing true \
 --vit_gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
 --average_tokens_across_devices true --use_logits_to_keep false \
 --dataloader_num_workers 0 --dataloader_persistent_workers false \
 --ddp_find_unused_parameters false --logging_steps 1 --save_steps 10 \
 --eval_strategy no --prediction_loss_only true --report_to tensorboard --callbacks rswa_phase \
 --seed 0 --data_seed 0 --add_version false --output_dir "$OUT" "$@"
