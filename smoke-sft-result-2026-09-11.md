# finetune.py 最小 smoke SFT 结果（2026-09-11）

- 目的：验证 MOSS 官方 `finetune.py` 在 DKUCC 上全流程可用（数据→训练→保存）
- 环境：moss312 env，torch 2.14.0+cu130，transformers 5.17.0，单卡 A40（作业 62950）
- 数据：`/work/qt28/moss/data/smoke_train.jsonl`（2 条：jfk.wav + asr_example_zh.wav）
- 命令：

```bash
cd /work/qt28/moss/MOSS-Transcribe-Diarize
source /work/qt28/moss/dkucc/env.sh
python finetune.py \
  --train_jsonl /work/qt28/moss/data/smoke_train.jsonl \
  --model_name_or_path /work/qt28/moss/models/MOSS-Transcribe-Diarize \
  --output_dir /work/qt28/moss/checkpoints/smoke-interactive \
  --per_device_train_batch_size 1 --max_steps 2 --learning_rate 1e-5 \
  --max_length 8192 --bf16 --gradient_checkpointing \
  --logging_steps 1 --save_strategy no --report_to none --dataloader_num_workers 0
```

## 结果

| 项目 | 数值 |
|---|---|
| step 1 loss / grad_norm | 0.5576 / 45.75 |
| step 2 loss / grad_norm | 1.1144 / 98.0 |
| train_loss（平均） | 0.836 |
| train_runtime | 3.87 s（2 steps） |
| 产物 | `/work/qt28/moss/checkpoints/smoke-interactive/`（含 model.safetensors 1.81GB、processor、tokenizer、train_results.json） |
| 日志 | `/work/qt28/moss/logs/finetune-smoke-20260911.log` |

结论：数据格式、tokenizer/processor 对齐、bf16 + gradient checkpointing、Trainer 训练与保存全部正常。
（注：两个 loss 来自不同样本，不可用于判断收敛；正式 baseline 需按 Phase 2 流程。）
