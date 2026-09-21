# 设计依据与验证边界

采集时间：2026-09-15，Asia/Shanghai。来源是既有 SSH 连接上的只读检查；本任务没有提交或取消远程训练/评估作业。

| 凭据 | 已确认的内容 | 不能据此推出的结论 |
|---|---|---|
| `source_protocol.json` | 原 536 train/26 dev 的路径和哈希、SP4/FSDP1、LR=1e-7、完整会议设置 | 新恢复 Trainer 已验证 |
| `source_selection.json` | 原有 dev 选择器选中 step 150，score=11.509634670974838 | step 150 已消除所有循环 |
| `existing_test_error_reference.json` | 当前已存在的 step-150 Test 报告 CER/cpCER、56 场 SFT 样本及 1 场截断，保留源文件哈希 | CER 等于逐 token 扰动率；该 Test 仍是盲测 |
| 远程目录只读检查 | `training/checkpoint-150` 实际存在，trainer state global_step=150；官方 MOSS HEAD 为 `61bc29cd4120be7b5d3b761b64cd5dff57263642` | 新实验初始权重哈希已验证；启动前仍要 hash |
| `prior_diagnostics_status.json` | 旧诊断结束为 `complete_with_errors`，failed=true | 旧诊断成功完成 |
| `prior_diagnostics_summary.json` | 52 条 HF D1 记录；16 条 D0 的 vLLM 状态均 missing | HF/vLLM 路径一致性通过 |
| `runtime_transformers_trainer.py` | 当前远程安装环境中的真实 Trainer 源码副本 | 本地已安装同版本或已执行 GPU parity |
| `runtime_swift_identity.json` | 当前训练环境实际 import 的 editable Swift 路径及版本 | 新训练入口已完成 GPU 运行 |

旧诊断目标为已知 Test `alimeeting/test/R8005_M8009`。它只用于确认工程风险，不进入本实验训练/Dev案例库或选择器。

## 归一化核查

远程 Transformers 源码的 `get_batch_samples` 会预取整个梯度累积窗口。`_get_num_items_in_batch` 在 SP 分片前统计 labels；因果标签移位路径使用 labels[...,1:]。之后根据 `average_tokens_across_devices` 汇总各 rank，并可能依 `parallelism_config` 去除非数据并行副本。

Swift `Seq2SeqTrainer.compute_loss` 又执行自身的 world-size 补偿，`GatherLoss.backward` 也有 SP 梯度倍率。因此 DESIGN 中 N 指唯一源标签总数，运行时的 `num_items_in_batch` 字面值可能不同。必须记录有效分母与完整梯度 parity；不能随手增加 `/4` 或 `/8`。

本地审查还确认 Swift SP sampler 不传入 shuffle 参数，不能只使用 `train_dataloader_shuffle=false` 来保持成对视图顺序。

## 当前验证状态

- 已完成：技术设计、当前源码接口检查、数据/初始化来源核对、旧诊断边界核对。
- 已完成：独立 golden token 数组和数值权重检查，见 `independent_contract_review.json`；其中旧 v1 强制错误候选检查已被 v2 sparse 替代，不能据此声称低概率策略已通过。Sol 的全量 tokenizer 审计已读回，见 `../artifacts/tokenizer_audit.json`，536 条记录、355402 个合法数值时间片段、10892633 个 target token、2544562 个 tokenizer-only 候选位置，原数据与 tokenizer 哈希实时核对相同。
- 已完成：独立 `.venv-test` 环境中首批 13 项 CPU 测试通过，含 PyTorch autograd 权重梯度 oracle；最终以收尾时 pytest artifact 为准。Python 3.12.13，PyTorch 2.14.0+cpu；依赖锁定记录见 `requirements.cpu-test.txt`。原 Anaconda PyTorch 导入的 XCCL 符号冲突未修改。
- 已完成：v2 中期 19 项测试通过（`../artifacts/pytest-v2-interim.xml`）；`independent_sparse_review.json` 保存主任务独立的 10000 位置合成采样检查，温度分支 187 次、实际变更 33 次、采到正确答案 154 次，验证“低概率采样且允许正确答案”的语义。这些是合成 CPU 数据，不是 MOSS 实际错误率或效果。
- 已完成：最终 21 项 CPU 测试（`../artifacts/pytest-final.xml`）、16 个 Python 源文件语法、四组参数/累积步数/402 步 scheduler horizon/FSDP 配置、概率上限及零概率配置；静态凭据见 `final_static_review.json`。Git Bash `bash -n train.slurm` 退出 0。
- 已完成：主任务对首版数据流、采样、标签对齐、分布式适配、校准/评估分母及启动说明的代码审查；这不代替实际框架运行。
- 待完成：真实音频官方 collator 验证和训练 manifest；G2 数值比较、G3 专用计划、G4 跨后端比较、G5 整会评估的完整自动化工具，详见 `../README.md`。
- 待GPU执行：G1–G5、新入口最长会议显存、SP梯度与恢复一致性、实际自由生成对照。

本文件与 DESIGN 由主任务维护，Sol 负责实验代码；最终运行证据以新实验自身的可验证结果文件为准。
