## 最新状态（2026-09-14）

63520 前置检查全部通过；正式任务 63521 已运行，目录 /work/qt28/moss/results/ms-swift-63521。先 vLLM 完整 Base dev 26 条，再从 Base 原生 MS-Swift 全数据训练；不能把 Base 评测阶段写成已经发生参数更新。TensorBoard 本地 6006、同步状态 tensorboard_sync.json 已跟随 63521。检查实际阶段以 run/status.json、train-to-5.log 和 trainer_state 为准。前置日志 231 个文件已下载至 artifacts/preflight-63520，SHA-256 已核对。

# 当前任务：原生 MS-Swift 全参数 full-attention SFT

最新指令：dev 生成评分使用 vLLM，dev CE 仍用 MS-Swift。63506 最长会议验证通过；63507 检查继承旧 eval_steps=500，未通过；63512 用相同 eval_steps=1 的新起始 checkpoint 重做恢复与评估，已通过。63508 已取消，不能放行旧依赖链。服务器端 launch_vllm_training.py PID 2290859 正在跟随 vLLM 检查 63518，检查通过后自动提交正式训练；最新任务号读取 TASK/vllm_dispatch.json 和 TASK/current_run.json。需先通过 vLLM 最短/最长 dev Base+SFT 加载检查，然后以新条件提交完整数据试运行。用户明确提醒使用 sequence_parallel_size N；当前固定 N=4。

新环境 /work/qt28/moss/envs/ms-swift-20260914，新代码 /work/qt28/moss/dkucc/ms_swift_20260914；训练调用 swift sft，不得重新提交旧的自定义训练入口。

当前使用 SP=4 + FSDP1 + 原生 activation_cpu_offload + CELOSS_PARALLEL_SIZE=512。MOSS_IMPLICIT_CAUSAL=1 仅用于单条无 padding 的完整会议：右侧补齐到 SP 整数倍，补齐标签 -100，位置连续，使用原生因果注意力。真实输入、标签和全部音频张量已对最短/最长会议逐项检查。未来 KV mask / packing 需关闭此模式。MS-Swift 上游源码没有改动。

失败记录：63496 加载 API 名称；63497 CLI 长度选项；63498 精度默认值冲突；63499 单卡/SP4 短会议通过但最长 Whisper FFN OOM；63500 FSDP2 原生卸载 DTensor storage 不兼容；63501 FSDP1 短会议通过但最长 SDPA 显式 mask OOM；63502 依赖失效已取消；63503/63505 定位 SP -1 补齐位置被误识别成 packed sequence。63506 最长 116527 token 更新和保存通过，峰值 38.01 GiB。63512 确认 optimizer 683 个参数状态均为 step=2，恢复 LR=9.975124378109452e-7、更新后 LR=9.950248756218905e-7，dev CE 更新前后均可计算。

后续步骤：
1. 检查 63506 outcome.json/job_exit.json 和最长会议 checkpoint。
2. 原生恢复验证凭据：/work/qt28/moss/results/ms-swift-resume-63512/resume_check.json。vLLM 安装日志 TASK/bootstrap_vllm.log，独立环境 /work/qt28/moss/envs/vllm-moss-20260914；未通过 vllm_check.slurm 的 Base/SFT 长输入检查不能启动完整训练。
3. 提交新的完整训练任务并确认真正开始，对全部 536 训练记录、26 dev 工作；每步 train CE。完整 dev 原生 loss 和 vLLM 生成已在 step 5、10、15、20 执行；按用户新指令，step 20 后改为 step 30、40、50。dev 资格和停点规则复用原方案，试运行最多 50 次更新、scheduler 尺度 402。
4. 更新 current_run.json、dkucc/evaluation/current_run.json，把 remote_run 切至实际正在监看的作业。TensorBoard sync_tensorboard.py 每 60 秒原样同步原生 events，已验证旧本地服务 6006 能显示新标签；同步 PID 见 sync_process.json。
5. 结束后下载所有训练/评估日志和预测，生成本轮可视化报告，不用旧报告冒充新结果。测试集不参与选择。

用户要求助手每 45 分钟检查；无变化时安静，失败、完成、需登录等实质变化时通知。当前没有可调用的 automation_update 工具，只更新本地跟进入口，不声称已修改自动化调度器。旧 LR 搜索 63458 与归档 63473 已完成，63472 未训练。

本轮必须重新用 vLLM 生成全部 26 条 Base dev，不复用旧 HF Base 分数。生成固定 BF16 greedy、完整音频和相同 prompt、131072 上下文/65536 输出上限、原 parser/评分/停点。128-token 工程验证不作为完整评分或速度倍率证据。

2026-09-14 本轮修复：63514 缺少独立环境 bin 的 PATH；63516 已生成音频但附加函数 RPC 显存统计被 vLLM 拒绝，已移除此统计调用，缺失峰值记录 null、保留原生显存日志，不启用不安全序列化。63518 重新验证，Base 最短/最长输入 token 已逐项一致并生成 128 token；继续检查 SFT 权重及 HF 小对照。TensorBoard 后续同时记录原生 CE 与 vLLM dev 指标。

2026-09-14 存储清理：按用户指令删除全部旧 `full-attention-v2-*` 模型、checkpoint、优化器及独立梯度快照，共释放约 250.83 GiB；旧日志和评测 JSON 保留。MS-Swift 全部 checkpoint、vLLM 导出和 Base 模型未删除。复核旧 v2 模型/状态文件计数为 0。

63518 SFT 加载发现原生保存重复 lm_head 别名；export_vllm.py 对共享配置、形状/dtype、张量 SHA-256 严格验证一致后仅删除推理副本中的重复别名，剩余字节哈希保持一致。63520 已完成真实 Base/SFT 最短/最长输入生成和 HF 128-token 对照。两条小实验耗时倍率约 3.01 和 1.16，排除启动时间，不代表完整 dev 加速。
