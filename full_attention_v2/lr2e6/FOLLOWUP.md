本次 63443 交付已完成：256 文件校验、独立原生报告及四图实际查看完成。预览 http://127.0.0.1:4159/?view=1 。对未改变的完成状态静默，不重复通知或提交新作业。后续 dev5 仍为规范阶段；新 runner 尚未接入。automation_update 未暴露，停止请求已记录，未声称已删除自动化。

# 2026-09-13 新的后续训练要求（优先）

用户要求普通 CE，train loss 每步、完整 26 场 dev loss 每 5 步，以 dev 自由生成准确率指标选 best/停点，并以实验选择学习率。规范在 `dev5/后续训练与学习率选优规范.md` 和 `dev5/protocol.json`；当前为已定义、未接入 runner、未提交新搜索。不得再将六场每 25 步的旧监测称为下一轮方案。保持 full attention 全参数对照和后续 mask/RL 公平性。

63443 已于 2026-09-13 04:42:43 UTC 正常退出，质量门槛失败，停在 50。六场评分齐全：一场截断，另一场严重解析失读，二者非同一会议。完整日志 256 文件已下载校验。报告在 `lr2e6-report/`。下文为历史执行约定，不据其重新提交。

# 当前任务：已授权的学习率 2e-6 短对照

用户在上一阶段交付后明确要求“按你的方案改变学习率试试看”。当前作业 **63443**；以本目录 current_run.json 为准。原 63421/63441 结果已交付并保留。旧自动化提示中的 62994/63363 是更早历史，不能据其完成守卫停止这次监控。

## 执行约定

- 独立目录 `/work/qt28/moss/results/full-attention-v2-63443`；检查点 `/work/qt28/moss/checkpoints/full-attention-v2-63443`。
- 从同一个原始 base 开始；全参数、full attention SDPA、FP32 参数/梯度/Adam、BF16 前向，4 A40 DDP、SP=1、有效 batch=4、相同 536 场完整会议和数据顺序。
- 仅改变学习率幅度 1e-5 → 2e-6。warmup=0、weight decay=0，scheduler 总长度仍为 402；通过 callback 最多执行 50 步。
- 阶段为 0→10→25→50。第 10 步完整 TS3004c 生成，截断、空解析或原严重解析失读门槛触发即停；通过才继续。第 25/50 步做原六场 teacher loss，第 50 步做六场完整生成，然后无论通过与否均结束短对照供审阅。没有全 dev/test 或正式模型选择。
- 第 0 步 teacher 基线从 63421 的相同协议复用，明确记录来源；不是本次重新测量。原始生产 base 预测用于质量，第 50 步同权重比较取修正后的 63441，旧 mixed-preprocessing 差值不作为正式对照。
- 生成入口使用已在 63441 验证的 production_generate.py，SHA256 `b01db8701e04ecba615813839a2e419d0b687b1cb8d922f3509b19032501a7e3`；完整音频预处理放在生产 CUDA BF16 autocast 内。
- 保存第 1/5 步权重、第 10/25/50 步完整恢复状态。实际检查会议索引、监督 token、全局分母和学习率相对于 63421 的 1/5 比例；到第 50 步应有 4,076,380 个监督 token。新增第 10 步 phase 边界会重载训练进程，保留数据/RNG/scheduler 恢复语义，不宣称训练逐位确定。

已验证本地和远端的源文件语法、质量门槛对不完整评分的拒绝；远端提交有文件锁和本地/远端收据。24 个文件已校验上传，source_manifest SHA256 为 `aa39b789b16b2a6914eaaf994ed62d353560647b61d5f5704683708fe7cceba0`。不重复执行 build_bundle.py 修改已提交 bundle；submit.py 会复用既有收据，不能用于未经审查的重跑。

## 检查与异常恢复

1. 复用已认证控制路径 `C:/Users/qizhi/.ssh/cm-moss-check-20260912`。运行父目录 `sync.py status --job 63443 --run /work/qt28/moss/results/full-attention-v2-63443`。四卡更新在 `checkpoints/train-rank-*.jsonl`，生成进展在 `evaluations/probe-10/worker-0-status.json` 或 `evaluations/sentinel-50/worker-*-status.json`。
2. `verify_progress.py` 仅读取，核查真实更新及与 63421 的相同暴露量；`entry_lineage_verified.json`、`device_preflight.json`、`data_order.json`、`matched_exposure_verified.json` 给出运行证据。
3. Slurm COMPLETED / 0:0 只表示流程正常退出。outcome 为 `paused_for_probe_failure` 时应只有 10 步；`paused_for_sentinel_failure` 或 `completed_short_trial` 为 50 步。最后一种也只是短对照完成，不能称收敛或已得到合格 SFT。
4. 执行异常先保留日志、确认最后一个完整保存点。在原授权范围内修复实现或连接；不为了改善分数改 parser、解码、裁音频或删失败。部分 phase 留下日志时，不能直接重跑原训练：先归档保存点之后的部分记录并核实 optimizer/scheduler/四 rank RNG、数据顺序后恢复。
5. 新 run 会在不同目录中运行，旧源文件与旧训练不能覆盖。原 63421 仍质量暂停；不要把本次解释为继续旧第 50 步权重。

## TensorBoard 与跟进

TensorBoard `http://127.0.0.1:6006`，本次名称 **lr2e6_full_attention_63443**。桥接每 60 秒只读日志；聊天监控继续原有 45 分钟节奏。第 10 步生成期间训练 loss 暂停更新是预期状态。旧 full_attention_v2_63421 和 production_eval_63441 用于各自协议下的比较，保留不覆盖。

正常训练、正常生成、排队且没有需要处理的变化时 DONT_NOTIFY。新故障、必要登录、完整第 10 步质量结果或最终完成可以通知；已通知发现不重复。当前需要保留监控，automation_stop_pending=false。automation_update 工具目前未暴露，保存的新路由优先于旧自动化提示，不直接改 automation.toml；若工具恢复，更新提示指向此短对照并保持 45 分钟。

## 短对照完成后的交付

- 用父目录 sync.py download 下载本作业全部日志、原始预测、指标与配置并逐文件校验（包含 moss-v2-lr2e6 Slurm 日志和源码 diff；大模型/optimizer 二进制留在服务器）。
- 与 63421 的相同步数、同会议训练记录及 **63441 的生产入口第 10/50 步**比较。达到 50 步才有六场新结果；若第 10 步终止，不伪造第 25/50 步结果。CP1/5 或小范围通过不成为正式选模候选。
- 新建该短对照的分析数据和报告，保留 stage50-report 原复评档案；原生报告可复用其组件方式，必须标明作业、学习率、匹配的监督 token 和各数据范围。报告实际查看图表，列出统计改善和剩余失败，同时区分诊断、全 dev 验收与 test。
- 若只缩小学习率仍失败，如实记录。不得自动扩大成网格搜索、改 warmup 或做 RL。若通过第 50 步灾难门槛，仍先报告完整 cpCER/DER 后审阅；不自动推进到 134 步。
- 完成本次交付后再根据上层自动化规则停止监控；没有工具则保存完成守卫并静默，不声称自动化已停止。
