# 当前优先任务：63458 dev5 学习率搜索已授权并启动

先读取 C:\Users\qizhi\Documents\ChatGPT\说话人日志\dkucc\full_attention_v2\dev5\FOLLOWUP.md 和对应 current_run.json。旧 63443 已完成，不得据旧守卫停止新监控。每步 train、每5步完整26场 dev loss/生成，以dev质量早停；正常进展静默，按45分钟检查。

本次 63443 交付已完成：256 文件校验、独立原生报告及四图实际查看完成。预览 http://127.0.0.1:4159/?view=1 。对未改变的完成状态静默，不重复通知或提交新作业。后续 dev5 仍为规范阶段；新 runner 尚未接入。automation_update 未暴露，停止请求已记录，未声称已删除自动化。

# 2026-09-13 新的后续训练要求（优先）

用户要求普通 CE，train loss 每步、完整 26 场 dev loss 每 5 步，以 dev 自由生成准确率指标选 best/停点，并以实验选择学习率。规范在 `dev5/后续训练与学习率选优规范.md` 和 `dev5/protocol.json`；当前为已定义、未接入 runner、未提交新搜索。不得再将六场每 25 步的旧监测称为下一轮方案。保持 full attention 全参数对照和后续 mask/RL 公平性。

63443 已于 2026-09-13 04:42:43 UTC 正常退出，质量门槛失败，停在 50。六场评分齐全：一场截断，另一场严重解析失读，二者非同一会议。完整日志 256 文件已下载校验。报告在 `lr2e6-report/`。下文为历史执行约定，不据其重新提交。

# 当前优先路由：学习率 2e-6 短对照 63443

用户在上一阶段交付后明确要求按方案降低学习率试验。新作业 63443 已提交并进入训练阶段；请先阅读 `C:/Users/qizhi/Documents/ChatGPT/说话人日志/dkucc/full_attention_v2/lr2e6/FOLLOWUP.md` 和 current_run.json。最大执行 50 步，10 步先做完整 TS3004c 质量门槛，完整 dev/test 不在此次短对照内。旧 63421/63441 完成守卫仅属历史，不能阻止新授权工作或停止正在使用的监控。保持 45 分钟、正常进展静默；不得绕过质量门槛。

---

# 当前任务：按方案执行 full attention SFT v2

用户已于 2026-09-12 明确要求“现在开始按照方案执行”，并追加安装 TensorBoard。当前运行信息以同目录 `current_run.json` 为准。旧训练 62994、评估 63363 及其报告已交付，继续保留，不能再次作为当前待完成任务。

## 最新状态：生产入口完整复评 63441 已完成，训练继续暂停

2026-09-13 08:15（北京时间）交付整理。63441 于 07:12:55 完成，Slurm COMPLETED / 0:0，8 项新生成全部完成，三组评分为 12/2/2 条，无执行错误或缺失。相同 base 与 TS3004c 重复记录有重叠，不能视为 16 场独立会议。

修正后 base 的全部 26,337 个生成 token 与历史逐个相同，原始文本也相同。第 50 步仍有 2/6 场截断：TS3004c 连续 1,657 个 Mm.；R8001_M8004 连续 51,141 个相同 token（下）。第 10 步 TS3004c 完整诊断也截断，4,085 个非法 Música 标签，cpCER 99.6092%。AliMeeting cpCER 18.0413% → 25.1156%，AMI 8.9708% → 15.5734%；各自 DER .25 为 8.2830% → 15.8252%、12.2926% → 13.6158%。旧混合预处理比较继续保留为历史诊断。

131 个复评日志已下载并逐文件校验，归档 SHA256 `def0038986fad3350a75ff487c7f2b2c4d8f54531d73579e91218581a48c6e89`。同一个 stage50-report 原生报告已更新，20 份 CSV、7 组 PNG/SVG 保留，新增图与桌面/窄屏、来源面板已实际检查。TensorBoard 连接正常，新增 run `production_eval_63441`；原训练曲线最后一步仍为 50。bridge 每分钟只读刷新继续保留。

**当前没有待检查的训练/评估作业。正式训练未完成，尚未做全部 26 场 dev 与 56 场 test，也未产生合格 SFT。** `diagnostics/下一次学习率单因素短对照.md` 已定义只降低学习率至 2e-6 的后续短对照，尚未提交；本次不自动恢复旧训练、不放宽门槛或修改评分。

完成守卫：先查看 notification-state.json。此完成事件通知一次后，普通心跳不得重复下载、重建报告或重复发结果。监控原因已结束；如果 automation_update 变为可用，按上层要求删除 automation id=moss。本轮工具不可用，已记录 automation_stop_pending，不直接编辑 automation.toml，不声称已经停止自动化；后续仍不可用则 DONT_NOTIFY，不重复告知工具缺失。用户追加新工作时依新请求更新路由。

## 06:05 历史状态：当时生产一致复评尚在运行

最近一次静默核查（2026-09-13 07:01 北京时间）：63441 正常 RUNNING，8 项生成已完成 7 项，七个进程均退出 0；剩余 AliMeeting R8001_M8004 正在生成（53,760 token，最近 23:01:00 UTC），进度持续变化，尚在 3 小时时限内。没有已观察到的执行错误，完整评分尚未产生，继续等待整批结果，不重复通知部分输出。详见 `diagnostics/last_quiet_check.json` 和 `production_workers.json`。

2026-09-13 06:05 北京时间更新。63439 于 05:34 COMPLETED / 0:0，四组各 2 条评分完整，无执行错误。原 v2 预处理下：base 和第 1 步正常 EOS；第 10 步仅 10 个合法片段，最后结束 65.95 秒，随后反复 `[66.70][66.70][?]`，该单元 4,657 次、非法标签共 4,666，cpCER 99.492%；第 20 步连续 1,643 个 “Mm.”，cpCER 22.763%。两者均 65,536 token 截断。失败最早在已测第 10 步观察到；不能声称精确起点或单调退化。60 个文件已归档校验，hash `a973f0c622a62f03f434e17e8d06341526b7d15793ff6a7e7ed2e6334efca47f`。

**重要更正：**历史评估和生产入口在 CUDA BF16 autocast 内完成音频 prepare_inputs，63421 的原 generate.py 在其外；因此此前“推理路径完全一致”的说法不成立。63440 的单因素探针已 COMPLETED：token IDs/mask/音频长度/mapping 相同，feature 张量虽均为 float32，最大绝对差 0.001187、RMS 0.0002606。base 在生产环境下的前 512 token 与历史全部一致，原环境复现本轮前缀；两者从第 35 token 分叉。第 10 步两环境仍有非法标签循环（`?` / `Música`）。短前缀不能证明完整质量通过，也不能把所有重复都归因于预处理。13 个探针文件已归档校验，hash `95d7aac5e2dd6bff58958887775b5a6abc7681b28c5c1bd49311d1d3c9296d6e`。

已生成仅恢复生产音频预处理上下文的派生 worker，精确修改与 hashes 在 `diagnostics/preprocessing_fix.diff/json`，不覆盖冻结原脚本或原结果。训练、teacher 验证和权重未改变。旧 63421/63432/63439 生成保留为原 v2 预处理路径的诊断，其与历史 base 的差值不用于生产对照验收。报告顶部已明确更正，TensorBoard 状态页也显示此说明；旧曲线保留。

**当前作业 63441**：`/work/qt28/moss/results/full-attention-v2-diagnostic-63441`，四 A40、3 小时时限。生成 8 项：第 50 步原六场，加完整 TS3004c 的新 base 与第 10 步。worker 文件在 `cases/*/worker-*-status.json`，与此前 `checkpoint-*` 布局不同；source、probe_evidence、derivative_source_manifest 记录来源。06:02 四路已实际生成，无执行错误。按 45 分钟正常推进仍静默。

下次检查 `python dkucc/full_attention_v2/sync.py status --job 63441 --run /work/qt28/moss/results/full-attention-v2-diagnostic-63441`，并读取 `cases/*/worker-*-status.json`。完整判定：Slurm COMPLETED/0，job_exit=0、outcome completed、8 项新生成无执行错误，`evaluations/sentinel-50` / `base-check` / `step10-check` 三组各 12/2/2 条完整评分，无 errors/missing。outcome 的 base_recheck 另比较新旧完整 base tokens。base-check 中 `predictions/sft` 是候选标签，实际是原始 base，不能误写为训练模型。诊断共享历史 base 副本，不是 16 个独立会议。

完成后 download 同 job/run，校验并追加原生报告与导出图；看生产一致结果再作后续决定。**正式训练继续停在 50；此评估不会自动恢复训练，不开始新的训练、不放宽质量门槛、不改评分。** 若修正后质量仍失败，准备具体单因素训练对照供后续处理，不能把旧混合预处理的差值继续当作因果证据。长期 full/mask/熵 RL 输入约束见 `diagnostics/预处理一致性更正.md`。

本轮将通知 63439 完成及已确认的协议更正；后续不重复这些已知发现，仅 63441 完成、不同故障或需用户操作时通知。当前监控仍有用途，不删除；automation_update 工具目前不可用，不直接编辑 automation.toml。

## 04:00 历史状态：63432 已完成，63439 尚在运行

最近一次静默核查（2026-09-13 04:54 北京时间）：63439 仍 RUNNING（约 1 小时）。base / 第 1 步 / 第 10 步完成；前两者正常 EOS，第 10 步达到 65,536 token 上限。第 20 步仍推进至 44,288 token，最近状态 20:53:55 UTC，尚在 2 小时时限内，无已观察到的执行错误。本批尚未全部评分，不据部分结果交付或重复通知。新 base 实际输出 25,546 token，历史为 26,337，均正常 EOS；批次完成后需核对推理来源与输出差异，不能预设所有推理逐位确定。检查记录 `diagnostics/last_quiet_check.json`；下一次仍优先等待 63439 完整结果。

2026-09-13 04:00 北京时间：63432 已于 03:34 COMPLETED / 0:0，job_exit=0，两组各 2 条评分完整，无执行 errors 或 missing。第 25 步已出现 1,692 个连续 “Mm.” 片段；第 50 步重复为 1,695 个，全部 65,536 个 token 与首次结果逐个相同，权重来源和原文本也一致。两者均达到生成上限。只能确认该会议上的早期失败与重复可复现，不能解释为整个训练逐位确定或单凭它断言过拟合。

34 个复核日志文件已下载、逐文件验证，归档 `artifacts/63432/logs-63432.tar.gz`，460,616 bytes，SHA256 `158b1f4582e121bdfad9e22d3892c08662c0aa5f68d094a7616c65ec34c80e3d`。原生阶段报告、CSV、PNG/SVG 已追加复核比较，原首次 HTML 另存 `stage50-report/MOSS_50步阶段报告_首次评估.html`。最终报告仍没有完整 dev/test。

参考覆盖新发现：最后参考片段结束 2,452.69 秒，音频长 2,970 秒；两次循环起点约 2,502 秒，位于最后标注之后。尚未听审尾段，不得据无标注断言静音或把该段全部文字视为幻觉。完整音频和评分不改变。

按原方案先定位发生阶段，已启动 **63439**，运行目录 `/work/qt28/moss/results/full-attention-v2-diagnostic-63439`。使用现有 1/10/20 步权重及原始 base，四张 A40 各做完整音频推理，**没有新训练更新**。04:00 四 worker 正常生成；index 3 另有约 24.4 GiB 原有占用，推理仍有进展，不比较独占性能或终止其他进程。路由 `diagnostics/current_run.json` 与 `early_run.json`。`checkpoint-0/predictions/sft` 是冻结 worker 的统一候选标签，实际模型为原始 base，报告必须正确标识。

下次 45 分钟检查优先 `python dkucc/full_attention_v2/sync.py status --job 63439 --run /work/qt28/moss/results/full-attention-v2-diagnostic-63439`，并读 `checkpoint-{0,1,10,20}/worker-*-status.json`。非正式作业状态保存到 `diagnostics/latest_status-<job>.json`，不再覆盖正式 `latest_status.json`。正常进行仍静默。完成后检查四组各 2 条评分、outcome 与原始 token/片段轨迹，再 download 同一 job/run，校验全部日志并追加报告。解释边界与待核查项见 `diagnostics/阶段定位与尾段核查.md`。已知 63432 结果在本轮通知，后续不要重复；仅 63439 完成、不同故障或需操作时通知。正式训练继续暂停，不擅自开始新的训练或绕过质量门槛。

## 03:00 历史状态：第 50 步质量暂停，63432 尚在运行

2026-09-13 03:00 北京时间核查：正式训练 63421 于 01:50 按规则停在第 50 步，`outcome.status = paused_for_sentinel_failure`。Slurm COMPLETED / 0:0 表示暂停流程正常退出，不表示整轮训练完成或验收通过。6 场 sentinel 的 base/SFT 共 12 条均完成评分，无执行错误或缺失；SFT 有 1 场截断，其余 5 场正常 EOS，无空解析或严重解析失读。

失败会议 AMI/dev/TS3004c（49.5 分钟）连续生成 1,695 个 “Mm.” 片段，并交替写入说话人和递增时间戳，最终耗尽 65,536 token。原 parser 对合法输出的解析没有严重失读；此前单 token 连续重复指标只得到 6，不能代表这种结构化循环。新增连续文本片段统计仅用于解释，不改变评分或暂停标准。固定六场 teacher loss 为 0.5931 → 0.4327 → 0.4248（0/25/50 步）；但 cpCER：AliMeeting 18.04% → 19.86%，AMI 8.97% → 15.26%。这说明 teacher loss 改善不能保证完整生成质量，尚不能单独归因为过拟合。

247 个训练与阶段验证日志文件已下载、逐文件校验，归档 `artifacts/63421/logs-63421.tar.gz`，15,469,398 bytes，SHA256 `382a056a622a962af05c1a180296a71b3096ef4448a40a75ac7f2e8cfd578105`。阶段报告为 `stage50-report/MOSS_50步阶段报告.html`，原生可交互报告及源码位于 `stage50-report/app`，验证记录 `stage50-report/verification.json`。桌面、窄屏、来源面板与四幅导出图已查看。报告只覆盖 50 步与六场 dev sentinel；26 场完整 dev、56 场 test 尚未执行。旧 v1 报告保持原样。

推理复核 **63432** 已启动，两张 A40 分别运行 checkpoint-25 的同会议生成与 checkpoint-50 的独立重复，**training_updates = 0**。目录 `/work/qt28/moss/results/full-attention-v2-diagnostic-63432`，路由见 `diagnostics/current_run.json`。冻结的原 generate.py、BF16 greedy、完整音频、65,536 cap 和评分器均保持不变；此诊断不属于正式 checkpoint 选择。03:00 两个 worker 仍推进，分别约 32,768/33,280 token，无已发现执行错误，结果尚未完成，不能提前判断循环或通过。

下一次 45 分钟检查优先执行：

```powershell
python dkucc/full_attention_v2/sync.py status --job 63432 --run /work/qt28/moss/results/full-attention-v2-diagnostic-63432
```

还应读取两处 `checkpoint-*/worker-*-status.json`。正常推理中静默。完成时核对 Slurm、status、outcome、各 checkpoint 的 quality_status/metrics_summary 和原始预测，再按同一参数运行 `sync.py download --job 63432 --run /work/qt28/moss/results/full-attention-v2-diagnostic-63432` 保存全部复核日志。重点比较 25 步何时进入循环，以及 50 步重复能否复现；保留失败会议，不能筛除它或更改 parser 获得合格结果。根据实证提出下一步并保持 base full attention SFT 及未来 mask KV / 熵 RL 的共同协议。**正式训练继续暂停，不绕过第 50 步质量门槛，不静默从 base 重启。**

本次暂停及阶段报告会在本轮通知；记录见 `notification-state.json`。后续不要重复通知已知暂停。诊断尚在进行，整个任务 `delivery_complete=false`，每 45 分钟监控仍有用途，不能按旧 v1 完成守卫删除。当前没有 automation_update 工具，不直接改 automation.toml。TensorBoard 桥接仍每 60 秒正常读取日志，训练 loss 停在 50 为预期；Text 状态已明确显示质量暂停。桥接 PID 以 `dkucc/tensorboard/processes.json` 为准。

## 当前执行与完成条件

- 当前正式作业：63421；结果 `/work/qt28/moss/results/full-attention-v2-63421`；权重与恢复状态 `/work/qt28/moss/checkpoints/full-attention-v2-63421`。63415 于 2026-09-12 15:25 UTC 在第 16 步期间被 Slurm 主机内存上限 240 GiB 终止，完成了 15 步但尚无 checkpoint；其 83 个日志文件已下载并逐文件验证，失败记录不覆盖。
- 63421 从同一个 base、同一 seed、同一数据顺序重启；不是从第 15 步恢复。`host_runtime.py` 包装原入口，每步更新完成后同步、回收 Python 循环引用并释放闲置 pinned host 缓存；不修改仍通过 hash 校验的 train.py、memory_sft.py 或实验协议。额外保存第 1/10/20 步，之后仍按原定边界保存；作业内存申请为 384 GiB。
- 新增 `test_host_runtime.py` 已通过闲置缓存释放、活跃张量保留、梯度/loss 和 RNG 不变检查。训练逐步的主机内存和释放效果记录在 `host-memory-rank-*.jsonl`，应确认最长会议第 14 步与旧故障第 16 步经过后内存回落，再称长会训练稳定。
- 9 月 13 日 00:06 已独立核对前 4 步，清理后各 rank RSS 合计约 9–12 GiB、闲置 pinned 缓存由约 78–99 GiB 降至不足 0.5 GiB；第 1 步完整恢复点已确认。可运行 `python dkucc/full_attention_v2/recovery/verify.py` 更新内存释放和原故障批次验收，运行 `verify_launch.py` 核对实际数据顺序；均为只读远端检查。
- 当前完整恢复点保留 20/25/50；1/10 权重保留，优化器等状态已按预定 retention 清理，不能再称其完整可恢复。处理意外中断时先确认 checkpoint_complete 与 optimizer/scheduler/四卡 RNG 都齐全，并归档之后的部分日志，不直接重跑 deploy.py。当前第 50 步属于质量暂停，不能仅因保存点齐全就恢复更新。
- 最终验收 63414 已 COMPLETED / 0:0。最长 116527 token 的 DDP4 更新成功，最高显存 28.263 GiB；参数/梯度/Adam 为 FP32，BF16 前向。保存值逐参数检查一致，完整音频 BF16 重载 logits 完全一致。
- 单卡累积与 DDP4 抽样梯度差 0.71095%；相同 DDP 重复差 0.86728%。连续与恢复差 0.38279%；两次相同恢复差 0.32482%。相同配置存在数值波动，不应要求训练逐位复现。数据、每卡 RNG、调度器延续一致；原始严格门槛失败和重复证据保留。
- 第 0 步先建立 6 场 teacher-forced 基线并检验随机 sampler 的全部恢复边界，然后开始原始 base 的正式训练。整个调度器始终 402 步；callback 每 25 步及 134/268/402 暂停并保存完整恢复状态。
- 每 25 步：6 场完整音频 teacher loss/类型/熵与独立重复前缀诊断。第 50 步：6 场完整自由生成；若截断、空解析或严重失读，则暂停并调查，不擅自放行。
- 第 134/268/402 步：26 场全 dev 自由生成。按原方案的 eligibility、macro 分数、早停和早期优先规则决定继续与选择。算法已实现并自动执行，无需再次问用户是否继续每一阶段。
- 选择冻结后，合格 SFT 执行原 56 场 test；没有合格 SFT 时如实报告，不通过筛掉失败会议或更改 parser 获得合格结果。
- `status.json` 表示当前阶段；`failure.json` 表示执行错误；`selection.json` 表示开发集决策；`outcome.json` 表示完成或因哨兵失败暂停。Slurm COMPLETED 本身不代表训练验收通过，应结合这些文件。

## 跟进方式

2026-09-13 01:42 北京时间续查：63421 仍在第 50 步 sentinel 验证。已完成 5/6 场，五场均正常 EOS，无截断、空解析或严重 parser 失读；最后一场 AMI TS3004c 仍实时产生 token（59,392，最近状态 17:42:32 UTC）。尚未到本轮 65,536 的生成上限，不能提前宣称截断或判定通过。三个 worker 已正常完成，rank 0 的进度持续变化，未发现执行挂起。下次检查优先读取完整 quality_status 与 outcome；若长场最终截断，按既定第 50 步暂停规则调查，保留完整原始输出。该次检查记录在 `sentinel-progress.json`，尚无需要用户操作的新情况。

最近一次静默检查：2026-09-13 00:54 北京时间，63421 正常 RUNNING，已完成 50 步并通过所有实际 batch/全局 token 分母核对；完整恢复状态保留 20/25/50，1/10 权重保留且状态已按原定 retention 清理。第 14 步最长会议和第 16 步原故障批次已通过，清理后 RSS 各 rank 合计分别 12.30/12.43 GiB；第 50 步为 20.94 GiB。25/50 步 fast 验证完成，正在六场 sentinel 完整自由生成，训练曲线停在 50 是预期验证阶段。快照 233 个文件已下载并逐文件验证；完整生成质量尚未判定，不能把 teacher loss 改善当作长会生成通过。下一次优先检查 sentinel quality_status、outcome 和是否按协议恢复训练；正常验证中仍静默。

用户要求的对话检查频率仍为 45 分钟。原 automation `moss` 已确认 ACTIVE / 每 45 分钟；其提示先读取 `dkucc/evaluation/current_run.json` 与 FOLLOWUP.md，所以这两处已指向本轮。当前没有可调用的 `automation_update`，未直接编辑 automation.toml。有工具时应将提示更新为本轮并保留 45 分钟和静默规则；新训练尚在运行时不要删除监控。

读取 `current_run.json`，使用 `python dkucc/full_attention_v2/sync.py status` 检查。SSH control path 为 `C:/Users/qizhi/.ssh/cm-moss-check-20260912`，采用已有 no-fd-passing proxy helper。需要网络工具权限时使用已授权的服务器操作；只有连接失效且确需认证时才打开可见 SSH 终端让用户登录。密码或 Duo 不通过聊天索取。

正常推进、排队、验证中均 DONT_NOTIFY。仅完整 dev 决策、完成、失败、暂停或必须用户操作时通知。不得因心跳触发而反复报平安。修复实现错误时先保留日志，保持模型、数据、学习率时钟、原始输出与评分协议；从已完成的恢复点处理，不能悄悄从 base 再训或覆盖旧结果。节点 `dkucc-core-gpu-dkurc-d-it09-3` 已观测到一张卡有约 22 GiB 额外占用，本次提交将其排除。

## TensorBoard

本地独立环境 `dkucc/tensorboard/.venv`，TensorBoard 2.21.0；页面 `http://127.0.0.1:6006`。`bridge.py` 只读 SSH 日志，每 60 秒同步训练 loss、LR、grad norm、显存和已完成的验证指标，不改变训练环境或训练进程。曲线刷新与对话 45 分钟检查是两件事。`bridge_status.json` 是连接情况，`processes.json` 是本地后台 PID。

重启本地监看使用 `dkucc/tensorboard/start.ps1` 或“打开训练监看.cmd”。不要重复安装、重复启动同一端口或杀死无关进程。SSH 断开时页面保留已有曲线，训练仍在 Slurm 继续。

## 完成后的交付

沿用用户原先要求：下载所有训练与测试日志并生成此次训练可视化报告。先 `python dkucc/full_attention_v2/sync.py download`，校验归档与逐文件 SHA256。验收/失败历史也应归档：63405、63406、63408、63410、63412、63414；63413 是未开始训练即取消的依赖作业。保留所有原始预测和失败输出，权重和大优化器二进制无需下载。

新报告应呈现精确全局 token 归一化的逐步训练 loss、6 场 teacher loss/熵、完整 dev 的自由生成指标、选择和失败情况、若执行则最终 56 场 test，以及与原始 base/旧 SFT 的可比指标。Trainer 在 callback 提前暂停后的 `train_samples_per_second`、`train_steps_per_second` 和跨阶段 `train_loss` 不适合直接当作完整训练统计；以每步分子分母、真实计时和 Slurm 总时间重算。

生成图后实际查看，报告链接与日志归档必须验证再交付。完成并交付本轮后，若 `automation_update` 可用，按上层指令删除 `moss`；否则保持完成守卫、DONT_NOTIFY，不反复通知工具缺失。
