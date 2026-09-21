# 局部 Unlikelihood：30 步问题样本验证

日期：2026-09-17。设计与验收：主代理；编码与执行：Luna xHigh。

## 本轮问题与范围

检验正常完整会议 CE + 短重复恢复 CE + 局部 UL，在用户指定的学习率 1e-6、30 次更新后，能否消除已知问题会议 R8005_M8009 的病态复读。只评估这一已知难例，不运行完整 Dev / Test，不将结果外推为全数据集修复。

优先 C（恢复 CE + UL），B（相同恢复 CE、UL=0）提供同条件对照，资源允许并行。复用经核对输入及解码协议的 Base 原始生成；旧 seed / LR / 事件协议不同的 SFT 仅作历史背景。本轮不做 A、D 或第二 seed。

资源核实后的执行顺序：账号 `qt28/faculty` 的 `MaxJobs=1`，单作业 `gres/gpu=4`、内存上限 `500000M`，无法并行运行两个 SP4 训练组。按用户优先判断能否解决问题的要求，先完成 C30 + 单问题样本生成；若 C 仍明确病态复读或严重遗漏，本轮直接收口为未解决，不追加 B。只有 C 明显改善、值得进一步归因时才运行已冻结的 B。缺少 B 的结果不能声称 UL 独有增量收益。

## 冻结训练合同

- Base 初始化，新的 optimizer / scheduler 状态，不自动 resume；seed 和 data seed 均为 1。
- 学习率 1e-6，沿用 402-update linear schedule、无 LR warmup；执行到 optimizer step 30 无条件停止。
- 保持官方音频 processor、完整音频、时间标记、full attention、原有全参数训练及精度协议。
- 每 update 4 个正常 source meeting views，另有每 source 1 个辅助事件；30 步共 120 个事件呈现。SP=4 的 rank 不能重复计数。
- B/C 共用冻结 source 顺序与事件清单；重复刚出现的合法正文 token，四个 slot 的总重复长度依次 3、3、4、4。
- 锚点只来自 train，排除结构/时间戳/说话人/音频占位与天然相邻相同 token；正确下一 token 必须不同，且有 16 个真实后续监督 token。
- 插入副本可被注意力看到，但 labels=-100。恢复 CE 仅监督原参考中的下一个 token 及后续 15 token，不增造 EOS。
- 为节约计算，本轮使用确定性合法锚点抽样，不额外运行全训练集模型难例筛选；这一简化限制负例难度覆盖，不能以失败直接否定所有 UL 方案。
- 辅助分支可截去最后一个监督 token 后的文本后缀；因果注意力下这不会改变当前监督所见历史。完整音频与提示仍保留。

每 update 的目标：

`clean = 四个正常会议的有效 token CE 总和 / 四会议有效 token 数`

`recovery = mean_events(exit_CE + mean(next_15_CE))`

`UL = mean_events(-log(1 - p(repeated_token | complete_repeated_prefix)))`

`B = clean + 0.1 * recovery`

`C = clean + 0.1 * recovery + 0.1 * UL`

辅助系数不使用旧版 5 步 warmup。没有合法事件应显式报告/失败，不静默退回 clean 并计为有效事件。UL 仅作用于退出边界，不能持续禁止该 token。

## 启动前实现验收

1. 核对因果移位：最后一份重复 token 的 logits 预测第一个正确退出 token；插入 labels 屏蔽、监督窗口 16 token、原始 clean EOS 保留。
2. UL 用数值稳定形式并在 FP32 求值；高重复概率时梯度不因粗暴 clamp 消失。B/C 的差异仅 UL 系数。
3. 验证梯度累积与 SP 归一化，特别是事件唯一拥有 rank、可导收集、world-size 补偿和零监督 rank；辅助 loss 不除以完整会议长度。
4. 固定模型/processor/数据/事件/代码配置指纹；断言 Base 起点及无旧 optimizer 状态。
5. 使用少量训练数据烟测，确认有效事件、有限 loss/梯度及内存。烟测权重不得作为正式初始化。

## 30 步之后

训练进程退出后，用冻结 checkpoint-30 对完整 R8005_M8009 自由生成。保持原 greedy、repetition penalty=1、presence/frequency penalty=0、相同上下文和生成上限。B/C 与复用 Base 核对 prompt IDs / 输入 / 协议。

记录并保留原始 token IDs、转录、EOS、截断、最长同 token 连续长度、短周期循环、CER/cpCER/删除错误与解析损失。检查循环上下文及参考转录，避免把自然重复直接判成病态。检查中间转录覆盖；最后时间戳到末尾不足以证明无漏段。

结论分层：

- 仍有已确认病态复读，或以提前 EOS、严重遗漏、解析崩溃换取无复读：本轮未解决问题。
- C 无病态复读且完整性保持：仅说明这套配置在该已知样本上通过验证。
- C 优于 B：支持局部 UL 在本轮的增量收益；B/C 都改善则需区分恢复训练本身的贡献。
- 事件/梯度/归一化不合格：实验无效，修正实现后再判方法。

不根据此 Test 样本反复调整系数或延长到 60/150 步。后续是否扩展另行决定。
