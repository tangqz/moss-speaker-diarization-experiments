# MOSS full-attention SFT 接续记录

本目录接续 Proma「说话人日志任务执行」，以 `../A40_动态RSWA与熵门控RL_兼容性评估_2026-09-11.md` 第 7–8 节为实施依据。

## 连接

已实测通过的共享连接方式：

```powershell
& 'C:\Program Files\Git\usr\bin\ssh.exe' -T -O proxy -S C:/Users/qizhi/.ssh/cm-moss qt28@dkucc-login-01.rc.duke.edu 'hostname; squeue -u qt28'
```

`-O proxy` 使用 OpenSSH 的 multiplex proxy 模式，无需 passenger 模式的文件描述符传递。Proma 的现有主连接必须仍然存活且已完成认证。见 [OpenSSH PROTOCOL.mux §9](https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.mux)。本目录上一级的 `ssh_moss.py` 支持命令、脚本与二进制文件传输；不保存口令、私钥或 Duo 信息。

## 修复边界

Proma 的旧 `lean_loss` 版本仍会构建整个序列乘词表大小的 logits；Accelerate 随后将其转为 FP32。其对 `_convert_to_fp32` 的赋值没有修改 `convert_to_fp32` 内部的同名局部函数。

本实现仅修改当次加载的 MOSS 实例的带标签前向：

- 沿用官方完整音频、完整历史、SDPA full attention 与 backbone。
- 对有效监督位置分块执行词表投影、FP32 交叉熵与重算；每块默认 512 token。
- 沿用因果 label shift 与 Trainer 的有效 token 分母，跨卡进行相同目标的归一化。
- SFT 只返回标量 loss，避免无用的完整 logits 被保留与上采样。
- Whisper 各层只对逐 token 的 `fc1 → GELU → fc2` 前馈网络分批重算，默认每批处理 8 个原有音频块；attention 的批次、完整会议的拼接、顺序和内容保持不变。无 dropout 的当前配置在安装时显式检查。
- 无标签推理调用官方原始 forward；checkpoint 的参数结构不变。

此 SFT 接口不返回逐 token logits，不可直接当作后续熵门控 RL 的实现。RL 阶段需另行实现精确的分块 entropy/log-probability 统计，并验证 rollout 与重算一致性。

## 验证与运行顺序

`baseline.slurm` 申请同一节点 4 张 A40，采用 DDP、SP=1、bf16、gradient checkpointing、单卡 batch=1。

1. 同一作业内核验四卡拓扑与 NCCL。
2. 对比 dense CE 和分块投影 CE 的数值与梯度，包括全忽略样本、不等长标签、梯度累积及四卡归一化。
3. 使用真实 MOSS 与短音频对比原版/修改版损失、绑定权重梯度和无标签推理结果。
4. 检查 536 条训练记录的路径、去重、与验证/测试音频路径交集；调用官方音频分块长度函数与文本展开函数，精确核验全部记录的长度、监督 token 与 EOS，并对最短/最长记录与实际音频特征处理结果交叉核验。超长直接失败，不进行隐式截断。
5. 用覆盖短、中、最长记录的八条样本进行四卡两步训练，保存 checkpoint，再恢复至第 3 步。
6. 以上成功后开始全部 536 条记录、3 epochs、学习率 1e-5 的 full-attention SFT。每 25 步保存，最多保留两份训练 checkpoint。

接续作业记录：62987 通过四卡 NCCL 与分块损失梯度检查；62988 通过真实 MOSS 对照；62990 完成全部 536 条数据检查，四卡试跑发现 GPU 0 的约 22GB 残留显存占用。62991 的资源探针确认进程 116725 在驱动中占 22602 MiB，而系统进程表中已不存在。

对 rank 0 的 backbone 启用 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 激活卸载，其他三卡保持原路径。62992 完成八条完整会议的四卡两步训练与 checkpoint 保存；loss 为 0.5981、0.3988，四卡最大进程峰值为 38.561 GiB。恢复时最长会议落到受限 GPU 0，Whisper 前馈层反向峰值仍超出其可用显存，因此加入上述前馈分块。

62994 使用 `resume_audio.slurm` 接续原 checkpoint-2。其 Whisper 对照通过：9 个音频块按 2 块处理，输出完全相等；抽查 conv1、首层 Q 投影、首层 fc1 的梯度相对差异分别为 2.109%、0.481%、0.708%，低于测试设定的 BF16 4% 相对容差。测试通过后以每批 8 块恢复四卡训练。此变更影响吞吐和 BF16 梯度舍入，保持 attention 与损失的数学目标。

数据报告 `preflight-62990.json`：536 条，最短 8585、中位 46586、P95 64694、最长 116527 token，held-out 音频路径交集为 0。恢复脚本通过 SHA-256 确认训练 manifest 未变，再复用该结果。状态以服务器日志为准；提交不表示训练或验证已通过。

服务器代码：`/work/qt28/moss/dkucc/memory_sft/`。
日志：`/work/qt28/moss/logs/moss-fa-ddp4-<jobid>.out`、`.err`。
数据核验：`/work/qt28/moss/results/preflight-<jobid>.json`。
训练输出：`/work/qt28/moss/checkpoints/full-attention-ddp4-<jobid>/`。

本次先启动 SFT 工程基线。当前服务器 `results/` 在交接时为空，未发现完整零样本 CER/DER/cpCER/DeltaCP 报告；效果对比仍须补齐，不能仅凭训练 loss 宣称质量改善。固定/动态 R-SWA、MS-Swift 注册与熵门控 RL 不在本次训练中启用。
