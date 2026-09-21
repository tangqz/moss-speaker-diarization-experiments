# MOSS output-only R-SWA, 2026-09-20

用户已授权执行修订实验计划。固定 LR=1e-7，推理 repetition penalty=1.02，作用于完整已生成输出，不包括 reference。

服务器任务目录：`/work/qt28/moss/dkucc/rswa_20260920`。可见 SSH 窗口持续显示 `logs/rswa-20260920-console.log`。

## 已核验

- 单卡数学 mask、前向/梯度、环形 KV、绝对位置、未来隔离：job 64084 通过。
- 两卡/四卡 Ulysses，Full、k=128、k=256：job 64086 通过。
- 官方 processor/collator 与短/最长会议输入一致：job 64086 通过。
- 618 份 train/dev/Test 音频 SHA-256：无重复，划分为 536/26/56。
- 原生 Swift 全参数短会议 SP1、SP4/FSDP，以及最长 116527-token 会议单步：job 64087 完成。最长一步约 226 秒，峰值 38.31 GiB；工程输入是同一会议重复四次，有效 batch=4，不能当作混合会议的平均速度。
- vLLM 原生 Triton kernel 的精确窗口、实际页回收、output-only penalty：job 64088 通过。该 job 的随后真实模型初始化因 override 回调收到 generic dummy config 失败；修复后真实模型检查另行运行。

## 数值复核与联调

- job 64089：SP4/DDP 完成；恢复后两步与连续两步的严格低误差比较未通过。job 64091 证明保存的 684 个权重张量、683 个 Adam 参数状态和调度器精确恢复；job 64095 的同实现重复训练也出现 BF16 差异。保留原始失败记录，以重复噪声及随机状态核验评定，不能宣称训练轨迹逐位相同。
- job 64096：真实 MOSS 的 vLLM FlexAttention / HF 固定历史 logits 对齐，并加入未修改的官方 Full 数值参照。固定长度仅用于工程核验；正式质量评测仍正常 EOS 停止。
- job 64097：四卡保存恢复的随机状态与有限梯度检查。
- job 64097 已通过：4 卡 Python/NumPy/CPU/CUDA RNG 全部精确恢复；模型、Adam 和调度器状态精确恢复，继续更新成功。
- job 64098：典型完整会议 k256 四卡更新通过，77.7 秒/update，18.4 GiB 峰值。Flex 推理典型会议约 10 token/s，最长预填充超过五分钟，因此主动停止此工程作业并保留原始结果；不能算作最长推理通过。
- job 64276 已完成：Triton 的短、典型和最长会议 R128/R256 长轨迹检查通过结构性核验；28 层 reference KV 未变，HF storage=P+k，vLLM 实际页回收保持上界。所有轨迹均无 margin>0.5 的首选 token 分歧。原始 BF16 固定误差阈值仍有失败，保留原报告。
- job 64279 的典型 Full 参照已完成。施加 1.02 penalty 后，典型 R128/R256 最大 KL≈0.01690、TV≈0.09105，与自定义 Full 在同一 pre-window 位置一致；最长 R256 最大 KL≈0.00185、TV≈0.02702。此项是数值复核依据，不是精度或质量提升证据。
- job 64279 的动态 attention 前向/梯度 42 案例通过，仅产生 3 个编译图；SP4 动态路径通过；12 个不同长度真实会议连续 3 次 R128 更新完成。导出与自由生成完成；CE 的 processor 接口问题修复后由 64284 只补跑 CE/评分，四份预测未修改，联调通过。
- job 64287 完成真实短/典型会议的四种模式全参数梯度核验。自定义 Full 对原生 Full 的梯度余弦为 0.999786/0.999868；不约束大窗口与自定义 Full 的 CE 完全相同，梯度差异低于原生重复参照。64107 的 processor 方法错误和 64285 的 CPU 卸载 stride 错误均保留原记录。
- job 64282 已通过实际 model forward 的四卡顺序与保存恢复核验：两次更新共 8 个唯一会议、四卡顺序一致。顺序日志排除 dataloader 预取，正式每个 epoch 校验 536 条唯一会议。
- 同一 64279 allocation 内的独立 300-token 计时检查已通过：每次请求前清空 GPU 音频 encoder cache，并逐 token 核对固定轨迹。工程质量 pilot 的旧计时不作为正式冷缓存速度结果。
- 完整 dev、正式三组训练、Test 和性能结论尚未完成。

完整数值放行结论见 `evidence/task/numerical_qualification.json`。原始严格 BF16 阈值失败标记不被改写；本次放行限定于同一后端的预注册实验。3 步工程模型的两条 Dev 输出虽然达到 EOS，但可解析文本和时间覆盖不足，不代表质量合格，也不进入正式选型。

## 实现约定

- `attention.py`：28 层解码注意力，完整 causal reference + 当前输出位置起向前 k 个位置；稀疏 block mask；训练不使用 KV cache。
- `cache.py`：HF 推理固定 P+k 存储，位置继续使用绝对逻辑编号。
- `moss_rswa_plugin.py`：沿用原生 Swift trainer、优化器、标准 CE、官方音频处理，显式保存窗口边界与输入证据。
- `moss_vllm_rswa.py`：复用固定 vLLM 版本的 RSWAAttention/RSWAManager；不修改共享安装环境；仅适配 Qwen3 解码层和 penalty 历史范围。
- `input_receipt.py`：保存完整 prompt IDs、时间标记位置、音频 token 数和 chunk mapping，核验实际 12.5/5/True 配置。
- `evaluate_generate.py`、`evaluate_ce.py`、`run_eval.py`：一致解码、标准 CE、已有 CER/cpCER/DER 口径；保存 prompt/marker/chunk 及重采样长度，每次计时前清空音频 encoder cache。
- 每个 Slurm job 启动时复制独立 source；后续部署不会改变正在执行的快照。

## 运行边界

只在自己的 GPU allocation 内做模型计算。当前账户最多一个运行作业、四张 GPU，后续验收串行排队。失败关卡不得被标记为通过；不修改 loss、截短会议、丢弃最长样本或冻结音频塔来绕过问题。

原始计划：`../reports/MOSS_输出KV滑动窗口_实验计划_20260920.md`。
