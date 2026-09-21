# MOSS 原生 MS-Swift full-attention SFT

这次迁移采用会议提出的“注册模型和模板，使用框架训练器”路线。旧训练实现留作历史归档；本目录不导入旧的 Trainer、memory_sft、forward、音频 FFN 分块或 loss 补丁。

## 实现边界

- `moss_plugin.py`：注册官方 MOSS 模型、模块路径和数据模板。模型加载后仅关闭训练 KV cache、恢复官方权重绑定。音频和标签由官方 processor / DataCollator 处理。
- `native_sft.sh`：真正调用 `swift sft`；Trainer、优化器、CE、梯度累积、序列并行、保存和恢复均使用框架实现。
- `runtime_callback.py`：使用框架公开 callback 接口记录参数状态，并在每 5 步的阶段边界停止进程、保存，释放 GPU 供生产推理使用。不修改梯度或 loss。
- `pipeline.py`：调度原生训练与 dev 推理。每阶段从原生 checkpoint 恢复 optimizer、scheduler 和 RNG。调度器不是自定义 Trainer。
- `evaluation/`：复用 CER/cpCER/DER、原始输出解析、输出诊断和 dev 停点规则；生成改用官方 vLLM MOSS 实现。不含旧训练代码。

## 保持与改变

| 项目 | 本次设置 |
|---|---|
| 初始化 | 同一官方 Base，重新开始，不从旧 SFT checkpoint 继续 |
| 数据 | 原 536 条完整会议训练记录，原 26 条 dev；目标文本不变 |
| 训练范围 | 全参数，音频编码器、连接层和语言模型均可训练 |
| 注意力 | full causal attention，无 KV mask |
| 长度 | 最大 131072；不切短会议，不打包；严格拒绝超长样本 |
| 学习率 | 1e-6；沿用原 linear scheduler、402 步衰减尺度、无 warmup |
| 优化器 | 原生 fused AdamW，betas 0.9/0.999，epsilon 1e-8，weight decay 0，梯度裁剪 1 |
| 数值精度 | FP32 参数，BF16 混合精度 |
| 全局 batch | 4 条会议；目标 SP=4、DP=1、每卡 microbatch=1、累积=4 |
| 随机顺序 | CLI seed/data_seed=0；当前 Swift SP sampler 内部固定使用 42，与旧自定义 sampler 顺序不同 |
| Loss | 每步原生 train CE；每 5 步对完整 dev 计算原生 eval_loss，首轮训练前也测一次 |
| Dev 准确率 | 每 5 步 vLLM 运行完整 26 会议，BF16 greedy、不加防复读解码参数；Base 用同一引擎重新测 |
| 停点 | 复用 cpCER/DER 的资格检查；min_delta=0.1、patience=3、tolerance=0.1；灾难性输出立即停 |
| 试运行上限 | 50 次更新；scheduler 总尺度仍为 402，不把阶段边界当作学习率重置 |
| 测试集 | 不参与选择；本次试运行没有自动测试集循环 |

学习率等沿用已有配置，目的是先观察框架迁移的影响，不声称 1e-6 已被证明最优。框架原生 eval_loss 的汇总方式应与旧报告的全 token 加权 loss 区分，不能直接拼为同一条曲线。

## 版本与路径

- MS-Swift：`0673cf75dca7d0b9b608b4a76632fb508ead5076`，安装报告 4.6.0.dev0。
- Transformers：5.12.1（当前 Swift 要求 `<5.17.0`，因此独立安装，不改旧生产环境）。
- PyTorch：2.14.0+cu130。
- 官方 MOSS 源码：`61bc29cd4120be7b5d3b761b64cd5dff57263642`。
- 新环境：`/work/qt28/moss/envs/ms-swift-20260914`。
- 服务端工作目录：`/work/qt28/moss/dkucc/ms_swift_20260914`。
- vLLM 独立环境 `/work/qt28/moss/envs/vllm-moss-20260914`，固定官方 MOSS README 指定 commit `68b4a1d582818e67adc903bf1b8fc5a5447da2fa` 的 wheel `0.23.1rc1.dev949+g68b4a1d58`。实际依赖记录于 `vllm_packages.txt`。
- 旧 HF 推理只保留作历史证据和工程对比，本轮 dev 选点全部使用同一 vLLM 引擎的新 Base/SFT 结果。

## 验证与启动

`validate.slurm` 依次检查最短/最长样本张量一致性、单卡一步、SP4 一步、最长会议 SP4 一步。重复的 4 条 smoke 记录只用于工程验证，正式训练仍使用全部 536 条记录。原生单卡与 SP4 的数值和保存、生产加载也需检查；不能把“进程启动”写成训练通过。

`train.slurm` 必须提供已成功完成的 `MOSS_GATE_RUN`、`MOSS_RESUME_GATE`、`MOSS_VLLM_GATE`；`pipeline.py` 会再次核对训练、恢复、vLLM 长音频验证结果和数据哈希。具体进展见本目录 `current_run.json` 与 `FOLLOWUP.md`。

SP4+FSDP1+SDPA 已在 63506 完成短会议及最长 116527 token 会议的一步更新和保存，最长峰值 38.01 GiB。MOSS 此组合并非上游已列明支持的模型组合；后续升级仍需回归验证，不引回旧训练补丁，也不悄悄冻结音频编码器或裁短会议。

实际长会议验证发现两处显存瓶颈：Whisper FFN，以及 SDPA 显式掩码。当前验证配置为 SP=4 + FSDP1 + 框架 activation_cpu_offload，另启用原生 `CELOSS_PARALLEL_SIZE=512`，仅分块计算同一 CE。FSDP2 的原生卸载在此版本触发 DTensor storage 访问错误，保留日志，不修改框架源码。

模板的 `MOSS_IMPLICIT_CAUSAL=1` 模式只接受单条无填充会议，将长度向右补至 SP 的整数倍，额外标签全部为 -100，位置单调连续，并省略冗余的全 1 attention_mask。因果关系保证这些尾部位置不能影响任意真实 token。这样也避开当前 Transformers 将 SP 的 -1 补齐位置误判为打包边界的错误。最短/最长真实输入前缀、监督标签和全部音频张量另行逐项检查。未来 KV mask、真正的多样本 padding 或 packing 必须关闭此模式并提供相应掩码；不能直接套用。

FSDP1 会使用框架原生混合精度和梯度规约，计算结果不应被描述成旧实现的逐位复现。升级 Transformers/MS-Swift 时需重新验证 FSDP 与 SP，尤其当前 FSDP1 已进入弃用周期。

未来 KV-mask SFT 与熵 RL 可以复用同一注册、数据和评估层，但需要独立验证相应训练/推理 attention 语义；本次不提前加入 mask 或 RL 目标。

## vLLM dev 评估协议

原生 teacher-forced dev CE 与生成评分是两项不同测量：CE 仍由 Swift 计算；vLLM 生成完整会议供 CER/cpCER/DER 和复读诊断使用。每阶段训练进程退出后，4 张 GPU 各运行一个 vLLM 引擎（TP1），与训练 SP4 分开配置。完整上下文 131072，最多生成 65536 token，温度 0，无额外重复惩罚；音频、prompt、EOS、parser、评分和停点规则保持一致。

`export_vllm.py` 为推理单独整理目录：模型 config 保持原样，补上 Base 的原版 processor 配置、远程加载映射和代码。没有额外输出权重别名时使用符号链接；若原生 checkpoint 同时保存 lm_head.weight 和已绑定的输入嵌入，则先验证共享配置、形状、dtype 和完整张量 SHA-256 完全一致，再仅在推理副本中去除重复别名。所有剩余张量数据重新读取并验证字节哈希相同，不做 dtype 或数值转换。原因是 Swift 的保存结果只有自定义 processor_class，没有完整 AutoProcessor 映射。不会改写训练 checkpoint，也不会转换权重。独立加载该处理器的 CPU 检查已通过。

`launch_vllm_training.py` 是服务器端一次性接续流程：等待安装成功标记，提交 vLLM 长短输入及 128-token HF 对照检查，核实成功凭据后提交正式训练。任何验证失败都会记录到 `vllm_dispatch.json` 并阻止训练启动。当前进度以该文件及服务器 `current_run.json` 为准；代码接入完成不代表 vLLM GPU 验证或完整 dev 跑分已经完成。

先核对 Base/SFT 的最短及最长 dev 展开后 prompt token 与官方 processor 完全相等，再做生成。128-token 检查只证明加载和长输入可运行，不作为完整准确率或加速倍率证据。正式 Base 重新生成 26 条，再在本次任务内缓存；旧 HF Base 分数不参与本轮选择。记录包含引擎启动时间和全 dev 墙钟时间，vLLM 独立工作进程不提供此版本可直接安全调用的逐请求 allocator 峰值，因此该字段记为 null；初始化显存、KV cache 和图捕获开销保留在原生引擎日志。评分脚本只对显存缺失值作兼容处理，CER/cpCER/DER 数学保持不变。

TensorBoard：本地 http://127.0.0.1:6006 。原生 train/loss、eval/loss、学习率、梯度范数由 Swift 写入；evaluation_events 写入 vLLM dev selection_score、各语料 CER/cpCER/DER025、截断/空输出数、资格与灾难性标记、完整评估耗时。Base 评分在 step 0、SFT 评分在真实 optimizer step；本轮在 step 5、10、15、20 完整评估，用户要求 step 20 后改为 step 30、40、50，禁止用 128-token 工程检查填充完整 dev 指标。同步进程每 60 秒读取服务器 current_run.json 自动跟随新任务。
