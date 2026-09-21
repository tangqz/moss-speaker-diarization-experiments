# MOSS 短程错误恢复实验：实施规范 v2 sparse

设计日期：2026-09-15。负责人：主任务完成设计与审查；`gpt-5.6-sol` 实施代码。

状态：技术设计完成；Sol 首版实现已通过主任务的 21 项 CPU 测试、四组参数与脚本语法检查。尚未进行本实验 GPU 验证或训练，完整的 GPU 数值比较与整会评估自动化工具仍需补全。本文数值是预注册的试验设置，不是已验证最优参数。采样条款取代 v1 的强制排除正确答案方案。

## 1. 研究问题与边界

主问题：在正确历史中最多出现一个局部错误事件，监督其后 16 个原始正确 token，是否改善带错误历史的持续转录？显式加入总长 2–3 的短重复，是否比单 token 替换有额外收益？模型候选 98% 取 temp=0，2% 才温度采样，正确候选保留；短重复仅以不超过 5% 的名额触发率加入。

先做共同 SFT 权重上的 30 步恢复微调 pilot。它回答“已有 SFT 模型能否修复”，不直接回答“从 Base 训练能否预防循环”。后者是 pilot 有效后的独立实验。

所有训练项均为正确标签的普通交叉熵 CE。新增的是输入扰动、监督位置和 CE 权重。它是独立的数据增强实验，不能继续称为完全未经扩展的原生 CE baseline。保留干净 baseline 分组。

第一版不加入长循环训练、unlikelihood、RL、动态 KV mask、音频冻结、LoRA、overlap loss mask、解码重复惩罚或难度挖掘。训练中每个辅助视图最多一个错误事件，也允许完全没有错误；后续输入恢复使用正确 token，已发生的事件始终留在历史里。

## 2. 核对过的工程事实

- 当前训练入口：`../ms_swift/native_sft.sh`；官方模型/processor 注册：`../ms_swift/moss_plugin.py`。
- MS-Swift 本地上游快照：`0673cf75dca7d0b9b608b4a76632fb508ead5076`。官方 MOSS 源码记录为 `61bc29cd4120be7b5d3b761b64cd5dff57263642`，运行前重新核验。
- 当前完整会议设置：full parameter，FP32 master weights + BF16 autocast，full causal SDPA，SP=4、DP=1、FSDP1 + activation_cpu_offload，microbatch=1，`CELOSS_PARALLEL_SIZE=512`，上下文上限 131072，训练 KV cache 关闭。
- 既有最长样本为 116527 token；旧入口曾通过最长样本更新，不代表新增的采样/恢复入口已经通过。
- 官方 collator 先产生完整 `input_ids` 与未移位 `labels`，prompt/audio 对应标签为 -100，真实结尾监督 EOS。
- `swift/sequence_parallel/sequence_parallel.py::pad_and_split_inputs` 会在 SP 切分前对 labels 和 loss_scale 左移一次。`per_token_loss_func_sp` 接受已移位标签；SP=1 路径自己移位。
- `swift/trainers/seq2seq_trainer.py::compute_loss` 已支持逐 token `loss_scale`，并用整个累积窗口的非忽略标签数归一化。因此不需要自定义 CE 或 backward。
- **现有 SP dataloader 忽略 `train_dataloader_shuffle=false`**：`get_sp_dataloader` 构造 `SequenceParallelSampler(..., seed=42)`，未传入 shuffle。成对有序视图必须由本实验显式的顺序 dataloader 保证，不能只改 CLI 开关。
- 2026-09-15 实时只读检查确认 `.../ms-swift-63643/training/checkpoint-150` 存在；本地 `raw/selection.json` 按 dev 选择 step 150。旧 D0/D1 汇总仍有 vLLM 项 missing，不作为路径一致性通过凭据。

### 固定路径与数据身份

| 用途 | 路径/标识 |
|---|---|
| 初始化权重 | `/work/qt28/moss/results/ms-swift-63643/training/checkpoint-150` |
| Base tokenizer/processor | `/work/qt28/moss/models/MOSS-Transcribe-Diarize` |
| 官方源码 | `/work/qt28/moss/MOSS-Transcribe-Diarize` |
| 训练环境 | `/work/qt28/moss/envs/ms-swift-20260914` |
| 已安装 Swift 源码 | `/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076`，editable 4.6.0.dev0；已用当前远程 Python 核对 |
| 训练清单 | `/work/qt28/moss/dkucc/ms_swift_20260914/data/train.jsonl`，536 条 |
| 训练清单 SHA-256 | `ff24eae8c1b2f9650654b3bc84b005e55d0206bb7d202d8f61b14bfb6e6d064f` |
| Dev 清单 | `/work/qt28/moss/dkucc/ms_swift_20260914/data/dev.jsonl`，26 条 |
| Dev 清单 SHA-256 | `3a3f3cce9966ae2a7d7ada1140a5db48b1efaabdbb1e19db5ef8ce0b51c2517e` |
| 计划部署的独立代码包 | `/work/qt28/moss/dkucc/short_recovery`（当前交付在本地；尚未部署） |
| 新结果目录 | `/work/qt28/moss/results/short-recovery-<run-id>/<arm>/seed-<seed>` |

不得修改旧模型、旧任务状态、旧评估结果或全局环境。新启动入口必须记录实际解析后的路径、全部源文件/配置哈希和模型权重哈希。

## 3. 四组对照与初始化

| Arm | 每条源会议的视图 | 目的 |
|---|---|---|
| `ce` | 1 个完整干净 CE 视图 | 同起点继续普通 CE |
| `clean_aux` | 完整干净 CE + 正确历史下的短续写 CE | 控制额外训练/短续写监督的收益 |
| `sub` | 完整干净 CE + 单锚点模型输出后的短续写 CE；输出正确时保持干净 | 局部识别错误恢复 |
| `mix` | 完整干净 CE + 50% 模型输出名额、50% 低概率短重复名额；未发生错误时保持干净 | 检查短循环的额外价值 |

`clean_aux/sub/mix` 使用相同源会议顺序、锚点、恢复标签、辅助监督预算和 lambda。`mix` 的短循环占用原有辅助名额，不额外增加样本。`sub` 与 `mix` 的替换候选可因训练权重不同而不同，这是在线采样的预期差异；记录实际事件。

每组都仅加载共同 step-150 **权重**，新建相同优化器、scheduler 和 RNG；新实验 step 从 0 计数。不可只为某一组继承旧优化器。需要原生断点恢复时才加载本实验自身全部训练状态。

默认：seed=0，30 次 optimizer update；训练池仍是全部 536 条，每步 4 条源会议，所以 pilot 只消费冻结顺序的前 120 条，不得称作已遍历全部训练集。后续完整一个源 epoch 为 134 步，重复 seeds 1、2；这些是后续配置，不自动启动。

优化参数：LR=1e-7，linear scheduler horizon=402，warmup=0，AdamW fused，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，max_grad_norm=1。scheduler horizon 与 30 步停止条件分开，沿用 callback 停止，不把 `max_steps=30` 当成 30 步衰减。

每步固定 4 个 source slots。`ce` 累积 4 次；其余累积 8 次，顺序为 c0,a0,c1,a1,c2,a2,c3,a3。同一累积窗口内无 optimizer update，probe 与其 clean view 的模型权重一致。禁止自动重排、packing、left padding、多个会议合为一个 microbatch；v1 生产只支持 SP4/DP1，SP1 仅用于小样本正确性检查。

## 4. token 坐标与错误事件的精确定义

完整干净输入 `z = prompt_audio || y || eos`，索引从 0 开始。锚点 `a` 是 `z` 内的正文 token 绝对位置，正确 token 是 `z[a]`。采样 logits 来自 **query=a-1**；它只看 `z[:a]`，不能用 logits[a]。

统一恢复标签为原序列 `z[a+1:a+1+K]`，默认 K=16。锚点必须保证这个窗口实际存在；不能用 padding 补足标签。正文结构在此窗口内可以自然出现，按真实顺序监督；至少 4 个窗口 token 必须完全处于正文 span。EOS 仅在原始真实结尾位于窗口内时监督。

### 4.1 替换 `substitute`

`z[a] -> v`，要求 v != z[a]。例如 `A B C D E` 变成 `A B X D E`，恢复目标从 D 开始。

- 新 input length 不变。
- aux labels 在原位置 `[a+1,a+1+K)` 等于正确原 token，其他位置全部 -100。
- 错误位置 a 不监督 X，也不监督 C；训练从带 X 的历史继续。
- 后续输入 D、E 正确且 X 保留；不在下一步把 X 改回 C。

### 4.2 重复插入 `repeat_previous`

在 a 之后额外插入 r 份 `z[a]`，r∈{1,2}。总连续出现次数 R=r+1，R∈{2,3}，各占一半。例：`A B C D E -> A B C C C D E`，仍从 D 恢复。

- 新 input length 增加 r；后续真实 token 的位置映射为 `old_index+r`。
- 额外副本位置 `[a+1,a+1+r)` 全部 labels=-100。
- 恢复监督位置 `[a+1+r,a+1+r+K)` 对应原始 `[a+1,a+1+K)`；不能跳过 D 或 E。
- 原 C 也不在 aux 中监督；完整干净视图仍监督全部原标签。
- 要求原序列 a 周围无相同 token 的真实相邻重复，且 `z[a+1] != z[a]`，以免把真实重复作为错误。

### 4.3 后续可选扩展

`substitute_then_repeat`：先把 C 替换成 v，再插入 r 份 v，目标仍从 D 开始，不能混同纯插入。v1 的 transform 可支持并测试，但默认训练配比为 0。长于总长 3 的重复不进入训练。

## 5. 选位置、选候选与跳过规则

### 锚点资格

用实际 tokenizer 的 offsets 与参考 target 中完整合法片段的 text span 取交集。格式记号 `[start][Sxx]text[end]` 中 start/end 是数值时间戳占位说明，实际记录例如 `[3.57][S01]Uh 'kay.[3.87]`，不能把 start/end 作为字面字符匹配；正文是数值起始时间和说话人标签之后、数值结束时间之前的 capture group。单片段要求 end>=start，片段之间允许 overlap。锚点 token 必须完全落在一个正文 span 内，并且独立 decode 无 U+FFFD、无控制符、非空/纯空白、无 `[`/`]`、不在 special IDs 中。跨正文/结构边界的 token 一律不选。

单 token decode 再 encode 必须恰好返回同一个 ID；这是保守的词表资格过滤，避免把部分 UTF-8 字节当完整字符。允许多字符 token，不把 token 数称作汉字数。原始序列从不 decode 后重新 tokenize；只按原 token ID 操作。

保护音频占位符、prompt、时间戳、说话人标签、EOS、BOS、PAD。真实数字出现在正文时可保留。训练语料结构解析失败的样本保留原始 clean CE，但不得强行构造事件；记录失败原因。

候选锚点按文本进度四分位分层。各 source slot 的目标四分位由稳定哈希(seed, source_key, occurrence)选定，在该区间的合法位置中均匀采一个；该区间空则从其他区间选择并记录。全无合格锚点是数据资格失败：pilot 规划器应报告并阻止生成宣称“全匹配”的计划，不能静默删会议。K/过滤规则只能通过新配置版本改变。

### 低概率扰动与 temp=0 主分支（用户最新要求）

在同一个更新前的模型权重上进行无梯度、eval 模式、BF16 compute 的正确历史 probe。默认使用当前模型而非冻结 Base、旧预测文件或 test failure token。

训练配置版本为 `short-recovery-v2-sparse`。默认值：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `model_history_probability` | 1.0 | 已选择的单锚点采用模型候选的概率；并非输入出错概率 |
| `sampling_probability` | 0.02 | 采用模型候选时，进入温度采样分支的概率 |
| `sampling_temperature` | 1.0 | 仅用于上述 2% 分支；其余 98% 为 temp=0 |
| `repeat_probability` | 0.05 | mix 中分配为 repeat 的辅助名额真正插入重复的概率 |

所有概率必须在 [0,1]，temperature>0。默认 sparse 配方不允许 sampling/repeat 超过 0.05；扩大概率须显式声明新的非默认实验配置，不能在候选不足时自动提高。配置允许概率为 0，必须验证其准确退化行为。

每个 substitute 名额按以下顺序处理：

1. 以 `model_history_probability` 决定是否采用模型候选；未触发则 clean_aux。
2. 触发后，以 98% 概率使用**完整原始词表的 argmax**。相同 logit 按最小 token ID 确定；若 argmax=gold，保留 gold；若 argmax 非法，回退 clean_aux 并记录 `illegal_argmax`。不得把第二名替上去制造错误。
3. 仅 2% 进入温度分支。用 FP32 logsumexp 得到完整词表 p；候选保留合法正文 ID，要求 `p(v)>=1e-4` 且 `p(v)>=0.01*p(gold)`，按 p 取前 32，**额外保证 gold 在集合内**（最多 33）。稳定平分按 ID。若候选原始概率总质量<0.5，回退 clean_aux；不能归一化极低概率尾部制造错误。
4. 在此集合按 `q(v)∝p(v)^(1/T)` 采样，允许选中 gold。选中正确答案直接保持干净，不重抽、不排除正确 token、不强制错误。
5. 仅最终候选合法且 !=gold 时调用 substitute 变换。K 个后续原始标签在所有分支相同。

logits 出现 NaN、+inf 或全部 -inf 时直接报错，不能作为干净 fallback 掩盖数值故障。合法的 -inf 表示概率 0。采样/分支 RNG 由 sha256(run_seed, source_key, occurrence, anchor, 独立 event_stream)生成；禁止 Python 随机 hash；协调 rank 决策并广播。

这里的“98% temp=0”指**选中锚点的候选来源**。每条完整序列仍只选一个锚点，其他历史保持原始正确 token，保证单事件研究边界；不把整场会议的全部历史替换成 argmax。因而全语料的输入 token 改动率远低于锚点错误率，不声称它等于测试 CER。

设 e0 为锚点上合法 greedy 错误比例，eT 为温度分支实际错误比例，η 为采用模型历史概率，ε 为温度分支概率，则 `e_anchor = η[(1-ε)e0 + ε eT]`（非法/质量不足回退计为未改动）。默认 ε=.02，所以引入随机采样所带来的绝对错误率增量最多 2 percentage points；模型本身的 temp=0 错误仍被如实保留。

### 用现有错误指标作量级参考，Dev 标定实际概率

本地已读的 step-150 完整 Test 结果：AISHELL-4 CER=13.9291%、AliMeeting=20.5111%、AMI=17.9452%；对应 cpCER=14.3033%、17.0004%、12.2674%。来源为 `../artifacts/ms-swift-63643/raw/evaluations/test-150/metrics_summary.json`，56 场 SFT 预测，AliMeeting 有 1 场截断。它们是已存在且已看过的指标，包含字符级插入/删除及长循环影响；不把某个百分数直接设置成逐 token 随机替换率，也不据此自动提高扰动。

正式运行前在 Dev26 和共同初始化权重上做一次固定 greedy token probe，每会议最多 64 个均匀分层的正文位置，记录未筛选正文与可用锚点两个分母。使用正确历史、query=a-1、完整词表 argmax、BF16 eval；按位置小块收集必要 logits，禁止汇聚全序列 logits。记录总体与各 corpus 的 token top1 错误率、非法输出率、数量/置信区间、模型和数据/配置哈希。

冻结的保守标定规则：ε不超过0.02；`repeat_probability=min(0.05, Dev合法锚点greedy错误率)`。以候选分布非 gold 的概率估计 eT，低质量 fallback 计 0；混合估计为 `emix=(1-ε)e0+εeT`，`η=min(1,(e0+0.02)/emix)`，emix=0 时 η=1。此式是额外噪声上限保护，不靠提高温度逼近 CER。校准报告必须记录完整候选集合质量与随机分支误差的估计，不能只用 Test CER 换算。初始 JSON 的 .02/.05 用于准备和测试，GPU 正式训练前必须绑定该校准产物；不得将尚未计算的 Dev 错误率填成已知数值。概率在共同起点标定后对所有 arm 冻结；训练期间报告实际错误率变化，不自动追着训练误差提高噪声。

### 稀疏短重复与相同监督预算

每个非 ce arm 每条会议都有一个辅助 slot，**辅助视图比例 1 不代表扰动概率 1**。sub 的名额用于上述模型历史策略；mix 计划 50% 模型历史、50% repeat_previous。每个 repeat 名额独立以校准后 repeat_probability 触发，否则 clean_aux；总长 2/3 在名额规划上各半，实际触发后的比例只报告，不为了配额补造错误。repeat=0 时不得插入任何 token。

repeat_previous 是定向插入增强；记录 query=a 下再次输出 z[a] 的原始概率，不能称其 .05 触发率为模型自身插入率。替换与插入均最多一个局部事件。保持 K 标签和相同 aux 权重，不因事件稀少把恢复 loss 自动放大。所有 control arms 可运行并丢弃相同 probe 获得计算对照；ce 保留轻量原 CE 参考，另报 GPU-hours。

采样与候选筛选 stop-gradient：不通过离散 token 或 q 反传，不加 REINFORCE 项。本次优化是给定构造历史下的正确标签 CE；不声称是包含采样分布参数导数的完整期望风险梯度。替换后原来那个错误输出本身不会被追溯改正，验收关注其后的正确推进，并保留整会 CER 中的原错误。

分别记录：候选采用/尝试率、temp>0 分支率、raw argmax 错误率、argmax 非法率、sample=gold 数、实际替换数、实际重复插入数；并报告“事件数/aux名额”“变化或插入token数/原始正文token数”两个不同分母。分组实际事件率可能不同；`mix-sub` 首先比较整体策略，不能单凭此归因于重复形状。sub 零错误则标无有效处理。30 步只有 120 个辅助名额，mix 的 60 个 repeat 名额在 5% 下平均仅 3 个事件；它是工程 pilot。任一拟比较的错误类型少于 20 个实际事件时不作该类型效能结论；20 也只是最低报告门槛，不能代替统计功效分析。不以提高默认概率补足，后续用更多训练步数/seeds 检验。

## 6. 损失与预算：避免恢复 CE 被长会议淹没

对一个 optimizer update，4 条干净会议所有有效标签数之和为 Nc；4 个 aux 窗口有效标签总数 Na=4K=64。记全 token CE 之和为 Sc、Sa。

`L = Sc/Nc + lambda(u)*Sa/Na`。

`lambda(u)=0.1*min((u+1)/5,1)`，u 是本实验从 0 开始的 optimizer update 序号。前 5 步线性增至 0.1。ce 的 lambda=0 且不提供 aux。loss 权重在分组间完全相同；mix 不能因两类错误而翻倍权重。

使用原生 `loss_scale` 实现，不覆盖 `compute_loss`：设 N=Nc+Na，干净有效 labels 的 scale=N/Nc，aux 有效 labels 的 scale=lambda*N/Na；其余 scale=0。框架以 N 归一化后恰好得到上述目标。

必须提前规划每个 update 的 Nc/Na 和所有 row 顺序，并在运行时核对实际计数。HF 在 forward 前会预取整个累积窗口并计算 num_items_in_batch；因此 aux 在 collator 阶段就必须只有 K 个有效标签。在线插入只能移动这些标签，不能改变有效数量；fallback 也维持 K。整个窗口缺行/跨错 update 必须报错。N 的数学含义是 DP1 的唯一源标签总数，不能把 SP rank 副本重复计为额外样本。

在当前 Swift 中 SP 计数、GatherLoss.backward、Trainer 的 world-size 补偿相互作用；**不得手工再乘/除 4 或 accumulation=8**。用独立 oracle 检查 SP1/SP4、长度不等、混合 clean/aux 下的 loss/梯度，数值验证通过后才放行。若实测上游归一化路径不符合本公式，修正输入侧权重契约并更新本文，禁止默默换成每条会议等权 CE。

当前 DP1 原生路径的运行时检查应为：`effective_N = raw_num_items_in_batch / accelerator.num_processes == planned_N`，因为 Swift 随后把归一化 loss 乘 num_processes；SP4 常见 raw 为 4N。这里仅核查有效分母，不在 loss 中再补一个倍率。要求 `average_tokens_across_devices=true`；若 Accelerate 的非数据并行修正已另行除去副本，导致上述关系不成立，明确拒绝此未验证组合，重新做数值验证，不能默默照用权重。

记录 clean token CE、aux token CE、加权总 CE、lambda、Nc、Na、正文/结构恢复标签数、实际错误事件数。加权训练 loss 不与旧普通 CE 曲线直接比较。Dev CE 始终使用原始模板/clean labels 的普通 CE。

## 7. 实施架构：轻量 Trainer 输入适配，原生优化与 CE

新增独立 package，不编辑 `ms_swift` 基线或 vendored Swift 源码。

1. `core.py`：纯 token 变换、坐标映射、候选过滤/采样、词表与 span 资格、预算权重、确定性 seed。
2. `prepare.py`：校验 train/dev 数据身份、tokenizer、官方 collator tokens；生成每个 update/源 occurrence/视图的有序 manifest、统计和哈希。原目标文本不改；元数据存 source key、anchor、K、kind、repeat count、权重和 raw length。
3. `plugin.py`：继承既有 MossTemplate，以新模板名注册；先调用原始 `_encode`，验证 EOS，随后按计划设置 aux labels、loss_scale 与元数据。结构/音频生成仍用官方 collator。**旧模板的 implicit padding 分支没有 pad loss_scale，子类必须同步补 0。**
4. `trainer.py`：继承 Swift Seq2SeqTrainer，仅适配顺序 dataloader 与 `training_step` 进入父类前的在线 probe/输入变换；不覆盖 CE/backward/optimizer/gradient clipping/save/load。注册只作用于新入口进程，日志如实写为实验 Trainer。
5. `train.py`：验证配置/计划/gate 后通过 SwiftSft 实例与 scoped TrainerFactory 映射启动，结束恢复映射。错误配置和未通过凭据 fail closed。
6. `evaluate.py`：固定 Dev 注入库、自由续写、完整生成诊断及分组比较；复用原 parser 和 score_v1，不从解析成功子集过滤失败案例。
7. `preflight.py` / `train.slurm` / `README.md`：生成可审阅命令，分层 GPU gate、原生恢复和证据归档入口；默认 dry-run，不自动 sbatch。

### 单个辅助 microbatch 的执行顺序

```
collator 取得完整干净 IDs/官方 features，aux labels 预先只保留 K
training_step 取出元数据、拷贝会修改的 tokens/labels/scales（音频 tensor 只读共享）
保存模块 train/eval 状态与 RNG
进入 no_grad + eval + 正常 BF16 context
调用父类 _prepare_inputs 和相同 template.forward_context，跑 clean probe
在当前 SP shard 找到 query=a-1（以及诊断 a）；只汇聚所需 1–2 行 logits
协调 rank 用稳定 RNG 采样，广播事件/token/原始概率统计，所有 rank 校验一致
释放 probe outputs；恢复所有模块状态/RNG（finally）
回到未 SP 切分的完整 IDs 上应用唯一事件，移动 labels/loss_scale，重新右补齐
重新生成 position_ids；核对真实 audio prefix 和 features 完全不变
调用父类 training_step(model, modified_inputs, ...原 num_items_in_batch...)
由原生 Swift 完成一次 forward、CE 和 backward
```

probe 的 labels/loss_scale/元数据从模型 kwargs 中去除，不能为了采样计算完整 CE。完整干净序列可用于 probe，因为因果 attention 不允许 query=a-1 看未来；训练结果与只给前缀 probe 的 logits parity 必须检查。禁止 gather `[L,V]` 全 logits；每个 rank 按真实 position_ids 定位 query，只 gather `[1,V]` 或 `[2,V]`。不用 SP 位置算术假设替代真实全局 position_ids；必要时为小向量调用框架 split 获取坐标。

v1 probe 可以保留各 rank 的本地全 logits，避免改模型 forward；无梯度临时张量用完立即删除。不共享带梯度 KV，不把主图同时留到恢复 forward。正常 clean backward 完成后才运行 aux probe/forward。探测时 FSDP 通信必须所有 rank 参与，即使只有一个 rank 拥有所需 query；不可只在 rank0 调模型。

精度/音频：复用 collator 已生成的 `input_features/audio_feature_lengths/audio_chunk_mapping`，训练和 probe tensor 字节相同；不能把编码器输出缓存后 detach，因为本实验仍全参数训练。每次辅助 forward 重新运行音频编码器。输入右侧补齐用于 SP，真实 token 前缀不变；insert 后 position_ids 连续增加，不能给重复项相同 position ID。

恢复分支完整保留原会议和目标输入长度（只加最多 2 token），只限制监督窗口；不裁音频或把窗口当作新的短音频。超出 131072 时该事件 fallback clean_aux 并记录；不能右裁原目标/EOS。真实多样本 padding/packing 不支持。

### 确定性 dataloader 与恢复

v1 DP1 顺序 sampler + 原生 DataLoaderShard，明确支持 skip_batches；SP ranks 同一顺序和同一 metadata，非 DP1 直接拒绝。不依赖当前上游硬编码 seed=42 的 sampler。每个 optimizer 边界保存 plan hash、配置 hash、next row index、source occurrence 和事件 audit；中途退出后仅从已保存 optimizer 边界恢复。原生 optimizer/scheduler/RNG 必须一起恢复。lambda 由恢复后的本实验 global_step 推导，不重置。

复现核验比较不间断 2 update 与 1 update 保存/恢复再 1 update：会议顺序、事件、LR、optimizer step、有效标签数完全相同；权重/梯度按相同后端的数值容差比较。禁止把旧 step150 的 global_step 当成本实验 u。

## 8. 数据隔离与固定评估库

训练只允许官方 train 536；Dev 26 用于标定与选点；已看过的 Test（包括 `alimeeting/test/R8005_M8009`）仅作 known-test diagnostic 和用户要求的既有错误率量级参考，不能进入候选库、选锚点或 checkpoint 选择。概率公式按 Dev 标定。此次依已知 Test 症状设计增强，因此最终在旧 Test 上的重复评测也不能包装为 fresh blind test。

真实“外”循环起点的转录进度尚未完成唯一的替换/插入/漂移归因；本设计不假定它属于其中一类。合成样本按自己明确的 edit 操作定义正确续写，不把故障生成中的绝对位置直接当作 gold 标签索引。已知故障的局部对齐与音频核查可作为附加诊断，但不决定本实验训练标签。

按 corpus+recording ID、规范路径、音频哈希检查 train/dev/test 隔离；不能仅比较不同相对路径字符串。训练 target hash 与 canonical train 必须一致。新增 clean/aux 行只能引用既有 train 身份。

在训练之前，用共同初始化 checkpoint150 构造固定 Dev 注入库；所有 arm 共享同一事件 token ID，评估时不按当前模型重新采样。每场会议选尽可能覆盖 25/50/75% 正文进度的 3 个不同合法锚点；每个锚点保存 clean、substitute、repeat2、repeat3 四种情况（最大 26×3×4=312）。sub 按同一 sparse 候选策略得到 gold/非法/无质量候选时，该 case 标 not_applicable，保留记录和分母；不得强制排除 gold 重抽。固定 repeat2/3 是条件恢复压力测试，在每个锚点实际注入，明确与训练的低频触发不同，其错误率不代表自然发生率。库身份必须独立于 arm/GA 配置。固定库禁止读 Test。

### 三层测量

1. **teacher-forced recovery**：前 1、4、8、16 个原始正确 token 的 CE/top1；只说明条件预测能力。
2. **自由恢复**：从同一个错误前缀真正 greedy 生成最多 256 token，保留 raw IDs。记录前 1/4/8/16 token 与正确续写 exact match、规范化正文 edit distance、删除数、附加重复长度、EOS、结构解析与时间推进。不能用 teacher-forced 准确率替代这一层。
3. **完整自由生成**：全部 26 Dev 完整会议；同一官方 vLLM BF16 greedy，temperature=0、top_p=1、repetition_penalty=1、presence/frequency=0，最大 context=131072，输出 cap=min(65536,context-prompt)。保存 token IDs、finish_reason 和 raw text；复用 CER/cpCER/DER025，并增加原始重复/尾部诊断。

局部 256-token cap 是测量上限：到 cap 记局部 horizon_censored，不等同整会转录截断；EOS 提前与已到真实末尾分开。局部 exact-match16 衡量起始轨迹；不能单独命名 sustained recovery。持续恢复的证据需同时报告 256-token 内是否再次循环、后续正确内容/编辑对齐进度、时间戳结构和删除率，不合成一个掩盖细节的分数。

完整输出 hard failures：触顶截断、空输出、明确的长循环、明显提前结束、重大解析丢失。长循环诊断预注册阈值：单 ID 连续≥32；最小周期 2–16 的连续重复区间≥64 tokens；相同结构片段/时间戳块持续重复也计入；按 token pattern 去重事件。阈值是异常筛查，需要将参考中的真实重复单列核验；正常 2–3 次真实重复不能直接判为失败。扫描 raw IDs/完整 raw text，包含 parser 丢弃的尾巴。

提前结束筛查：EOS + 最后有效语音时间比参考最后语音时间早超过 max(30s,参考最后时间的5%)，或无最后有效时间且参考非空。使用参考发声结束而非音频文件长度，允许合法 overlap/时间戳交错；标为待核对，在完成参考/输出核对前不能通过验收。还要报告删除率和完成度，防止生成一个很晚时间戳掩盖中间大量漏字。

另设自然重复保护集：从原 Dev 参考自动列出含真实重复的正文 span，比较 clean 输入下对应内容的 deletion/recall；不加新训练样本。

## 9. 运行顺序、选点与成败判定

工程阶段（不作效能结果）：

- G0：CPU 合同测试和真实 tokenizer/官方 collator 数据验证。
- G1：SP1 小会议，对照原始 native CE；probe full/prefix/cache parity；单次 clean+aux 更新。
- G2：相同小会议 SP4/FSDP1，逐位置/标签计数/加权 loss/gradient parity；跨 shard 边界的替换和插入；一个 shard 无有效标签。
- G3：最长完整会议 +2 插入，真实一更新与原生保存，峰值显存记录；4 source/8 microbatch 的预取内存也检查。目标≤42 GiB/48GiB A40且不 OOM；不通过就明确报告，不偷偷冻模块/裁音频。
- G4：2 update 连续 vs 1+恢复+1 的状态/事件一致性；导出后 HF/vLLM 模型加载、短/长正确前缀和错误前缀 32-token parity。BF16 后端近似误差用参考 logit margin 解释；不要求整条长生成逐位相同。
- G5：确认所有 arm 的 initial weights hash、train order/anchor/labels budget 一致；Dev greedy 校准产物及冻结概率已绑定；原始 dev CE 通道无 augmentation；准备全部 26 Dev 完整评估与固定库。

数值容差：CPU FP64 oracle rtol=1e-8/atol=1e-10；FP32 短序列 loss atol=1e-5、gradient cosine≥0.99999；BF16 SP loss abs diff≤0.02 且 gradient cosine≥0.99。near-tie next-token 定义 top1-top2 margin≤0.1；margin>0.1 却跨路径改变 top1 必须审计。容差不能代替正确计数和词表身份的严格相等。CUDA 非确定性影响需留误差向量，不自动放宽。

Pilot：同一初始 step0 评估一次；各组新增 update=5、15、30 保存/评估，dev clean CE 每5步。普通完整 Dev 出现 hard failure 时停该组并保留失败 checkpoint/预测；用其他组在共同已完成的 milestone 做比较，不把后续缺失看作零失败。

主对比 `sub-clean_aux`、`mix-sub`；`ce` 提供正常继续训练参考。完整 Dev cpCER/DER 各 corpus 相对共同起点及同 milestone ce 不能恶化>0.5 percentage point；自然重复保护集不得出现新增明显删词。附带报告原有 score，不只按一个合成分数挑选。

每组 checkpoint 先满足完整 Dev 功能/质量约束，再按固定注入库 free exact-match16 较高、局部重复率较低、最后较早 update 的顺序选点。teacher-forced loss 不参与替代功能验收。阈值调整只能记新 protocol version，不能看到结果后改本次规则。

支持结论：注入库短程/持续恢复改善且完整质量不退化，可以说短程恢复有效；若完整 Dev 对照本来 0 次长循环，不能宣称证明降低自然长循环率。pilot 只产生候选配方；多个 seeds、完整一个 epoch、会议级 paired bootstrap（2000 次，固定 seed）用于后续确认。重复锚点属于同一会议，不能当312个独立样本。

## 10. 日志与可复现交付

每事件 JSONL：schema_version, run/arm/seed, optimizer_step, source_slot/occurrence, source_key/split, source_target_hash, tokenizer_hash, anchor_abs/target, query_abs, event_kind, original/new IDs, insert_count, original-to-new recovery range, K, body_label_count, gold_prob, candidate_mass, candidate IDs/probs, sample seed, fallback_reason, clean/aux lengths, real/padded lengths, SP ranks event digest。

每 update：Nc/Na/N、lambda、clean/aux/total CE、有效事件数与 fallback 类型、训练/探测耗时、peak memory、LR、grad norm。日志仅 rank0 写汇总，每 rank 写一致性摘要。

完整交付至少包括：DESIGN.md、可解析的默认配置、实现代码、CPU 测试结果、脚本语法检查、dry-run manifest、待运行 GPU checks、启动/恢复/评估命令、文件哈希及真实限制说明。没有 GPU result.json 的 gate 必须保持 pending；不能用编译/启动成功代替实际训练通过。

首版实现范围：已提供稀疏输入变换、原生训练适配、Dev 校准、固定局部恢复库/生成、失败计数与有界工程训练入口。CPU tokenizer 审计不包含远程真实音频/collator；当前没有正式有序训练 manifest、Dev 校准结果或固定注入库。G2 数值对比 harness、G3 专用最长样本计划、G4 跨后端/恢复自动比较、G5 完整 Dev 自动评估仍未完整实现；`gpu_checks.py` 是执行外部检查并验证结果的外壳，不能将它本身视为已具备全部检查。README 已区分这些缺口与“已有代码但 GPU 尚未运行”的项目。

Sol 编码责任：遵守此设计；遇到框架接口的真实冲突先回报并给出具体修正，不将在线采样换成冻结 teacher，不改变标签/损失数学。默认只实现并跑本地测试，不提交 GPU 训练、不改其他活跃任务。完成后由主任务审查代码与结果。

## 11. 文献与结论边界

- [Scheduled Sampling (2015)](https://arxiv.org/abs/1506.03099)：模型生成历史参与训练的背景。
- [Scheduled Sampling for Transformers (2019)](https://aclanthology.org/P19-2049/)：两遍解码及 Transformer 历史依赖；本文三次顺序 forward（clean/probe/aux）是适配当前完整会议工程的选择。
- [Reducing Exposure Bias in Training RNN Transducers (2021)](https://arxiv.org/abs/2108.10803)：ASR 历史扰动的相关证据。
- [DITTO (2022)](https://arxiv.org/abs/2206.02369)：句子级重复自增强的研究，不证明 MOSS 的 2–3 token 是最佳长度；本实验没有照搬其重复惩罚目标。

这些文献支持实验方向，不提供本文参数或 MOSS 改善的验证结果。
