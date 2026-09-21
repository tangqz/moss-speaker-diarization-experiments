# MOSS 62994 训练前后评估协议 v1

比较原始 MOSS-Transcribe-Diarize 与 full-attention-ddp4-62994 的最终权重。两者共用原始 tokenizer、processor、官方模型代码与解析器。每场会议完整输入，不进行人为裁剪、切窗或测试集调参。

数据：AliMeeting dev 8 / test 20；AMI dev 18 / test 16；AISHELL-4 test 20，共 82 场。先验证集，再测试集；两种 split 分开统计。重新核对训练路径与会话 ID 不重叠，保存输入、标注和权重 SHA-256。AMI 输入使用现有 Mix-Headset；其他语料使用此前准备的 16 kHz 单声道音频。结果是本项目协议下的比较，不直接宣称复现论文分数。

解码：四张 A40 各运行一个独立进程，同一会议的两个模型分配到同一卡。bf16、SDPA、full attention、batch=1、greedy、num_beams=1、use_cache=true、logits_to_keep=1。最大输入与输出总上下文 131072，max_new_tokens=min(65536, 131072-prompt_len)。预算只依赖输入长度；EOS 前达到预算会明确标记 truncated，仍保留并评分预测。没有使用训练标签决定解码预算。每次模型加载后使用独立 JFK 短音频预热，预热不计入准确率和时延。

CER：按时间戳顺序合并片段文本，Unicode NFKC + casefold，去空白、标点、符号和控制字符，保留字母、数字及组合标记；计算字符 Levenshtein 距离。英语也按字符计算，不标为 WER。cpCER：每个说话人的文本依时间连接，在字符编辑距离矩阵上求最优一一说话人匹配，多余/缺失说话人以空串补齐。测试使用 MeetEval 的 cpWER 接口对逐字符空格分词结果进行交叉校验。DeltaCP=cpCER−CER，不截断负值。各数据集按参考字符总量汇总，不平均逐会议百分比。[MeetEval 算法与工具](https://github.com/fgnt/meeteval)

DER：pyannote.metrics 最优说话人匹配，显式保留重叠语音，分别报告 collar=0 与 collar=0.25 秒。以既有 RTTM 和 UEM 为准：AliMeeting 从 TextGrid 导出；AISHELL-4 使用官方 RTTM；AMI 使用 AMI-diarization-setup 的 only_words RTTM/UEM。按累计 reference speaker-time 汇总。无效零时长片段不进入 DER 时间区间，但原始预测完整保存。[pyannote DER 参数及实现](https://pyannote.github.io/pyannote-metrics/_modules/pyannote/metrics/diarization.html)

性能：逐会议保存完整预处理加生成时间、生成时间、首 token 时间、生成 token 数、RTF、tokens/s、进程峰值显存及开始时可用 GPU 显存。加载及预热独立记录，不计 RTF。RTF=累计处理秒数/累计音频秒数；并发总墙钟时间另行报告。若 GPU 有作业之外的显存占用，保留原始硬件记录并在报告注明。

失败处理：逐条原子保存原文、token IDs、解析片段、EOS/截断状态和错误；恢复时仅跳过已经成功保存的同一运行记录。缺失或推理失败不计作成功，也不填造指标；汇总列出覆盖率，并仅对两模型共同完成的会议给出配对比较。仅在 164 条预测均完成评分且所有 worker 正常结束后输出 EVALUATION_COMPLETE。

用户要求：评估结束后，把所有训练与测试日志、逐条预测、指标明细和运行配置下载到当前本地项目，并制作本次训练的可视化报告。训练权重和原始音频不属于日志下载范围。
