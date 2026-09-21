# DKUCC 环境与运行记录（MOSS-Transcribe-Diarize 说话人日志项目）

- 建立日期：2026-09-11（美国东部时间 EDT，会话时间 01:17–02:00）
- 账号：`qt28`（account=`faculty`，QOS=`normal`）
- 仓库快照：`OpenMOSS/MOSS-Transcribe-Diarize` @ `61bc29cd4120be7b5d3b761b64cd5dff57263642`
- 对应 Phase 0 交付物：环境冻结（本文）、版本锁文件、冒烟测试结果

## 1. 集群与 GPU（实测）

| 项目 | 实测值 |
|---|---|
| 登录节点 | `dkucc-login-01.rc.duke.edu` |
| `common-gpu` 分区 | 2 节点 × 8× **NVIDIA A40**（48GB 卡，可用 46068 MiB） |
| 本次测试节点 | `dkucc-core-gpu-dkurc-d-it09-8` |
| GPU 驱动 | 590.48.01，compute capability 8.6 |
| 其它分区 | `l20-gpu`（4×8×L20）、`h20-gpu`（8×H20）、`teaching-gpu`（Titan）、专用节点（4090/L20） |

## 2. 账号限额（sacctmgr 实测，对 common-gpu 生效）

| 限额 | 值 | 影响 |
|---|---|---|
| MaxTRES（每作业） | `cpu=1000, mem=500000M, gres/gpu=4` | **单作业最多 4 块 GPU** |
| MaxJobs | 1 | **同时只能运行 1 个作业** |
| GrpTRES | `cpu=400, mem=1500000M` | 账号级总量 |

**⚠️ 不要使用 `--exclusive`**：它会请求整节点 8 块 GPU，超出 `gres/gpu=4` 上限，作业将一直排队（`squeue` 中显示 `AssocMaxGRESPerJob`，而 `srun` 只提示 "queued and waiting for resources"）。
正确做法：`srun/sbatch -p common-gpu --gres=gpu:N`（N ≤ 4）。

## 3. 软件环境（已冻结）

| 组件 | 版本 / 路径 |
|---|---|
| 系统 Python | 3.9.25（过老，不使用） |
| Python 环境 | conda env `/dkucc/home/qt28/envs/moss312`，**Python 3.12.14** |
| conda | `anaconda/2023.7` 模块（conda 23.7.2） |
| uv | 0.12.13（`/dkucc/home/qt28/.local/bin/uv`）——**后续安装一律用 uv** |
| torch / torchaudio | 2.14.0+cu130 / 2.11.0 |
| transformers | 5.17.0 |
| accelerate | 1.15.0 |
| MOSS 代码 | `/work/qt28/moss/MOSS-Transcribe-Diarize` @ `61bc29c` |
| 模型权重 | `/work/qt28/moss/models/MOSS-Transcribe-Diarize`（safetensors = 1,817,113,576 B，与 HF 官方一致） |
| 环境脚本 | `/work/qt28/moss/dkucc/env.sh`（`source` 后获得全部路径/代理/环境） |
| 版本锁文件 | `/work/qt28/moss/dkucc/requirements-frozen-2026-09-11.txt`（88 行） |
| 脚本备份 | `/dkucc/home/qt28/moss-dkucc-backup/` |

## 4. 网络矩阵（计算节点实测，2026-09-11）

| 目标 | 直连 | 备注 |
|---|---|---|
| `pypi.org` | ✅ 可用（下载带宽约 2 MB/s 级） | pip 下载慢主要是带宽限制，不是解析问题 |
| `github.com` | ❌ 超时 | 代理 `proxy-dku.oit.duke.edu:3128` 快（~3.6s）；`proxy-china-prod-dku` 慢（~25s） |
| `huggingface.co` | ❌ 超时 | `proxy-dku` 代理可用（1.5s） |
| `hf-mirror.com` | ✅ **~21 MB/s** | 模型权重下载首选（1.7GB ≈ 1.5 分钟） |

`env.sh` 中内置 `DKU_PROXY_CHINA` / `DKU_PROXY_RESTRICTED` 变量与 `dku_proxy` 函数。

## 5. 冒烟测试结果（Phase 0 验收 ✓）

```bash
source /work/qt28/moss/dkucc/env.sh
MOSS_MODEL_ID=/work/qt28/moss/models/MOSS-Transcribe-Diarize \
  python /work/qt28/moss/dkucc/smoke_infer.py <音频.wav>
```

| 音频 | 时长 | 加载 | 生成 | 峰值显存 | 结果 |
|---|---|---|---|---|---|
| `jfk.wav`（英） | 11.0 s | 6.3 s | 9.0 s | 1.74 GiB | 3 段全部正确解析 |
| `asr_example_zh.wav`（中） | 5.55 s | 2.0 s（缓存热） | 3.6 s | 1.74 GiB | 输出正确（UTF-8 字节级校验通过） |

注：在 Windows 终端经 SSH 查看中文可能显示乱码，属于显示链路编码问题；模型输出本身为正确 UTF-8（已做字节级验证）。

相关日志（服务器）：`/work/qt28/moss/logs/smoke-zh-20260911.log`、`environment-*.txt`。

## 6. 常用命令

```bash
# 交互式 GPU 作业（不要 --exclusive）
srun -p common-gpu --gres=gpu:1 --cpus-per-task=16 --time=04:00:00 --pty bash -i

# 提交作业
sbatch /work/qt28/moss/dkucc/smoke_job.slurm
sbatch /work/qt28/moss/dkucc/full_attention_job.slurm

# 激活环境
source /work/qt28/moss/dkucc/env.sh

# 安装新包（uv，速度快）
uv pip install --python /dkucc/home/qt28/envs/moss312/bin/python <package>
```

## 7. 待办 / 阻塞项

- [ ] AliMeeting / AISHELL-4 / AMI 的服务器路径、版本与官方划分（by 世祥）——**阻塞 Phase 1 数据转换**
- [ ] Unlimited OCR 论文/代码与已修改的 attention 参考实现（by 金老师）——不阻塞 baseline
- [ ] CER/DER/cpCER/DeltaCP 指标定义与准确率容差（需老师拍板）
- [ ] MS-Swift 接入验证（Phase 2；当前环境已具备基础条件）
- [ ] 多卡（2–4 GPU）NCCL 拓扑验证（正式训练前）

## 8. 路径速查

```text
/work/qt28/moss/
├── MOSS-Transcribe-Diarize/   # 官方仓库 (61bc29c)
├── models/MOSS-Transcribe-Diarize/  # 模型权重 (1.7GB)
├── data/samples/              # jfk.wav, asr_example_zh.wav
├── dkucc/                     # env.sh, env_probe.sh, smoke_infer.py, *.slurm, 版本锁
├── logs/                      # 环境探针、pip、模型下载、推理日志
├── checkpoints/               # (训练输出)
└── results/                   # (评测输出)

/dkucc/home/qt28/
├── envs/moss312/              # conda 环境 (Python 3.12)
├── .local/bin/uv              # uv 0.12.13
└── moss-dkucc-backup/         # dkucc 脚本备份
```

- [x] 官方 finetune.py 最小 smoke SFT：2026-09-11 通过（2 步，train_loss=0.836；详见 smoke-sft-result-2026-09-11.md）

- [x] 多卡 NCCL 验证（2 卡）：2026-09-11 通过（torchrun --nproc_per_node=2，all-reduce/broadcast ok，作业 62952，A40×2）

## 9. 数据集到位记录（2026-09-11）

三套数据已全部下载并字节级核验（与官方 Content-Length 完全一致）：

| 数据集 | 目录 | 内容 | 校验 |
| --- | --- | --- | --- |
| AliMeeting | data/external/alimeeting/ | Train_Ali_far 73.24GiB、Eval_Ali 3.42GiB、Test_Ali 8.90GiB | 与阿里云 OSS 官方字节一致 |
| AISHELL-4 | data/external/aishell4/ | train_L 7.06GB、train_M 25.50GB、train_S 14.13GB、test 5.24GB | 与 OpenSLR SLR111 官方字节一致 |
| AMI | data/external/ami/ | 154 会议 Mix-Headset wav（9.42GiB）+ 标注（manual 1.6.2 / auto 1.5.1 / DOME / SocialRole） | DONE 159/159，FAIL=0；官方 test-16 会议全部在列 |

- 下载来源与实测速度：AliMeeting=阿里云上海 OSS（15-40MB/s）；AISHELL-4=www.openslr.org（0.5-2MB/s，约 4.4h）；AMI=爱丁堡 AMICorpusMirror（4 线程约 4MB/s）。
- 本地备份副本（服务器已核验完整，可自行决定是否保留）：D:\dkucc-downloadsishell4（与服务器字节一致）。
- 完整性扫描：gzip -t 全量校验运行于后台，日志 logs/targz-integrity-20260911.log。

## 10. Phase 1 数据转换完成（2026-09-11）

三数据集 → MOSS conversation JSONL 全部完成并通过校验（0 错误）：

| 数据集 | train | dev/eval | test |
| --- | --- | --- | --- |
| AliMeeting | 209 会议 / 111.4h / 186,364 段 | 8 / 4.2h | 20 / 10.8h |
| AISHELL-4 | 191 / 107.5h / 102,701 段 | — | 20 / 12.7h |
| AMI | 136 / 80.7h / 66,337 段 | 18 / 9.7h | 16 / 9.1h |

- 训练混合集 train_mix.jsonl（536 条，seed=0 打散）；验证 val_mix.jsonl（26 条）
- 测试集分开：test_alimeeting.jsonl / test_aishell4.jsonl / test_ami.jsonl（按执行文档要求）
- 产出：data/moss_jsonl（1194 条）、data/audio16k（36GB，16kHz 单声道）、data/manifests、data/refs（RTTM/UEM 评测参照）、data/reports（data_stats.json + phase1_validation.md，全部 0 错误）
- 转换约定：说话人按出场序映射 S01..；重叠语音保留；AISHELL 内联标签清理；AMI 同说话人词间隔 ≤0.5s 合并；指令=模型官方默认 prompt
- 转换器：data_prep/*.py（7 个脚本；副本于 dkucc/data_prep/，备份于 home moss-dkucc-backup/data_prep/）
