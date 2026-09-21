# DKUCC 训练 MOSS 模型：快速上手

更新时间：2026-09-11

本文把 DKUCC 官方页面和当前 MOSS-Transcribe-Diarize 微调说明整理成一条可以直接执行的流程。按目前掌握的信息，暂不纳入 GPU/存储计费，直接按教学/学习用途使用；正式使用前仍以管理员在登录后显示的资源状态为准。

## 0. 实测修订（2026-09-11，qt28 账号）

以下内容经实际集群验证，与后文旧描述冲突时**以本节为准**：

- **不要使用 `--exclusive`**：qt28@faculty 账号限额为每作业最多 `gres/gpu=4`，`--exclusive` 会请求整节点 8 块 GPU，作业将永久排队（原因 `AssocMaxGRESPerJob`）。
  正确用法：`srun/sbatch -p common-gpu --gres=gpu:N`（N≤4）。
- **MaxJobs=1**：qt28 同时只能运行 1 个作业（交互式会话也占名额；跑批作业前先退出交互式作业）。
- **Python 环境已就绪**：`/dkucc/home/qt28/envs/moss312`（Python 3.12.14 + torch 2.14.0 + transformers 5.17.0），
  通过 `source /work/qt28/moss/dkucc/env.sh` 激活；安装新包一律用 uv（`~/.local/bin/uv`），不要再走 pip/conda 解析。
- **网络**：PyPI 直连可用但带宽有限（~2 MB/s）；GitHub 需 `proxy-dku.oit.duke.edu:3128`；
  Hugging Face 模型下载走 `hf-mirror.com` 直连最快（实测 ~21 MB/s，1.7GB 约 1.5 分钟）。
- **模型权重**已下载到 `/work/qt28/moss/models/MOSS-Transcribe-Diarize`（safetensors 与官方一致）。
- 环境探针、冒烟测试、Slurm 脚本、依赖锁定文件均已就位，详见同目录 `environment.md`。

## 1. DKUCC 关键信息

| 项目 | 当前可用信息 |
|---|---|
| SSH 登录节点 | `dkucc-login-01.rc.duke.edu` |
| OnDemand | `https://dkucc-ondemand-01.rc.duke.edu` |
| 调度器 | Slurm |
| GPU 分区 | `common-gpu` |
| 交互式 GPU | `srun -p common-gpu --gres=gpu:1 --pty bash -i` |
| 个人配置/脚本 | `/dkucc/home/<netid>`，约 1 TB |
| 大数据/中间结果 | `/work/<netid>`，先自行创建目录 |
| `/work` 注意事项 | 文件超过 75 天会自动清理；无备份 |
| 登录节点用途 | 文件传输、编辑、提交/监控作业；不要在上面训练 |

DKUCC 主页列出的 GPU 资源包括教学用 4090/L20，以及学习和研究用 A40/L20。具体可见 GPU、数量和排队情况不要凭主页配置推断；进入作业内后必须用 `nvidia-smi -L` 和 `nvidia-smi` 记录实际值。

校外连接需要 DKU VPN。模型或 Python 依赖下载遇到网络限制时，按管理员文档使用 DKU 提供的代理，并遵守带宽限制。

## 2. 推荐目录布局

在登录节点执行：

```bash
ssh <netid>@dkucc-login-01.rc.duke.edu

mkdir -p /work/<netid>/moss/{data,models,checkpoints,logs,results}
mkdir -p /dkucc/home/<netid>/envs
cd /work/<netid>/moss
```

建议把代码、环境配置和小型 manifest 备份到 `/dkucc/home/<netid>`，把音频、模型缓存、checkpoint 和中间结果放到 `/work/<netid>`。两处都没有备份，重要 checkpoint 要另行下载保存。

## 3. 第一次登录：只做资源确认

不要在登录节点运行训练。先查看分区：

```bash
sinfo
```

申请一张 GPU 的交互式作业：

```bash
srun -p common-gpu --gres=gpu:1 --pty bash -i
```

进入 GPU 作业后，把本目录复制到服务器，然后运行环境探针：

```bash
bash dkucc/env_probe.sh | tee /work/<netid>/moss/logs/environment-$(hostname)-$(date +%Y%m%d-%H%M%S).txt
```

探针会记录主机、Slurm 作业、GPU、GPU 拓扑、CUDA、Python、PyTorch 和 Transformers 信息。它不下载数据，也不启动训练。

## 4. 准备 MOSS 环境

先在 GPU 作业内检查可用模块：

```bash
module avail
```

按 `module avail` 的实际结果加载 Python/CUDA 模块；模块名称不要猜。随后在代码目录安装环境。MOSS 官方当前测试组合为 Python 3.12 和 Transformers 5.x，微调还需要 `accelerate`：

```bash
cd /work/<netid>/moss
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[torch-runtime]" --torch-backend=auto
uv pip install accelerate
```

若服务器没有 `uv`，优先使用管理员提供的 Python/Conda 模块；不要在登录节点编译或安装大型 CUDA 依赖。`flash-attn` 先不作为 smoke 的前置依赖，等 CUDA、PyTorch 和显卡型号确认后再单独评估。

若依赖或模型下载无法直连，可在作业内临时设置 DKU 文档中的代理：

```bash
export http_proxy=http://proxy-china-prod-dku.oit.duke.edu:3128
export https_proxy=http://proxy-china-prod-dku.oit.duke.edu:3128
```

只有访问被该代理专门限制的地址时，才改用 `proxy-dku.oit.duke.edu:3128`；下载速度遵守管理员给出的上限。

## 5. 数据格式

MOSS 微调数据是 JSONL，每行一条 conversation，顺序固定为：文本指令、音频路径、assistant 目标文本。例如：

```json
{"conversation":[{"role":"user","message_type":"text","content":"Transcribe the audio with timestamps and speaker labels."},{"role":"user","message_type":"audio","content":"/work/<netid>/moss/data/example.wav"},{"role":"assistant","message_type":"text","content":"[0.00][S01]Welcome[0.72]"}]}
```

在接入 AliMeeting、AISHELL-4、AMI 前，需要先确认官方 train/dev/test 划分、音频路径、采样率、时间戳单位、说话人编号和重叠语音处理方式。不要把测试集混进训练 manifest。

## 6. 训练顺序

1. 三个数据集各抽少量样本，完成原始模型零样本推理。
2. 用官方 `finetune.py` 跑单卡 full-attention smoke SFT。
3. 用同一套数据和评测脚本跑 full-attention SFT baseline。
4. 再实现并测试固定窗口 RSWA，先 `k=128`、`k=256`。
5. 固定窗口结果稳定后，再做动态 `k`/可学习 mask，最后才考虑 RL。

每个实验至少记录：数据集和 split、代码/model revision、attention 模式、`k`、`max_length`、GPU 型号/数量、batch、解码参数、CER、DER、cpCER、DeltaCP、端到端延迟、RTF、audio seconds/second、吞吐量、峰值显存和失败信息。结果按数据集和音频时长区间拆分。

## 7. 先跑 smoke 作业

把 `smoke_job.slurm.example` 复制到服务器，替换其中所有 `<netid>`，确认 `smoke_train.jsonl` 和音频存在，然后在登录节点提交：

```bash
cd /work/<netid>/moss
mkdir -p logs
sbatch dkucc/smoke_job.slurm
squeue -u "$USER"
```

查看结果：

```bash
tail -f /work/<netid>/moss/logs/moss-smoke-<jobid>.out
cat /work/<netid>/moss/logs/moss-smoke-<jobid>.err
```

smoke 默认只跑 2 个 update step、单卡、`bf16`、gradient checkpointing 和较小的 `max_length`。它的目标是确认模型、音频、JSONL、CUDA 和 Trainer 全部连通，不代表正式超参数已经确定。

## 8. 正式 baseline

当前正式入口见 [memory_sft/README.md](./memory_sft/README.md) 与 [memory_sft/baseline.slurm](./memory_sft/baseline.slurm)。它接续 Proma 的长序列显存排查，采用 4 卡 DDP、SP=1、bf16 和 gradient checkpointing，依次执行数值/梯度、真实模型、全量数据、四卡训练和断点恢复验证，通过后启动完整训练集的 full-attention SFT。旧 `full_attention_job.slurm` 与 `.example` 保留为早期单卡模板，不是当前完整会议训练的入口。

`sequence_parallel_size=1` 表示关闭序列并行。按 [A40 兼容性评估](./A40_动态RSWA与熵门控RL_兼容性评估_2026-09-11.md)，确认有四卡后仍默认使用数据并行；序列并行须在长序列资源需求成立且自定义 attention/loss 全部验证后单独适配。

## 9. 使用时的安全边界

- 训练和评测必须通过 Slurm 作业运行，登录节点只做轻量管理。
- 不要把敏感数据放在 DKUCC；官方说明集群数据不备份。
- `/work` 超过 75 天的旧文件会自动清理；checkpoint 完成后及时下载或复制到受控存储。
- 计费信息按当前项目约定暂不作为启动条件，但资源仍受队列、配额和管理员策略限制。

## 参考

- DKUCC：<https://dkucc.dukekunshan.edu.cn/>
- MOSS 官方仓库：<https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>
- MOSS 官方微调说明：<https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md>
