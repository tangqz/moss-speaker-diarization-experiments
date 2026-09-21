# MOSS 训练监看

打开 **http://127.0.0.1:6006** 查看 TensorBoard。当前正式训练的名称为 `full_attention_v2_63421`；`full_attention_v2_63415` 是内存超限后停止的首次尝试，`qualification_63410` 是启动前的验收对照。旧记录保留供核查。

图表每 60 秒同步一次：

- `train/loss`：四卡损失分子合计 / 全局有效监督 token 数。
- `train/learning_rate`、`train/grad_norm`：实际学习率与裁剪前梯度范数。`train/learning_rate_x1e6` 将学习率乘以一百万，便于在小图中读取。
- `memory`、`performance`：各卡峰值显存、计算步骤耗时。
- `host_memory`：CPU 内存、闲置 pinned 缓存释放前后大小；其中 `peak_rss` 是进程历史峰值，`after_cleanup_rss` 是清理后的当前值。
- `validation`、`validation_types`、`entropy`：验证完成后出现的整场 teacher loss、各 token 类型 loss 和完整词表熵。
- `dev`、`sentinel`、`test`：对应完整生成评估完成后出现的 CER、cpCER、DER 和失败计数。

没有完成的验证不会显示虚构的数据点。右侧可调整平滑；检查训练尖峰时建议平滑为 0、关闭忽略离群点。

曲线暂时不动时，打开上方 **Text** 页签的 `status/current`：这里显示真实作业状态、最后完成步数，区分验证阶段、内存超限和运行失败。SSH 同步连接正常并不代表训练进程仍在运行。

电脑重启后，双击“打开训练监看.cmd”。需要保持已登录的 SSH 连接才能同步新指标；连接断开时已有曲线仍可查看，服务器训练继续运行。助手的主动检查仍按此前约定每 45 分钟执行。

安装使用项目独立环境，版本 TensorBoard 2.21.0；没有修改服务器训练环境，也没有安装 TensorFlow。依赖版本保存在 `requirements-lock.txt`，连接状态见 `bridge_status.json`。

依据：[TensorBoard 官方入门](https://www.tensorflow.org/tensorboard/get_started)、[PyPI 2.21.0](https://pypi.org/project/tensorboard/2.21.0/)。
