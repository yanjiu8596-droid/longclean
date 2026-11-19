# TableClear V1 场景 08 的 π₀.₅ LoRA 后训练

V1 的场景 08 回放提供了自然完成和改写开爪后完成的轨迹。本实验检验在同一 π₀.₅ 检查点上，用开爪修正后的动作窗口做监督微调，能否提高未见初态的官方任务成功率。训练和最终评测使用双相机 RGB、8D 机器人状态、任务指令和 7D 原生相对 OSC。

## 数据与训练

| 阶段 | 验收 |
| --- | --- |
| 原始轨迹 | V1 `42a8fa2` 的 120 条成功观测组轨迹。归档和规则分别通过 SHA-256 核验 |
| 数据重放 | 每个控制步收集双相机 RGB、状态及生效命令。状态以原版 LeRobot 的 PyTorch 浮点运算编码。初态及原始动作块边界核对仿真状态、时间、观测哈希和终局 |
| 划分 | 训练 96 条，其中改写 26 条。验证 24 条，其中改写 7 条。验证使用官方初态 24、37、40，按初态哈希隔离 |
| 保存 | LeRobotDataset v3 两个 split，20 Hz 无损图像。复读每集首末帧，核对动作、状态、图像和任务 |
| 训练 | 两臂各 600 次 AdamW 更新，学习率 1e-4。LeRobot 默认 π₀.₅ LoRA 模块，rank 8，alpha 16。冻结底座和原归一化 |
| 抽样 | 每次更新累积四个样本，两个共用的自然轨迹窗口，两个分别对准自然开爪或首次规则改写。先均匀抽布局，再抽轨迹。目标动作落在前 10 步，50 步的损失仅计有效动作 |
| 验证 | 统一验证窗口及噪声，按布局平均。以 200、400、600 步中最低的验证损失选适配器，相同分数取较早检查点 |

`train.py --arm smoke` 核对原始 flow matching 损失、有掩码损失、非零梯度、基础权重哈希、LoRA 保存和确定性续训。适配器还要通过原生 OSC 完整单集。

## 独立评测

四臂共用初态、推理种子、520 步预算、20 Hz OSC、50 步预测和 10 步执行。四臂分别为基础策略、基础策略加冻结 V1 开爪规则、自然开爪 LoRA、改写开爪 LoRA。两个 LoRA 臂直接输出动作。

- 普通集从固定的 BDDL 随机重置种子抽取 12 个新布局，每个使用两个推理种子。挑战集在 48 个预分配新种子中收集最多 8 个布局，入选条件为冻结 V1 规则实际改过至少一个命令。入选只看改写事件。每个挑战布局使用一个推理种子。
- 独立初态按原始状态和稳定后状态哈希去重，并与历史训练、验证及 V1 结题新状态比对。所有四臂在同一 GPU 上用冻结清单执行。完整批次最多 128 集。
- 主指标为官方 LIBERO 成功和按布局计的配对胜负。另报首个夹持和篮内释放的步数、释放是否保持到终局、非目标接触、总步数和耗时。场景 08 的收益门槛为改写 LoRA 相对基础策略至少在两个布局上获得净救援，总净胜为正，且普通集零回退。归因同时看自然 LoRA 对照。

## 运行与证据

在个人数据盘保留 V1 的项目、仿真环境和模型。模型侧 LeRobot 锁定 `772bc21ef1ab17b72e17ebb1593ff78725a21d55`，复用 V1 模型、处理器和分词器。用现有模型 Python 运行 `uv pip install --python "$MODEL_PYTHON" --target "$PERSONAL_RUN_DIR/python-extra" --no-deps -r requirements.txt`。启动时用 CPU 核验训练依赖、处理器哈希及一个小样本的 LeRobot v3 无损回读。

在 `tmux` 中运行以下命令，其中 `PERSONAL_V1_ROOT` 和 `PERSONAL_RUN_DIR` 由本机部署确定。

```bash
export TABLECLEAR_LAB="$PERSONAL_V1_ROOT"
bash run_lab.sh "$PERSONAL_RUN_DIR" score
```

最后一个参数可用 `replay-smoke`、`replay`、`dataset`、`train-smoke`、`freeze` 或 `score`。分阶段记录退出码，可从已校验的输出恢复。需要限制设备时可通过 `TABLECLEAR_ALLOWED_GPU_INDICES` 提供获准卡号。每个 GPU 阶段都检查健康、占用和 GPU 身份，空卡等待时不启动 CUDA 或 EGL。

保留 `replay-manifest.json`、逐集边界与物理审计、`dataset/build-manifest.json`、梯度与续训短测、选中适配器与完整 loss 日志、`test-plan.json`、逐集配对结果、选定视频、`summary.json` 和阶段日志。数据与模型制品留在个人数据盘和本地留痕，Git 分支收录可复现代码和协议。

面试可以从 V1 的汤罐释放失误展开，说明怎样把干预变成可重放的监督数据，怎样控制训练与验证泄漏，以及怎样用自然动作 LoRA 识别改写标签的贡献。训练规模、效果与回退以实际通过核验的证据为准。

模型与数据接口参考 [OpenPI 官方仓库](https://github.com/Physical-Intelligence/openpi) 和 [LeRobotDataset v3 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。
