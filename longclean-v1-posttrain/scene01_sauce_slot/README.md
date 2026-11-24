# 场景 01 第二罐的放置分叉

本实验复用 `875104 / 72` 的诊断续段。两臂从同一原始快照重放到第 569 步，在第一条持续开爪动作发出前分叉，观测到第 729 步。原任务上限为 520 步。放置规则读取双相机 RGB-D、机器人自身深度和夹爪状态。物体真值和官方目标只用于验收。

基线 A 重放存档动作。B 从先前的两帧及当前帧估计篮口和已入篮汤罐，按两罐同层落底的占地寻找空位。通过位置、时序和掌指扫过区域检查后才对准和下降。弃判时逐步执行基线动作。网格分辨率、净空约束和开发例的 Y 判定带假设见 [冻结协议](protocol.json)。

## 实验复跑

在实验室个人盘的既有 `review` 目录下建立本次独立运行目录，放入本目录的源码与协议。启动脚本复用已有 LIBERO 环境、校准、布局快照和名为 `scene01-sauce-budget-diagnostic-20261124-terminal` 的归档。原分叉、两套接管实现及校准文件的身份可以通过证据包的 SHA-256 核对。A 启动时检查原卡两次，占用时走 OSMesa CPU。B 复用 A 的后端。实验用不到模型推理。

```bash
sha256sum protocol.json slot.py run_pair.py launch.sh > frozen.sha256
tmux new-session -d -s slot569-a 'sha256sum -c frozen.sha256 && bash launch.sh A > A.log 2>&1'
# A.exit 为 0 后运行 B
tmux new-session -d -s slot569-b 'sha256sum -c frozen.sha256 && bash launch.sh B > B.log 2>&1'
```

## 本地验收

将以下三个证据包及各自同名的 `.sha256` 文件放在本地。核验使用 NumPy、此仓库的 `scripts` 和 `src`，并从配对包中恢复五个锁定的校准文件。验收会重算三帧视觉选点、所发动作、原生物理状态、连续双罐入篮及净空上界。首罐放置前的离线几何计算另外使用第 395–397 步的原始帧。可用 Matplotlib 生成两阶段的占位图。

```bash
python verify.py slot-pair-evidence.tar.gz budget-evidence.tar.gz vision.json audit.json
python early_slots.py slot-pair-evidence.tar.gz return-evidence.tar.gz early.json
python plot_slots.py audit.json early.json slots.png
```

已知开发例中，第 569 步规则因篮心估计跨帧波动而弃判。即使单看该帧也有 0/1,051 个位置同时满足篮壁与罐间净空。两臂 160 条后续动作完全一致。首罐放置前的三帧能找到双落点，已有直线下降检查对这两个位置均未准入。[选点与控制](slot.py) · [独立验收](verify.py) · [首罐双落点计算](early_slots.py)
