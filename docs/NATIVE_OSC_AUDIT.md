# 原生 OSC 动作协议核查

核查日期：2026-11-04（北京时间）。本次为本地代码检查、锁定上游源码检查和
平台无关测试；未连接 AutoDL、启动仿真、加载 CUDA 模型或运行评测。

## 结论

当前动作桥不能用于宣称原生 LIBERO 基准性能。问题涉及 Kinex 的 IK 转换、
RoboEnv 后端的控制器和仿真推进方式，不能只改 `space` 标签或缩放系数。
原生 OSC 路径仍待实现。历史 600-action smoke 的 `success=false` 保持原样。

本次起始本地 HEAD 与 GitHub 实时 `main` 均为
`10a103adbd195692dc2922e5a0ee2622de482495`，修改文档前工作区干净。

## 本次定位到的执行链路

```text
SmolVLA predict_action_chunk + postprocessor
  -> actions[N][7]
run_vla_subtask
  -> 前 6 维乘 [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
  -> pose_delta，经 run_policy / joint_sequence 逐步 IK
  -> 7 维绝对关节目标 + 单独的夹爪命令
RoboEnv LIBERO binding
  -> world.step({arm: joints, gripper: ...}) 暂存目标
  -> JOINT_POSITION.set_goal(..., set_qpos=target)
world.wait / 后端 tick
  -> 临时抑制 controller.set_goal
  -> env.step 内保留已暂存的关节目标
```

本地入口：

- `.kinex/tools/run_vla_subtask/tool.py`：构造缩放、动作空间和调度频率。
- `.kinex/tools/run_policy/tool.py`：调用 `joint_sequence` 并分设备写入。
- `kinex_devices/smolvla/driver.py`：模型后处理后返回动作块。
- `scripts/run_kinex_smoke.mjs`、`scripts/run_direct_episode.mjs`：显式请求 30 Hz。

锁定上游源码证据（均为本次读取，不代表服务器当前代码已重新核验）：

| 来源 | 关键事实 |
|---|---|
| [Kinex policy.py](https://github.com/stacx-lab/kinex/blob/15207927c6669fe98f1213a3fd05148e0fe938d3/packages/core/assets/python/kinex/policy.py#L35) | `joint_sequence` 从一次状态快照开始，把 Cartesian 动作块转换为绝对关节序列。 |
| [RoboEnv backend.py](https://github.com/stacx-lab/RoboEnv/blob/dd2a0e3101313deb772db434ee5989648ab96d9f/runtime/src/roboenv_runtime/adapters/libero/backend.py#L83) | 初始化固定 `controller="JOINT_POSITION"`、`control_freq=20`。 |
| [RoboEnv step/tick](https://github.com/stacx-lab/RoboEnv/blob/dd2a0e3101313deb772db434ee5989648ab96d9f/runtime/src/roboenv_runtime/adapters/libero/backend.py#L163) | `step` 写入 `set_qpos`；`tick` 将 `set_goal` 临时替换为空操作。`_advance` 虽有 OSC 注释，本文件未提供公开 OSC 动作入口。 |
| [RoboEnv devices.py](https://github.com/stacx-lab/RoboEnv/blob/dd2a0e3101313deb772db434ee5989648ab96d9f/runtime/src/roboenv_runtime/bindings/backends/libero/devices.py#L99) | arm/gripper binding 要求当前机器人 `action_dim == 8`，不能直接换控制器而继续沿用该绑定。 |
| [RoboEnv adapter.py](https://github.com/stacx-lab/RoboEnv/blob/dd2a0e3101313deb772db434ee5989648ab96d9f/runtime/src/roboenv_runtime/adapters/libero/adapter.py#L44) | 评分详情固定报告 `native_osc_equivalence=False`。 |
| [LIBERO ControlEnv](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py#L12) | 默认控制器 `OSC_POSE`、20 Hz，`step(action)` 直接交给原生环境。 |
| [LeRobot LIBERO wrapper](https://github.com/huggingface/lerobot/blob/772bc21ef1ab17b72e17ebb1593ff78725a21d55/src/lerobot/envs/libero.py#L339) | relative 模式设置 `use_delta=True`，7D 动作直接传入环境；默认先执行 10 次 `[0,0,0,0,0,0,-1]` 稳定动作。 |

## 原生路径必须保持的契约

| 项目 | 当前路径 | 原生接入要求 |
|---|---|---|
| 动作 | 6D 增量转 IK，再与夹爪分别写入 | 后处理后的同一条 7D 动作整体交给原生 OSC；不再经过外部物理量缩放或 IK。 |
| 控制器 | JOINT_POSITION/set_qpos | OSC_POSE，relative 模式，保留原生目标更新与动力学控制。 |
| 时间 | 工具请求 30 Hz，后端 20 Hz | 以原生 20 Hz 为准；一条策略动作对应一个控制帧，记录实际执行帧数。 |
| 初始化 | 5 个关节保持帧 | 原生基线与桥接路径使用相同 seed、官方 init-state index、稳定动作及帧数；记录状态与资产哈希。 |
| 观测 | FK+mount、轴角、由 aperture 重建的两指位置 | 与原生 `robot0_eef_pos/quat`、`robot0_gripper_qpos` 逐项对比；8D shape 相同尚不能证明数值语义一致。 |
| 相机 | RoboEnv render 垂直翻转，模型 driver 再翻转 H/W | 逐像素或带方向标志的参考画面对比；确认相机对应、分辨率及完整预处理，不凭张量 shape 判断。 |
| 评分 | 仿真器独立判定，保留不等价标记 | 继续使用官方任务与成功判定；不能通过修改评分元数据来宣称等价。 |

上述观测与图像差异是待验证项，本次没有证明它们一定错误。
仅把工具频率改为 20 Hz，也不能修复控制器不等价。

## 无 GPU 可继续的工作

1. 以锁定 LeRobot 的原生 `OffScreenRenderEnv`/relative 路径作为参考，设计并实现
   一个整体提交 7D 动作的执行接口；协议验证拒绝维度错误、非有限值和不兼容控制器。
2. 在 RoboEnv 适配层明确 OSC 控制模式、设备绑定、调度和终止职责。实现须在仿真
   所有者线程执行；每条动作只推进一次控制帧，不能提交后又额外 `world.wait` 重复推进。
   暂不直接修改锁定上游或把现有 joint binding 冒充 OSC binding。
3. 编写不依赖 MuJoCo/CUDA 的协议和调用边界测试：完整 7D 同步提交、20 Hz、
   动作数量、终止后不再推进、无二次缩放/IK。Mock 通过只证明接口行为。
4. 准备相同动作序列、相同初始状态的原生/桥接轨迹对比程序和日志字段；至少记录
   代码版本、模型 revision、controller 配置、原始动作、实际控制帧与仿真时间。
5. 整理高层 LLM/VLM provider 的图像输入、物体清单和视觉进度输出契约、固定提示词、
   模型版本及 token/调用日志。真实 provider 凭据和调用能力尚未核验；不把占位配置算完成。
6. 补齐三种方法的公共 episode 总控制步数与调用预算；当前 `max_actions` 只是单次
   子任务上限，矩阵展开本身不证明预算公平或批量 runner 已验收。

目前可在 Windows 开发协议、测试和文档；不安装 Kinex、MuJoCo 或 LIBERO 到 Windows。
后端真实动力学与轨迹等价仍需 Linux 仿真实测，不能用静态检查替代。

## 按当前 EGL/CUDA 配置等待 GPU 的验收

用户手动开启 GPU 后再核验实际 SSH 入口和服务器代码版本，不重复安装已有环境。

1. `nvidia-smi`、运行模型所用 Python 的 CUDA 检查、LIBERO EGL 双相机渲染。
2. 锁定模型在 CUDA 加载，保存真实输入/输出与显存信息。
3. 在相同官方初始状态下，对原生参考与新桥接分别回放零动作、单轴小动作、
   夹爪开合及固定短动作序列；比较 qpos/qvel、末端状态、仿真时间、动作帧和图像。
   根据实际数值噪声记录容差与最大误差，不能先填通过。
4. 原生 direct-VLA 单集，再到 Kinex 新桥接单动作及 `libero-10-01` 完整单集。
   执行完成与 `success=true` 分别记录；失败轨迹保留。
5. 高层 provider 的真实视觉清单与闭环状态更新验收通过后，才考虑 3 场景、3 方法、
   3 seeds、每 seed 5 episodes 的正式矩阵。全部方法共享冻结的初始化与总预算。

OSC 本身并不在数学上依赖 GPU；这里等待 GPU 指项目当前的 EGL 仿真与 CUDA
策略验收流程，不能据此称所有无 GPU 准备已完成。

## 本次实测与历史记录

本次 Windows 本地使用已有 Python 3.13，未安装新依赖：

```powershell
$env:PYTHONPATH = "src"
py -3.13 -m unittest discover -s tests -v
py -3.13 -m tableclear.cli matrix --config configs/v1_smoke.toml
py -3.13 -m tableclear.cli matrix --config configs/v1_eval.toml
```

结果：18 tests passed；smoke 展开 1 job；正式配置展开 135 jobs，初始状态索引 0–14。
这些检查未执行任何真实机器人动作，也未证明 OSC 等价。

RoboEnv 的 1558 passed / 130 skipped、设备导出、模型静态验收和旧主机
600-action `success=false` 均仍属于历史记录，本次未重跑。
