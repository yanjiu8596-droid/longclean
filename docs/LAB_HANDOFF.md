# TableClear V1 实验室交接

## 从另一台电脑继续

在实验室电脑登录自己的 GitHub 和 Codex 账号，克隆本仓库并在 Codex 打开 `longclean`。如果已有副本，先检查工作区，再执行 `git pull --ff-only`。Linux GPU 服务器使用自己的仓库副本，按同一提交运行。两台机器用 `git rev-parse HEAD` 核对实验版本。

实验室 Windows 电脑用自己的系统用户登录。已安装 GitHub CLI 时，可以在 PowerShell 完成一次浏览器授权，再从历史提交 `bf4a9fa` 复用本机的提交名和邮箱。提交身份只写入这份仓库的本地配置。

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git --hostname github.com
gh auth status --hostname github.com
git clone https://github.com/yanjiu8596-droid/longclean.git
Set-Location longclean
git config --local user.name (git show -s --format=%an bf4a9fa)
git config --local user.email (git show -s --format=%ae bf4a9fa)
git ls-remote --exit-code origin HEAD
```

Git for Windows 自带 Git Credential Manager 时，也可直接用 HTTPS 克隆，并在 GitHub 要求身份验证时完成浏览器登录。GPU 服务器如需单独访问私有仓库，使用已授权的仓库级只读凭据。 [GitHub 官方凭据配置](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)

```text
请阅读 docs/LAB_HANDOFF.md 和 versions.lock，接手 TableClear V1。
先确认实验室电脑能 SSH 进 Linux 服务器，再在个人数据盘准备依赖和模型。
通过场景 01 的 CUDA、EGL 和原生 OSC 验收后，按交接运行冻结的配对实验。
```

- Git 保存源码、评测入口、固定版本和交接文档。模型、资产、Python 环境、结果和访问凭据保存在各自的用户空间。
- 需要回看历史结果时，单独从原电脑的 `TableClear-V1-实验档案` 复制 `formal/v1-formal.zip`、`动作块配对结果.md`、`schedule-ablation` 和 `pi05-cpu-preflight`。π₀.₅ CPU 预检在旧 AutoDL 主机完成，其记录可供核对。
- 当前历史基线为 SmolVLA 直跑 31/45，其中场景 01 为 4/15。本轮 18 集小试、54 集消融和复验等待实验室 GPU 上机。服务器连接地址与凭据通过实验室授权渠道获取。

如果需要带上原对话，可在 Codex 中连接目标主机、在目标主机保存同一个仓库项目，然后从聊天底部的运行位置选择 **Hand off**。在共享电脑上，直接开新会话并用上面的交接文字即可。 [OpenAI 官方文档](https://learn.chatgpt.com/docs/remote-connections#hand-off-a-chat-between-hosts)

## Linux 服务器准备

从实验室电脑确认 SSH 登录后，在服务器用 `findmnt`、`df -h`、`nvidia-smi` 核对个人数据盘和获准使用的卡。把下面的 `lab` 设为已确认可写的个人目录。服务器的 `longclean`、`roboenv` 和 `lerobot` checkout 都放在此目录下。已有的干净、版本匹配的环境可直接复用。

```bash
lab=/absolute/path/to/your/tableclear
workspace="$lab/workspace"
roboenv="$lab/roboenv"
lerobot="$lab/lerobot"
cd "$lab/longclean"
export AUTODL_TMP="$lab"
export UV_PYTHON_INSTALL_DIR="$lab/cache/uv-python"
source scripts/setup_cache.sh
```

1. 按 `versions.lock` 将 RoboEnv、LeRobot checkout 固定到指定提交。RoboEnv 用 `tools/bootstrap.sh`、`roboenv sources sync --backends libero` 和 `roboenv prepare libero` 准备锁定的 LIBERO。用 `roboenv sources status --backends libero` 核对 LIBERO 提交，用 `roboenv doctor libero` 核对仿真解释器。把实际 Python 3.10 虚拟环境的 `bin/python` 路径填入 `sim_python`。 [RoboEnv 准备说明](LINUX_BOOTSTRAP.md)
2. 为 LeRobot 创建独立 Python 3.12 环境。旧环境实测的 Torch 是 `2.8.0+cu128`，先核对实验室驱动支持 CUDA 12.8。以下命令在用户目录安装模型依赖。 [PyTorch 官方安装组合](https://pytorch.org/get-started/previous-versions/#v280)

   ```bash
   uv venv --python 3.12 "$workspace/.kinex/model-python"
   model_python="$workspace/.kinex/model-python/bin/python"
   uv pip install --python "$model_python" --index-url https://download.pytorch.org/whl/cu128 \
     'torch==2.8.0' 'torchvision==0.23.0'
   uv pip install --python "$model_python" -e "$lerobot[smolvla,pi]" \
     'transformers==5.22.1' 'sentencepiece==0.2.1' 'Brotli==1.2.0' PyYAML
   uv pip check --python "$model_python"
   ```

3. 安装 SmolVLA 模型配置并准备权重。`install_smolvla_device.sh` 用于首次创建工作区设备，已有同版本配置时跳过该命令。已有模型可按目录原样复制到服务器用户数据盘。复制 π₀.₅ 后用 `--prepare-tokenizer --verify-weights` 离线验收，首次下载用 `--download`。

   ```bash
   bash scripts/install_smolvla_device.sh "$workspace"
   "$model_python" scripts/download_smolvla.py \
     --output "$workspace/models/smolvla_libero" \
     --base-output "$workspace/models/smolvlm2_500m_video_instruct"
   "$model_python" scripts/native_pi05.py --workspace "$workspace" --download
   ```

## 运行并续跑

用 `nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv` 取得实验室分配的物理卡 UUID。每个 `--gpu` 指定一个 `UUID@EGL_INDEX`。预检会用正在运行的模型和仿真进程核对两个 GPU 绑定，确认后再进入批量。CUDA 和 EGL 可能采用不同的设备排序。 [NVIDIA CUDA 环境变量](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) · [robosuite EGL 实现](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/renderers/context/egl_context.py)

在服务器 `tmux` 会话中，复用上面的环境变量，填入实际解释器和卡的参数。给本次实验选定一个输出目录，续跑时保留该目录和代码版本。

```bash
sim_python=/absolute/path/to/libero-venv/bin/python
run="$workspace/results/lab-backbone-v1"
"$model_python" scripts/run_native_optimization.py
"$model_python" -u scripts/run_native_optimization.py --run \
  --workspace "$workspace" --roboenv "$roboenv" --sim-python "$sim_python" \
  --output "$run" --gpu GPU-UUID-FOR-FIRST-CARD@0 --gpu GPU-UUID-FOR-SECOND-CARD@1
```

- 默认按门槛执行完复验。`--through preflight` 可先验收每张卡的一步 OSC 和主卡上的完整场景 01 单集，`--through pilot` 可停在 18 集小试。所有阶段共享原生 OSC，评测集使用 520 步上限和完整任务指令。
- 三卡时加第三个 `--gpu`，第三卡优先领取开发条件。每卡同时运行一个 episode，一组内的全部策略在同卡执行。复验分给所有获准的卡。先用预检和实际日志评估显存、耗时和渲染，再调整后续批次并发。
- `status.json` 记录当前阶段和错误。各卡在独立目录保留原始动作、episode 记录和运行日志。已完成组会核对并复用，部分完成组用原 GPU 续跑。运行错误的日志保留在原尝试目录，官方 `success=false` 保持有效。

## 实验读数与下一步

- 小试 9 组，开发 18 组。开发索引为 `18、20、23、27、33、46`，每场景各一套，seed 为 0。按三场景净收益和调用次数冻结 `selection.json`。复验为 36 个本轮未参与候选选择的条件，曾用于历史 V1。π₀.₅ 的新块长只有通过开发门槛才加入复验。
- 各阶段 `summary.json` 汇总官方成功数、配对胜负、按场景分层的 95% 配对 bootstrap 区间、双方成功时的步数差、推理耗时、调用数和失败尝试。`pairs.json` 保留配对元数据和证据路径。
- 先读 `pilot/scene-01-review.json`。优先回放一组 π₀.₅ 救回的轨迹和一组两模型都失败的轨迹，如有反向案例也核对。把 `CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID` 设为原集记录的 UUID 和 EGL 编号。然后运行 `scripts/replay_native_video.py --episode-dir "$run/<artifact>" --output <review_dir> --sim-python "$sim_python" --roboenv "$roboenv"`，复核抓取、运输、松爪和目标实际落点。视频脚本用 `ffmpeg` 和 `ffprobe`，并校验状态、双相机哈希和帧数。
- 如果两模型仍反复在同一抓取或运输环节失败，先核验成功示范，再决定是否微调。已经抓住但反复落在篮外时，先验证检测信号，再决定是否测试一次触发式恢复。
