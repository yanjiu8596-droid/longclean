# Third-Party Components

The eventual runtime integrates with these external projects:

- Kinex: <https://github.com/stacx-lab/kinex>
- RoboEnv: <https://github.com/stacx-lab/RoboEnv>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA: <https://huggingface.co/papers/2506.01844>

LIBERO code is MIT-licensed and its datasets are released under CC BY 4.0.
Kinex documentation declares Apache-2.0. RoboEnv currently has no visible
top-level license file, so this repository references a pinned external checkout
and does not redistribute RoboEnv source code.

The local `.kinex/tools/run_policy` override is derived from Kinex 1.0.0
(Apache-2.0). It narrows the generic observation schema to the fixed SmolVLA
observation used by this project.

The pinned `lerobot/smolvla_libero` model card declares Apache-2.0. The exact
model and LeRobot revisions are recorded in `versions.lock`.

Review and update this file before publishing binaries, model weights, simulator
assets, recordings, or modified upstream source.

