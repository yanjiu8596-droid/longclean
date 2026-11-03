# Remote Kinex Workspace

Use one workspace per simulator/robot pair:

```bash
mkdir -p /root/autodl-tmp/tableclear-workspace
cd /root/autodl-tmp/tableclear-workspace
source /root/autodl-tmp/longclean/scripts/setup_cache.sh
/root/autodl-tmp/roboenv/.venv/bin/roboenv setup \
  libero/libero-panda-living-room
```

The LIBERO Panda workspace uses these device names:

```text
arm
gripper
camera
camera_wrist
world
```

Copy the project tools and skills after exporting the devices:

```bash
mkdir -p .kinex/tools .kinex/skills
cp -r /root/autodl-tmp/longclean/.kinex/tools/* .kinex/tools/
cp -r /root/autodl-tmp/longclean/.kinex/skills/* .kinex/skills/
```

When the simulator is running on port 8123:

```bash
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
/root/.local/bin/kinex -p "/devices"
/root/.local/bin/kinex -p "/load all"
/root/.local/bin/kinex -p "/ping all"
```

After installing and loading the SmolVLA model device, run the one-episode
agent smoke:

```bash
bash /root/autodl-tmp/longclean/scripts/run_kinex_smoke.sh
```

Finish one simulator session before exporting a different robot catalog.
Do not reuse a Kinex session after the RoboEnv episode changes.

