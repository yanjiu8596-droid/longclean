"""The V1 simulator's JSON response pipe, with per-action sidecar capture."""

import json
import os
from pathlib import Path
import subprocess

from common import HERE


class Simulator:
    def __init__(self, lab, output, seed, *, init=None, dense=False, video=False):
        self.output = Path(output)
        self.command = [
            str(lab.sim_python), "-B", "-u", str(HERE / "sim_worker.py"),
            "--lab", str(lab.root), "--output", str(output), "--seed", str(seed),
        ]
        self.command += ["--sampled"] if init is None else ["--init", str(init)]
        if dense:
            self.command.append("--dense")
        if video:
            self.command.append("--video")

    def __enter__(self):
        self.log = (self.output / "simulator.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        uuid = environment.get("CUDA_VISIBLE_DEVICES", "")
        index = environment.get("MUJOCO_EGL_DEVICE_ID")
        if uuid.startswith("GPU-") and index is not None:
            # The pinned robosuite binding expects an ordinal for its EGL probe.
            environment.update(CUDA_VISIBLE_DEVICES=index, TABLECLEAR_EGL_GPU_UUID=uuid)
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding="utf-8", env=environment,
        )
        try:
            self.initial = self._receive()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Missing simulator response. See {self.output / 'simulator.log'}")
        return json.loads(line)

    def step(self, actions):
        self.process.stdin.write(json.dumps(actions, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.process.stdin.close()
            self.process.stdout.close()
            status = self.process.wait(timeout=90)
            if status and exc_type is None:
                raise RuntimeError(f"Simulator exited {status}. See {self.output / 'simulator.log'}")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise
        finally:
            self.log.close()
