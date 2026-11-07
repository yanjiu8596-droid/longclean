"""Serve a single native LIBERO episode to a local Kinex controller."""

from __future__ import annotations

import json
import os
import signal
import socketserver
from pathlib import Path


def capture_for_kinex(output: Path, frame: int) -> dict:
    import numpy as np
    from PIL import Image

    raw_images, images = [], []
    with np.load(output / "observation.npz", allow_pickle=False) as obs:
        for camera in ("camera1", "camera2"):
            rgb = Image.fromarray(obs[camera])
            raw = output / f"{frame:04d}-{camera}-raw.png"
            oriented = output / f"{frame:04d}-{camera}.png"
            rgb.save(raw)
            rgb.transpose(Image.Transpose.ROTATE_180).save(oriented)
            raw_images.append(str(raw.resolve()))
            images.append(str(oriented.resolve()))

        # Match pinned LeRobot LiberoProcessorStep with float32 quaternion math.
        quat = obs["eef_quat"].astype(np.float32)
        one = np.float32(1)
        w = np.clip(quat[3], -one, one)
        denominator = np.sqrt(one - w * w)
        axis_angle = np.zeros(3, dtype=np.float32)
        if denominator > 1e-10:
            axis_angle = (quat[:3] / denominator) * (np.float32(2) * np.arccos(w))
        state = np.concatenate([obs["eef_pos"], axis_angle, obs["gripper_qpos"]]).astype(np.float32)

    return {"raw_images": raw_images, "images": images, "state": state.tolist()}


def serve(socket_path, output, initial, budget, validate, advance) -> None:
    frame = 0
    current = {**initial, **capture_for_kinex(output, frame), "frame": frame, "terminal": False}
    (output / "initial.json").write_text(json.dumps(current) + "\n", encoding="utf-8")

    class Handler(socketserver.StreamRequestHandler):
        timeout = 10

        def handle(self):
            nonlocal frame, current
            try:
                request = json.loads(self.rfile.readline(65_537))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                operation = request.get("op")
                if operation == "step":
                    if current["terminal"]:
                        raise ValueError("episode already ended")
                    if type(request.get("frame")) is not int or request["frame"] != frame:
                        raise ValueError("stale native observation")
                    actions = request.get("actions")
                    validate(actions)
                elif operation != "observe":
                    raise ValueError("expected observe or step")
            except (ValueError, OverflowError) as exc:
                self.reply({"error": str(exc)})
                return

            if operation == "step":
                result = advance(actions[:budget - frame])
                executed = result["executed"]
                start = frame
                frame += executed
                terminal = result["success"] or result["done"] or frame == budget
                current = {
                    **current, **result, **capture_for_kinex(output, frame),
                    "frame": frame, "terminal": terminal,
                }
                trace.write(json.dumps({
                    **current, "start_step": start, "actions": actions[:executed],
                }, allow_nan=False) + "\n")
            self.reply(current)

        def reply(self, response):
            try:
                self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                # The trace and frame counter preserve the result if a client disconnects.
                pass

    class Server(socketserver.UnixStreamServer):
        def handle_error(self, request, client_address):
            # A failed simulation step can leave the world partly advanced.
            raise

    def stop(_signal, _frame):
        raise SystemExit(0)

    with (output / "actions.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
        with Server(str(socket_path), Handler) as server:
            try:
                os.chmod(socket_path, 0o600)
                signal.signal(signal.SIGTERM, stop)
                server.serve_forever()
            finally:
                socket_path.unlink(missing_ok=True)
