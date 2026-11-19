"""Evaluate held-object truth without changing any frozen simulator response."""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE_SIMULATOR_SHA = "d518e5024d24ea26b220566672603c0c3877412f23bc480a3e6ee92f32c8502b"


def load_native():
    path = ROOT / "simulator.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SIMULATOR_SHA:
        raise RuntimeError("The frozen simulator source changed")
    sys.path.insert(0, str(ROOT))
    name = "_actual_progress_frozen_simulator"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen simulator")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def instrument(native):
    class PhysicsAuditSimulator(native.Simulator):
        def __init__(self, bddl, output, seed, video):
            self._grasp_audit_path = output / "grasp-audit.jsonl"
            # The ordinary Worker creates a new directory for every attempt.
            # Refuse to append new evidence to a previous attempt.
            with self._grasp_audit_path.open("x", encoding="utf-8"):
                pass
            super().__init__(bddl, output, seed, video)

        def capture(self, **metrics):
            result = super().capture(**metrics)
            before = native.array_hash(self.env.get_sim_state())
            if before != result["state_sha256"]:
                raise RuntimeError("Native capture changed the simulator state")
            inner = self.env.env
            robot = inner.robots[0]
            held = {obj: bool(inner._check_grasp(robot.gripper, inner.objects_dict[obj]))
                    for obj in native.GOALS}
            if held[native.SOUP] != result["audit"]["held_soup"]:
                raise RuntimeError("Independent grasp audit disagrees with native soup audit")
            if native.array_hash(self.env.get_sim_state()) != before:
                raise RuntimeError("Read-only grasp checks changed the simulator state")
            row = {"step": result["step"], "state_sha256": before,
                   "goals": result["audit"]["goals"], "held_by_target": held}
            with self._grasp_audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            # Exact original response: this sidecar never reaches an observer.
            return result

    native.Simulator = PhysicsAuditSimulator
    return native


def main():
    instrument(load_native()).main()


if __name__ == "__main__":
    main()
