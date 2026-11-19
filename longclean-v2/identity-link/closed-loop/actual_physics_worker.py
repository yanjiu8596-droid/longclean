"""Use the frozen native worker protocol with an evaluator-only grasp sidecar."""
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
NATIVE_SIMULATOR_SHA = "d518e5024d24ea26b220566672603c0c3877412f23bc480a3e6ee92f32c8502b"


def worker_class(api):
    """Return a Worker whose ordinary outputs never include the extra truth."""
    if api.sha(api.D / "simulator.py") != NATIVE_SIMULATOR_SHA:
        raise RuntimeError("The frozen simulator source changed")

    class PhysicsAuditWorker(api.Worker):
        def __init__(self, bddl, target, seed, video=False):
            self.path = target
            target.mkdir(parents=True, exist_ok=False)
            args = [str(api.SIM_PYTHON), "-B", "-u", str(HERE / "actual_physics_simulator.py"),
                    "--output", str(target), "--bddl", str(bddl), "--seed", str(seed)]
            if video:
                args.append("--video")
            self.log = (target / "simulator.log").open("w")
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log, text=True)
            self.trace = (target / "actions.jsonl").open("x", buffering=1)
            self.npz = target / "observation.npz"
            self.current = self.recv()

    return PhysicsAuditWorker
