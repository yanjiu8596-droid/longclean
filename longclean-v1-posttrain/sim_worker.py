"""Use V1's simulator and capture per-step observations independently of policy."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import random
from subprocess import check_output
import sys
from xml.etree import ElementTree

from common import BUDGET, DEFAULT_LAB, Lab, SCENE, SOUP, state_vector


def verify_egl():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not uuid or not uuid.startswith("GPU-"):
        raise RuntimeError("Select an authorized GPU UUID before launching LIBERO")
    xml = ElementTree.fromstring(check_output(["nvidia-smi", "-i", uuid, "-q", "-x"], timeout=15))
    if not any(
        process.findtext("pid", "").strip() == str(os.getpid()) and "G" in process.findtext("type", "")
        for process in xml.findall("gpu/processes/process_info")
    ):
        raise RuntimeError(f"LIBERO does not have an EGL context on the requested GPU {uuid}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", type=int)
    start.add_argument("--sampled", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    lab = Lab(args.lab)
    lab.add_imports()
    sys.path[:0] = [str(lab.roboenv / "src"), str(lab.roboenv / "runtime/src")]
    from roboenv_runtime.adapters.libero.config import ensure_libero_config

    ensure_libero_config()
    from libero.libero.envs import env_wrapper
    import numpy as np
    from run_native_osc import simulate

    if args.sampled:
        random.seed(args.seed)
        np.random.seed(args.seed)

    original_env = env_wrapper.OffScreenRenderEnv
    frames = []
    audit = []

    class RecordingEnv(original_env):
        def reset(self):
            self.ticks = 0
            return super().reset()

        def step(self, action):
            result = super().step(action)
            self.ticks = getattr(self, "ticks", 0) + 1
            if self.ticks < 10:
                return result
            if self.ticks == 10:
                verify_egl()

            obs = result[0]
            if args.dense:
                frames.append((
                    obs["agentview_image"][::-1, ::-1].copy(),
                    obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
                    state_vector(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]),
                ))

            env = self.env
            gripper = env.robots[0].gripper
            predicates = env.parsed_problem["goal_state"]
            targets = {goal[1] for goal in predicates}
            contacts = sorted(
                name for name, obj in env.objects_dict.items()
                if name not in targets and name != "basket_1"
                and env.check_contact(gripper.contact_geoms, obj.contact_geoms)
            )
            audit.append({
                "step": self.ticks - 10,
                "soup_held": bool(env._check_grasp(gripper, env.objects_dict[SOUP])),
                "soup_in": any(goal[1] == SOUP and env._eval_predicate(goal) for goal in predicates),
                "goal": bool(self.check_success()),
                "gripper_opening": float(np.abs(obs["robot0_gripper_qpos"]).sum()),
                "non_goal_contacts": contacts,
            })
            return result

    env_wrapper.OffScreenRenderEnv = RecordingEnv
    try:
        simulate(
            args.output, lab.roboenv, args.seed, args.init or 0, None, BUDGET, SCENE,
            args.output / "replay.mp4" if args.video else None,
            rgbd=True, sampled_reset=args.sampled,
        )
    finally:
        env_wrapper.OffScreenRenderEnv = original_env
        if audit:
            with (args.output / "audit.jsonl").open("w", encoding="utf-8") as handle:
                for row in audit:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        if frames:
            front, wrist, state = zip(*frames, strict=True)
            np.savez(args.output / "dense.npz", front=front, wrist=wrist, state=state)


if __name__ == "__main__":
    main()
