"""Native OSC worker. The runner owns the evaluator-only move operation."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import mujoco
import numpy as np

from common import BASKET, BUDGET, CAMERAS, CHEESE, CHUNK, FULL_TASK, GOALS, ROBOENV, SOUP, array_hash, repo_imports, sha

repo_imports()
sys.path[:0] = [str(ROBOENV / 'src'), str(ROBOENV / 'runtime/src')]
from roboenv_runtime.adapters.libero.config import ensure_libero_config
from run_native_osc import validate_actions


def scene_path():
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict

    suite = get_benchmark_dict()['libero_10'](task_order_index=0)
    task = suite.get_task(0)
    return Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file


def install_scene(output):
    import re

    source = scene_path()
    text = source.read_text(encoding='utf-8')
    text, language_count = re.subn(r'\(:language[^\n]*\)', f'(:language {FULL_TASK})', text, count=1)
    anchor = '(In tomato_sauce_1 basket_1_contain_region)'
    if language_count != 1 or text.count(anchor) != 1:
        raise ValueError('Unexpected scene-01 language or goal')
    text = text.replace(anchor, f'{anchor} (In {CHEESE} {BASKET})')
    target = output / source.name
    target.write_text(text, encoding='utf-8')
    return target


class Simulator:
    def __init__(self, bddl, output, seed, video):
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv
        from robosuite.utils import camera_utils

        self.camera_utils = camera_utils
        self.output = output
        self.bddl = bddl
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                      control_freq=20, camera_depths=True, horizon=BUDGET + 10)
        self.frame = 0
        self.moved = False
        self.previously_in = set()
        self.encoder = None
        random.seed(seed)
        self.env.reset()
        self.env.seed(seed)
        self.env.reset()
        self.raw = array_hash(self.env.get_sim_state())
        for _ in range(10):
            self.obs, *_ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        inner = self.env.env
        self.initial_soup_orientation = inner.sim.data.body_xmat[inner.obj_body_id[SOUP]].reshape(3, 3).copy()
        robot = self.env.env.robots[0]
        robot.controller.use_delta = True
        assert self.env.env.control_freq == 20 and robot.controller.name == 'OSC_POSE' and robot.action_dim == 7
        assert set(self.goals()) == set(GOALS)
        assert not any(self.goals().values())
        if video:
            self.encoder = subprocess.Popen([
                'ffmpeg', '-nostdin', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', '512x256', '-framerate', '20', '-i', 'pipe:0', '-c:v', 'libx264',
                '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'replay.mp4'),
            ], stdin=subprocess.PIPE)
            self.encode()

    def goals(self):
        inner = self.env.env
        return {p[1]: bool(inner._eval_predicate(p)) for p in inner.parsed_problem['goal_state']}

    def xyz(self):
        inner = self.env.env
        return {name: inner.sim.data.body_xpos[idx].tolist() for name, idx in sorted(inner.obj_body_id.items())}

    def encode(self):
        if self.encoder is not None:
            rgb = np.concatenate([self.obs[key][::-1, ::-1] for key in CAMERAS.values()], axis=1)
            self.encoder.stdin.write(rgb.tobytes())

    def capture(self, **metrics):
        sim = self.env.env.sim
        observation = {name: self.obs[key] for name, key in CAMERAS.items()}
        array = {**observation, 'eef_pos': self.obs['robot0_eef_pos'], 'eef_quat': self.obs['robot0_eef_quat'],
                 'gripper_qpos': self.obs['robot0_gripper_qpos']}
        for name, key in CAMERAS.items():
            camera = key.removesuffix('_image')
            array[f'{name}_depth'] = self.camera_utils.get_real_depth_map(sim, self.obs[f'{camera}_depth']).squeeze(-1)
            array[f'{name}_intrinsic'] = self.camera_utils.get_camera_intrinsic_matrix(sim, camera, 256, 256)
            array[f'{name}_extrinsic'] = self.camera_utils.get_camera_extrinsic_matrix(sim, camera)
        np.savez(self.output / 'observation.npz', **array)
        robot = self.env.env.robots[0]
        site = robot.controller.eef_name
        return {'step': self.frame, 'raw_sha256': self.raw, 'bddl_sha256': sha(self.bddl),
                'state_sha256': array_hash(self.env.get_sim_state()),
                'observation_sha256': {key: array_hash(value) for key, value in array.items()},
                'audit': {'goals': self.goals(), 'positions': self.xyz(),
                          'site_xyz': sim.data.get_site_xpos(site).tolist(),
                          'site_rotation': sim.data.get_site_xmat(site).tolist(),
                          'held_soup': bool(self.env.env._check_grasp(robot.gripper, self.env.env.objects_dict[SOUP]))},
                **metrics}

    def advance(self, actions):
        validate_actions(actions)
        if not 1 <= len(actions) <= CHUNK or self.frame + len(actions) > BUDGET or self.env.check_success():
            raise ValueError('Invalid block length or terminal episode')
        drop_events = []
        regrasp_steps = 0
        previous = self.goals()
        done = False
        for executed, action in enumerate(actions, start=1):
            self.obs, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32))
            self.frame += 1
            self.encode()
            current = self.goals()
            inner = self.env.env
            if any(inner._check_grasp(inner.robots[0].gripper, inner.objects_dict[k]) for k in self.previously_in):
                regrasp_steps += 1
            self.previously_in.update(k for k, value in current.items() if value)
            drop_events.extend({'step': self.frame, 'object': k, 'in_basket': value}
                               for k, value in current.items() if previous[k] != value)
            previous = current
            if all(current.values()) or done:
                break
        return self.capture(executed=executed, done=bool(done), transitions=drop_events,
                            completed_regrasp_steps=regrasp_steps)

    def move(self, offset):
        goals = self.goals()
        if self.moved or self.frame % CHUNK or goals != {k: k == CHEESE for k in goals}:
            raise ValueError('Move requires the first placed target and a fresh block boundary')
        if len(offset) != 2 or not np.all(np.isfinite(offset)) or not .119 <= np.linalg.norm(offset) <= .121:
            raise ValueError('Expected a fixed 12 cm offset')
        inner = self.env.env
        sim = inner.sim
        # Align both pose snapshots with qpos. No control step or time advance occurs.
        sim.forward()
        obj = inner.objects_dict[SOUP]
        joint, = obj.joints
        geom = obj.contact_geoms
        if inner.check_contact(inner.robots[0].gripper.contact_geoms, geom):
            raise ValueError('Soup currently touches gripper')
        coords = self.xyz()
        original = sim.data.qpos.copy()
        velocity = sim.data.qvel.copy()
        pose = sim.data.get_joint_qpos(joint).copy()
        current_orientation = sim.data.body_xmat[inner.obj_body_id[SOUP]].reshape(3, 3)
        # Compare against the settled asset pose so the authored local cylinder axis is irrelevant.
        if (current_orientation @ self.initial_soup_orientation.T)[2, 2] < .966:
            raise ValueError('Soup is tilted relative to its settled orientation')
        pose[:2] += offset
        if not (-.27 <= pose[0] <= .27 and -.29 <= pose[1] <= .13):
            raise ValueError('Pose lies outside the table clearance bounds')
        sim.data.set_joint_qpos(joint, pose)
        sim.forward()
        try:
            names = [name for name, candidate in inner.objects_dict.items() if name != SOUP for name in candidate.contact_geoms]
            names += list(inner.robots[0].gripper.contact_geoms)
            for src in geom:
                for other in names:
                    ids = [sim.model.geom_name2id(name) for name in (src, other)]
                    if mujoco.mj_geomDistance(sim.model._model, sim.data._data, *ids, .004, None) < .003:
                        raise ValueError('Soup candidate touches or crowds another object')
            assert np.array_equal(velocity, sim.data.qvel)
            first, last = sim.model.get_joint_qpos_addr(joint)
            assert last - first == 7 and set(np.flatnonzero(original != sim.data.qpos)) <= {first, first + 1}
            assert all(np.allclose(pos, self.xyz()[name], atol=1e-10, rtol=0)
                       for name, pos in coords.items() if name != SOUP)
            assert self.goals() == goals
        except BaseException:
            sim.data.qpos[:] = original
            sim.forward()
            raise
        self.obs = self.env.regenerate_obs_from_state(self.env.get_sim_state())
        self.moved = True
        return self.capture(offset_m=list(offset), old_soup_xyz=coords[SOUP])

    def close(self):
        try:
            if self.encoder:
                self.encoder.stdin.close()
                if self.encoder.wait() != 0:
                    raise RuntimeError('FFmpeg failed')
        finally:
            self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bddl', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    ensure_libero_config()
    if args.prepare:
        print(str(install_scene(args.output)), file=protocol)
        return
    with protocol:
        simulator = Simulator(args.bddl, args.output, args.seed, args.video)
        try:
            print(json.dumps(simulator.capture()), file=protocol)
            for line in sys.stdin:
                message = json.loads(line)
                try:
                    operation = message['op']
                    if operation == 'step':
                        response = simulator.advance(message['actions'])
                    elif operation == 'move':
                        response = simulator.move(message['offset'])
                    else:
                        raise ValueError(f'Invalid operation {operation}')
                except ValueError as error:
                    response = {'rejected': str(error)}
                print(json.dumps(response, allow_nan=False), file=protocol)
        finally:
            simulator.close()


if __name__ == '__main__':
    main()
