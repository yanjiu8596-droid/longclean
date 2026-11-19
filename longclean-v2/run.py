"""Prepare reproducible fixtures and execute matched π₀.₅ trials."""
import argparse
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from common import (BUDGET, CAMERAS, CHEESE, CHUNK, COMMIT, D, FULL_TASK, GOALS, GPU, LABELS, METHODS, OFFSETS,
                    PREFIX_TASK, REPO, SAUCE, SIM_PYTHON, SOUP, V1, WORKSPACE, read, repo_imports, sha, write)
from vision import Frame, Progress, Schedule, load_locator, sauce_location

repo_imports()
from driver import predict_chunk
from lerobot.processor.env_processor import LiberoProcessorStep
import native_pi05
from run_native_osc import verify_gpu_placement


class Worker:
    def __init__(self, bddl, target, seed, video=False):
        self.path = target
        target.mkdir(parents=True, exist_ok=False)
        args = [str(SIM_PYTHON), '-B', '-u', str(D / 'simulator.py'), '--output', str(target),
                '--bddl', str(bddl), '--seed', str(seed)]
        if video:
            args.append('--video')
        self.log = (target / 'simulator.log').open('w')
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.trace = (target / 'actions.jsonl').open('x', buffering=1)
        self.npz = target / 'observation.npz'
        self.current = self.recv()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.proc.stdin.close()
        self.proc.wait(timeout=50)
        self.proc.stdout.close()
        self.log.close()
        self.trace.close()
        if self.proc.returncode and error is None:
            raise RuntimeError(f'Simulator failed in {self.path}')

    def recv(self):
        raw = self.proc.stdout.readline()
        if not raw:
            raise RuntimeError(f'Simulator closed pipe in {self.path}')
        return json.loads(raw)

    def call(self, data):
        self.proc.stdin.write(json.dumps(data, allow_nan=False) + '\n')
        self.proc.stdin.flush()
        state = self.recv()
        if 'rejected' not in state:
            self.current = state
        return state

    def step(self, actions, target, instruction):
        start = self.current['step']
        result = self.call({'op': 'step', 'actions': actions})
        if 'rejected' in result:
            raise RuntimeError(result['rejected'])
        actual = actions[:result['executed']]
        assert start + len(actual) == result['step']
        self.trace.write(json.dumps({'start_step': start, 'target': target, 'instruction': instruction,
                                     'actions': actual, **result}, allow_nan=False) + '\n')
        return actual

    def goal(self, obj):
        return self.current['audit']['goals'][obj]

    def terminal(self):
        return self.current.get('done') or self.current['step'] >= BUDGET or all(self.current['audit']['goals'].values())


class Pi05:
    def __init__(self, gpu):
        torch.manual_seed(30)
        self.model = native_pi05.open_service(native_pi05.model_directory(WORKSPACE))
        self.processor = LiberoProcessorStep()
        self.gpu = gpu
        self.placement = None

    def reset(self, seed):
        for component in self.model.values():
            component.reset()
        torch.manual_seed(seed)

    def act(self, worker, task, limit=CHUNK):
        with np.load(worker.npz, allow_pickle=False) as obs:
            batch = {f'observation.images.{name}': torch.from_numpy(obs[name]).permute(2, 0, 1)
                     .unsqueeze(0).float().div(255) for name in CAMERAS}
            batch['observation.robot_state'] = {
                'eef': {kind: torch.from_numpy(obs[f'eef_{kind}']).unsqueeze(0) for kind in ('pos', 'quat')},
                'gripper': {'qpos': torch.from_numpy(obs['gripper_qpos']).unsqueeze(0)},
            }
        actions = predict_chunk(self.model, {**self.processor.observation(batch), 'task': task})
        if self.placement is None or self.placement['simulator_pid'] != worker.proc.pid:
            self.placement = verify_gpu_placement(self.gpu, worker.proc.pid)
        return actions[:min(limit, BUDGET - worker.current['step'])]


def instruction(obj):
    return f'put the {LABELS[obj]} in the basket'


def fingerprint(state):
    return {key: state[key] for key in ('step', 'bddl_sha256', 'raw_sha256', 'state_sha256', 'observation_sha256')}


def replay_prefix(worker, prefix, tracker):
    assert fingerprint(worker.current) == prefix['initial']
    for chunk in prefix['chunks']:
        actual = worker.step(chunk['actions'], CHEESE, PREFIX_TASK)
        assert fingerprint(worker.current) == chunk['end']
        event = tracker.observe(Frame.read(worker.npz), CHEESE, actual)
        if event and not worker.goal(CHEESE):
            raise RuntimeError('False placement assertion in the frozen common prefix')
    assert worker.current['audit']['goals'] == {obj: obj == CHEESE for obj in GOALS}


def make_prefix(policy, bddl, folder, layout_seed):
    policy.reset(60)
    tracker = Progress()
    with Worker(bddl, folder / 'prefix', layout_seed) as worker:
        prefix = {'initial': fingerprint(worker.current), 'chunks': []}
        while worker.current['step'] < 320 and not worker.goal(CHEESE) and not worker.terminal():
            actual = worker.step(policy.act(worker, PREFIX_TASK), CHEESE, PREFIX_TASK)
            event = tracker.observe(Frame.read(worker.npz), CHEESE, actual)
            prefix['chunks'].append({'actions': actual, 'end': fingerprint(worker.current),
                                     'cheese_observed': tracker.complete(CHEESE),
                                     'cheese_truth': worker.goal(CHEESE)})
            if event and not worker.goal(CHEESE):
                prefix['failure'] = 'prefix_false_placement'
                break
        if not worker.goal(CHEESE):
            prefix.setdefault('failure', 'prefix_did_not_place_cheese')
        elif not tracker.complete(CHEESE):
            # A common sensor-only hold lets delayed release evidence settle without hand motion.
            for _ in range(2):
                actual = worker.step([[0, 0, 0, 0, 0, 0, -1]] * CHUNK, CHEESE, PREFIX_TASK)
                tracker.observe(Frame.read(worker.npz), CHEESE, actual)
                prefix['chunks'].append({'actions': actual, 'end': fingerprint(worker.current),
                                         'cheese_observed': tracker.complete(CHEESE),
                                         'cheese_truth': worker.goal(CHEESE)})
                if tracker.complete(CHEESE) or not worker.goal(CHEESE):
                    break
        if worker.current['audit']['goals'] != {obj: obj == CHEESE for obj in GOALS}:
            prefix.setdefault('failure', 'prefix_not_at_one_goal')
        prefix['observed_cheese'] = tracker.complete(CHEESE)
        prefix['end'] = fingerprint(worker.current)
        copyfile(worker.npz, folder / 'checkpoint.npz')
    write(folder / 'prefix.json', prefix)
    return prefix


def scripted_soup(worker):
    """Feasibility control only. Use simulator state to demonstrate a legal trajectory."""
    asset = read(V1 / 'grasp-reference.json')
    rotation = np.asarray(asset['base_rotation_matrix'])
    start = np.asarray(worker.current['audit']['site_xyz'])
    actual = np.asarray(worker.current['audit']['positions'][SOUP])
    grab = actual + asset['grasp_xyz_offset_m']
    high = grab + [0, 0, .16]
    safe = start.copy()
    safe[2] = max(.74, start[2])
    others = [np.asarray(worker.current['audit']['positions'][name]) for name in (CHEESE, SAUCE)
              if worker.goal(name)]
    basket = np.asarray(worker.current['audit']['positions']['basket_1'])
    destinations = [np.array([basket[0] + x, basket[1] + y, .66])
                    for x in (-.028, .028) for y in (-.031, .031)]
    over_basket = max(destinations, key=lambda p: min((np.linalg.norm(p[:2] - other[:2]) for other in others),
                                                    default=0))
    lower = over_basket.copy()
    lower[2] = .555
    stages = [(safe, 16, -1), (high, 33, -1), (grab, 44, -1), (grab, 24, 1),
              (high, 29, 1), (over_basket, 44, 1), (lower, 28, 1), (lower, 20, -1), (over_basket, 20, -1)]
    for index, (point, count, grip) in enumerate(stages):
        if index == 4 and not worker.current['audit']['held_soup']:
            return
        for _ in range(count):
            if worker.terminal():
                return
            now = worker.current['audit']
            dp = 1.4 * (point - np.asarray(now['site_xyz'])) / .05
            dr = Rotation.from_matrix(rotation @ np.asarray(now['site_rotation']).T).as_rotvec() / .5
            action = np.r_[np.clip(np.r_[dp, dr], -.85, .85), grip].tolist()
            worker.step([action], SOUP, 'reference scripted soup')


def reference(policy, bddl, target, seed, prefix, offset):
    tracker = Progress()
    with Worker(bddl, target, seed) as worker:
        replay_prefix(worker, prefix, tracker)
        if offset is not None:
            boundary = worker.call({'op': 'move', 'offset': offset})
            if 'rejected' in boundary:
                return {'rejected': boundary['rejected']}
            copyfile(worker.npz, target / 'moved.npz')
        else:
            boundary = worker.current
        policy.reset(61)
        while not worker.terminal() and not worker.goal(SAUCE) and worker.current['step'] < boundary['step'] + 240:
            worker.step(policy.act(worker, instruction(SAUCE)), SAUCE, instruction(SAUCE))
        if not worker.terminal() and not worker.goal(SOUP):
            scripted_soup(worker)
        return {'success': all(worker.current['audit']['goals'].values()), 'steps': worker.current['step'],
                'boundary': fingerprint(boundary), 'goals': worker.current['audit']['goals']}


def prepare_fixture(policy, locate, bddl, folder, seed):
    marker = folder / 'fixture.json'
    prefix = None
    if marker.is_file():
        previous = read(marker)
        if previous.get('reason') != 'soup_not_visible_before_move':
            return previous
        prefix = read(folder / 'prefix.json')
        copyfile(marker, folder / 'fixture-before-occlusion-control.json')
    if prefix is None:
        if folder.exists():
            folder.rename(folder.with_name(folder.name + f'.attempt-{os.getpid()}'))
        folder.mkdir(parents=True)
        prefix = make_prefix(policy, bddl, folder, seed)
    initial_soup = locate(folder / 'checkpoint.npz') if 'failure' not in prefix else None
    record = {'layout_seed': seed, 'prefix': prefix['end'], 'observed_cheese': prefix['observed_cheese'],
              'soup_before': initial_soup, 'candidates': []}
    if 'failure' in prefix:
        record['reason'] = prefix['failure']
    elif not prefix['observed_cheese']:
        record['reason'] = 'prefix_cheese_not_observed'
    else:
        quiet = reference(policy, bddl, folder / 'reference-quiet', seed, prefix, None)
        record['reference_quiet'] = quiet
        if not quiet.get('success'):
            record['reason'] = 'quiet_reference_unsolved'
        else:
            for i, offset in enumerate(OFFSETS):
                result = reference(policy, bddl, folder / f'reference-{i}', seed, prefix, offset)
                record['candidates'].append({'offset': list(offset), **result})
                if result.get('success'):
                    record['offset'] = list(offset)
                    record['perturbed_checkpoint'] = result['boundary']
                    break
            else:
                record['reason'] = 'no_solved_perturbation'
    record['eligible'] = 'offset' in record
    write(marker, record)
    return record


def evaluate(policy, locate, bddl, fixture, target, layout_seed, policy_seed, method, moved):
    if (target / 'result.json').is_file():
        return read(target / 'result.json')
    if target.exists():
        target.rename(target.with_name(target.name + f'.attempt-{os.getpid()}'))
    prefix = read(fixture / 'prefix.json')
    design = read(fixture / 'fixture.json')
    tracker = Progress(locate)
    with Worker(bddl, target, layout_seed, video=True) as worker:
        replay_prefix(worker, prefix, tracker)
        pre = fingerprint(worker.current)
        boundary = worker.call({'op': 'move', 'offset': design['offset']}) if moved else worker.current
        assert fingerprint(boundary) == (design['perturbed_checkpoint'] if moved else pre)
        copyfile(worker.npz, target / 'fork.npz')
        policy.reset(policy_seed)
        before = design['soup_before']
        schedule = Schedule(tracker, method == 'replan', before['xyz_m'] if before else None)
        schedule_events = []
        state_events = []
        completed_reassigned = []
        first_movement = None
        regrasp_steps = 0
        order = []
        while not worker.terminal():
            step = worker.current['step']
            if (method != 'direct' and step <= pre['step'] + CHUNK and not schedule.reordered
                    and not tracker.complete(SOUP) and not tracker.complete(SAUCE)):
                soup = locate(worker.npz)
                info = schedule.change(soup['xyz_m'] if soup else None, sauce_location(Frame.read(worker.npz)))
                if 'after' in info:
                    schedule_events.append({'step': step, **info})
                if first_movement is None:
                    first_movement = {'step': step, **info}
            obj = None if method == 'direct' else schedule.next(step)
            if method != 'direct' and obj is None:
                reason = 'observed_complete'
                break
            task = FULL_TASK if obj is None else instruction(obj)
            if not order or order[-1]['target'] != obj:
                order.append({'step': step, 'target': obj})
            if obj is not None and worker.goal(obj):
                completed_reassigned.append({'step': step, 'target': obj})
            actual = worker.step(policy.act(worker, task), obj, task)
            regrasp_steps += worker.current['completed_regrasp_steps']
            event = tracker.observe(Frame.read(worker.npz), obj, actual)
            if event:
                state_events.append({'step': worker.current['step'], 'matches_audit': worker.goal(obj), **event})
        else:
            reason = 'goal' if all(worker.current['audit']['goals'].values()) else 'budget'
        final = worker.current
    record = {'layout_seed': layout_seed, 'policy_seed': policy_seed, 'method': method, 'moved': moved,
              'success': all(final['audit']['goals'].values()), 'terminal_goals': final['audit']['goals'],
              'steps': final['step'], 'after_fork_steps': final['step'] - pre['step'], 'terminal_reason': reason,
              'prefix_sha256': sha(fixture / 'prefix.json'), 'base_checkpoint': pre,
              'trial_checkpoint': fingerprint(boundary), 'initial_motion_observation': first_movement,
              'schedule_changes': schedule_events, 'target_runs': order, 'state_events': state_events,
              'observed_terminal': tracker.status(), 'completed_reassigned_chunks': completed_reassigned,
              'completed_regrasp_steps': regrasp_steps,
              'gpu_placement': policy.placement}
    write(target / 'result.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'dev', 'fixture', 'pilot'))
    parser.add_argument('card', type=int, choices=GPU)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    base = args.run.resolve()
    plan = read(base / 'plan.json')
    assert plan['v1_commit'] == COMMIT == subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU[args.card] and int(os.environ['MUJOCO_EGL_DEVICE_ID']) == args.card
    bddl = base / plan['bddl']
    assert sha(bddl) == plan['bddl_sha256']
    policy, locate = Pi05(GPU[args.card]), load_locator()
    if args.phase in ('smoke', 'dev', 'fixture'):
        qualified = 0
        for row in plan[args.phase]:
            if row['card'] != args.card:
                continue
            seed = row['layout_seed']
            folder = base / 'fixtures' / str(seed)
            record = prepare_fixture(policy, locate, bddl, folder, seed)
            print('FIXTURE', seed, record['eligible'], record.get('offset'), record.get('reason'), flush=True)
            if args.phase == 'fixture' and record['eligible']:
                qualified += 1
                if qualified == 4:
                    return
            if args.phase in ('smoke', 'dev') and record['eligible']:
                for moved in (True, False):
                    for method in METHODS:
                        target = base / 'dev' / f'{seed}-s30' / ('moved' if moved else 'sham') / method
                        result = evaluate(policy, locate, bddl, folder, target, seed, 30, method, moved)
                        print('EPISODE_VERIFIED', seed, moved, method, result['success'], result['steps'],
                              len(result['schedule_changes']), flush=True)
        if args.phase == 'fixture':
            raise RuntimeError(f'Only {qualified} of four held-out fixtures qualified on card {args.card}')
        return
    frozen = read(base / 'frozen.json')
    for name, digest in frozen['code'].items():
        assert sha(D / name) == digest, name
    for row in frozen['cases']:
        if row['card'] != args.card:
            continue
        seed = row['layout_seed']
        folder = base / 'fixtures' / str(seed)
        assert sha(folder / 'fixture.json') == row['fixture_sha256']
        assert sha(folder / 'prefix.json') == row['prefix_sha256']
        for method in row['order']:
            target = base / 'pilot' / row['tag'] / method
            result = evaluate(policy, locate, bddl, folder, target, seed, row['policy_seed'], method, row['moved'])
            print('EPISODE_VERIFIED', row['tag'], method, result['success'], result['steps'],
                  len(result['schedule_changes']), flush=True)


if __name__ == '__main__':
    main()
