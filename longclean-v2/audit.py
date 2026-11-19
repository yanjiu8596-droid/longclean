"""Verify development behavior, freeze the pilot, and report paired outcomes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

from common import CHEESE, D, METHODS, SAUCE, read, sha, write


def eligible(root, rows):
    result = []
    for row in rows:
        root_dir = root / 'fixtures' / str(row['layout_seed'])
        if (root_dir / 'fixture.json').is_file() and read(root_dir / 'fixture.json')['eligible']:
            result.append(row)
    return result


def verified_changes(arms, paths):
    first = arms['direct']
    for arm, entry in arms.items():
        for key in ('moved', 'layout_seed', 'policy_seed', 'prefix_sha256', 'base_checkpoint', 'trial_checkpoint'):
            assert entry[key] == first[key], (arm, key)
        assert not entry['schedule_changes'] or arm == 'replan', arm
    events = arms['replan']['schedule_changes']
    assert len(events) <= 1
    if not events:
        return False
    event = events[0]
    assert event['distance_m'] >= .08 and event['after'][0] == SAUCE and CHEESE not in event['after']
    def at_boundary(method):
        with (paths[method] / 'actions.jsonl').open() as stream:
            return next((row for row in map(json.loads, stream) if row['start_step'] == event['step']), None)
    changed = at_boundary('replan')
    fixed = at_boundary('fixed')
    return bool(changed and fixed and changed['target'] == SAUCE and changed['actions'] != fixed['actions'])


def freeze(root):
    plan = read(root / 'plan.json')
    assert all((root / 'fixtures' / str(row['layout_seed']) / 'fixture.json').is_file() for row in plan['dev'])
    dev = eligible(root, plan['dev'])
    assert dev
    checks = []
    for row in dev:
        seed = row['layout_seed']
        for moved, name in ((True, 'moved'), (False, 'sham')):
            paths = {method: root / 'dev' / f'{seed}-s30' / name / method for method in METHODS}
            arms = {method: read(path / 'result.json') for method, path in paths.items()}
            change = verified_changes(arms, paths)
            assert moved or not arms['replan']['schedule_changes'], 'Unperturbed development control changed order'
            checks.append({'layout_seed': seed, 'moved': moved, 'executed_change': change,
                           'false_visual_confirmations': {method: sum(not event['matches_audit'] for event in arm['state_events'])
                                                          for method, arm in arms.items()},
                           'success': {method: arm['success'] for method, arm in arms.items()}})
    assert any(row['executed_change'] for row in checks if row['moved']), 'No development trial executed a replan'
    write(root / 'development-verification.json', {'rows': checks, 'layouts_qualified': len(dev)})
    candidates = eligible(root, plan['fixture'])
    accepted = sorted([row for card in (2, 3) for row in [v for v in candidates if v['card'] == card][:4]],
                      key=lambda row: row['layout_seed'])
    assert len(accepted) == plan['layout_count'] == 8
    cases = []
    for i, row in enumerate(accepted):
        layout = row['layout_seed']
        fixture = root / 'fixtures' / str(layout)
        seeds = [(True, n) for n in plan['suffix_seeds']] + [(False, plan['sham_seed'])]
        for j, (moved, policy_seed) in enumerate(seeds):
            tag = f'{layout}-s{policy_seed}-' + ('moved' if moved else 'sham')
            order = list(METHODS)
            offset = (i + j) % len(order)
            cases.append({'tag': tag, 'layout_seed': layout, 'policy_seed': policy_seed, 'moved': moved,
                          'card': 2 + (i + j) % 2, 'order': order[offset:] + order[:offset],
                          'fixture_sha256': sha(fixture / 'fixture.json'), 'prefix_sha256': sha(fixture / 'prefix.json')})
    if (root / 'frozen.json').exists():
        raise FileExistsError(root / 'frozen.json')
    files = ('common.py', 'simulator.py', 'vision.py', 'run.py', 'prepare.py', 'start.sh', 'audit.py',
             'test_schedule.py', 'check_observer.py')
    write(root / 'frozen.json', {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(root / 'plan.json'),
                                'code': {name: sha(D / name) for name in files}, 'cases': cases})
    print('FROZEN', len(cases), len(METHODS) * len(cases), 'episodes')


def summarize(root):
    frozen = read(root / 'frozen.json')
    for name, digest in frozen['code'].items():
        assert sha(D / name) == digest, name
    assert sha(root / 'plan.json') == frozen['plan_sha256']
    cases = []
    by_condition = defaultdict(list)
    layouts = defaultdict(int)
    representatives = []
    for case in frozen['cases']:
        tag = case['tag']
        paths = {method: root / 'pilot' / tag / method for method in METHODS}
        arms = {method: read(path / 'result.json') for method, path in paths.items()}
        for method, arm in arms.items():
            assert arm['method'] == method and arm['prefix_sha256'] == case['prefix_sha256']
        actual = verified_changes(arms, paths)
        placements = {}
        for method, path in paths.items():
            with (path / 'actions.jsonl').open() as stream:
                placements[method] = [event for block in map(json.loads, stream)
                                      for event in block['transitions'] if event['step'] > arms[method]['base_checkpoint']['step']]
        row = {'case': case, 'executed_change': actual, 'post_fork_goal_transitions': placements, 'arms': arms}
        cases.append(row)
        key = 'moved' if case['moved'] else 'sham'
        by_condition[key].append(row)
        if key == 'moved':
            fixed = arms['fixed']['success']
            replan = arms['replan']['success']
            layouts[case['layout_seed']] += int(replan) - int(fixed)
            if actual and replan and not fixed:
                representatives.append(tag)
    metrics = {}
    for cond, group in by_condition.items():
        per_method = {}
        for method in METHODS:
            rows = [trial['arms'][method] for trial in group]
            successful = [row['after_fork_steps'] for row in rows if row['success']]
            per_method[method] = {
                'successes': sum(row['success'] for row in rows), 'episodes': len(rows),
                'median_steps_after_fork_if_success': median(successful) if successful else None,
                'episodes_regrasping_completed_object': sum(row['completed_regrasp_steps'] > 0 for row in rows),
                'audited_false_visual_confirmations': sum(not e['matches_audit'] for row in rows for e in row['state_events']),
            }
        metrics[cond] = {'methods': per_method,
                         'replan_wins_over_fixed': sum(t['arms']['replan']['success'] and not t['arms']['fixed']['success'] for t in group),
                         'replan_losses_against_fixed': sum(t['arms']['fixed']['success'] and not t['arms']['replan']['success'] for t in group),
                         'emitted_plan_changes': sum(bool(row['arms']['replan']['schedule_changes']) for row in group),
                         'executed_changes': sum(row['executed_change'] for row in group)}
    assert len(cases) == 24 and len(by_condition['moved']) == 16 and len(by_condition['sham']) == 8
    report = {'status': 'complete', 'layouts': 8, 'episodes': 72, 'metrics': metrics,
              'net_wins_by_layout': layouts,
              'passes_benefit_criterion': sum(v > 0 for v in layouts.values()) >= 2
              and metrics['sham']['replan_losses_against_fixed'] == 0,
              'representative_rescues': representatives, 'cases': cases}
    write(root / 'summary.json', report)
    pending = root / 'evidence.pending.zip'
    with ZipFile(pending, 'w', ZIP_DEFLATED, compresslevel=3) as archive:
        for name in ('plan.json', 'frozen.json', 'summary.json', 'development-verification.json'):
            archive.write(root / name, name)
        for path in sorted(root.glob('observer-replay-*/summary.json')):
            archive.write(path, path.relative_to(root))
        bddl = read(root / 'plan.json')['bddl']
        archive.write(root / bddl, bddl)
        for name in frozen['code']:
            archive.write(D / name, f'source/{name}')
        for path in sorted((root / 'fixtures').glob('*/fixture.json')):
            archive.write(path, path.relative_to(root))
            for name in ('prefix.json', 'checkpoint.npz'):
                item = path.with_name(name)
                archive.write(item, item.relative_to(root))
        for row in frozen['cases']:
            for method in METHODS:
                trial = root / 'pilot' / row['tag'] / method
                for name in ('result.json', 'actions.jsonl', 'fork.npz', 'replay.mp4'):
                    archive.write(trial / name, trial.joinpath(name).relative_to(root))
    with ZipFile(pending) as check:
        assert check.testzip() is None
    output = root / 'evidence.zip'
    pending.replace(output)
    (root / 'evidence.sha256').write_text(f'{sha(output)}  evidence.zip\n')
    print({key: val for key, val in report.items() if key != 'cases'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'summarize'))
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else summarize)(args.run.resolve())


if __name__ == '__main__':
    main()
