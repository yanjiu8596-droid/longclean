"""Replay and audit a fixed-policy development trajectory after a verifier change."""
import argparse
import json
from pathlib import Path

from common import read, write
from run import Worker, fingerprint, replay_prefix
from vision import Frame, Progress, load_locator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    root = args.run.resolve()
    trial = root / 'dev' / f'{args.seed}-s30' / 'moved/fixed'
    history = read(trial / 'result.json')
    fixture = root / 'fixtures' / str(args.seed)
    prefix = read(fixture / 'prefix.json')
    design = read(fixture / 'fixture.json')
    tracker = Progress(load_locator())
    observed = []
    output = root / f'observer-replay-{args.seed}'
    with Worker(root / read(root / 'plan.json')['bddl'], output, args.seed) as worker:
        replay_prefix(worker, prefix, tracker)
        boundary = worker.call({'op': 'move', 'offset': design['offset']})
        assert fingerprint(boundary) == history['trial_checkpoint']
        with (trial / 'actions.jsonl').open() as stream:
            for row in map(json.loads, stream):
                if row['start_step'] < boundary['step']:
                    continue
                actions = worker.step(row['actions'], row['target'], row['instruction'])
                assert fingerprint(worker.current) == fingerprint(row)
                event = tracker.observe(Frame.read(worker.npz), row['target'], actions)
                if event:
                    observed.append({'step': row['step'], 'matches_audit': worker.goal(row['target']), **event})
    before = {event['step'] for event in history['state_events'] if not event['matches_audit']}
    after = {event['step'] for event in observed if not event['matches_audit']}
    report = {'layout_seed': args.seed, 'original_false_confirmations': sorted(before),
              'replayed_false_confirmations': sorted(after), 'events': observed,
              'known_false_confirmation_removed': bool(before) and not before & after}
    write(output / 'summary.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
