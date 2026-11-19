"""Define the scene and prespecify calibration and held-out candidate pools."""
import argparse
from pathlib import Path
import subprocess

from common import BUDGET, CHUNK, COMMIT, D, SIM_PYTHON, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        raise FileExistsError(output / 'plan.json')
    command = [str(SIM_PYTHON), '-B', '-u', str(D / 'simulator.py'), '--prepare', '--output', str(output)]
    bddl = Path(subprocess.check_output(command, text=True).strip())
    development = [{'layout_seed': 923100 + i, 'card': 2 + i % 2} for i in range(6)]
    candidates = [{'layout_seed': 924100 + i, 'card': 2 + i % 2} for i in range(32)]
    write(output / 'plan.json', {
        'schema': 'longclean-v2-object-moved/1', 'v1_commit': COMMIT, 'bddl': bddl.name, 'bddl_sha256': sha(bddl),
        'budget': BUDGET, 'chunk': CHUNK, 'dev': development, 'smoke': development[:2], 'fixture': candidates,
        'suffix_seeds': [34, 35], 'sham_seed': 34, 'layout_count': 8,
        'eligibility': 'A common reference prefix places cream cheese. Reference succeeds with and without the move.',
        'scope': 'Native scene 01 derivative with three goals, once-per-trial 12 cm translation of the soup.',
    })
    print('PLAN', output, bddl.name, sha(bddl))


if __name__ == '__main__':
    main()
