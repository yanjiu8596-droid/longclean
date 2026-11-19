"""Frozen scene, policy, and experiment parameters."""
from pathlib import Path
import hashlib
import json
import os
import sys

D = Path(__file__).resolve().parent
LAB = Path(os.environ.get('TABLECLEAR_LAB', Path.home() / 'tableclear'))
REPO = LAB / 'longclean'
ROBOENV = LAB / 'roboenv'
WORKSPACE = LAB / 'workspace'
SIM_PYTHON = ROBOENV / '.venv-sim/libero/bin/python'
COMMIT = '42a8fa2d01a60e82d3da8707b421f6b6d722aad1'
V1 = WORKSPACE / 'results/lab-backbone-v1/review/scene01-02-08-integrated-regression'
SOUP = 'alphabet_soup_1'
CHEESE = 'cream_cheese_1'
SAUCE = 'tomato_sauce_1'
GOALS = (CHEESE, SOUP, SAUCE)
BASKET = 'basket_1_contain_region'
LABELS = {CHEESE: 'cream cheese box', SOUP: 'alphabet soup can', SAUCE: 'tomato sauce'}
FULL_TASK = 'put the alphabet soup, the tomato sauce, and the cream cheese box in the basket'
PREFIX_TASK = 'put the cream cheese box in the basket'
CAMERAS = {'camera1': 'agentview_image', 'camera2': 'robot0_eye_in_hand_image'}
BUDGET = 780
CHUNK = 10
ATTEMPT = 130
# The original objects remain available. Prefer the relatively open negative-X direction.
OFFSETS = ((-.12, 0), (.12, 0), (0, .12), (0, -.12))
MOVE_THRESHOLD_M = .08
METHODS = ('direct', 'fixed', 'replan')
GPU = {card: os.environ.get(f'TABLECLEAR_GPU_UUID_{card}', 'unconfigured') for card in (2, 3)}


def repo_imports():
    sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src'), str(REPO / 'kinex_devices/smolvla')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()
