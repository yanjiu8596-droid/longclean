"""RGB-D evidence and a small observation-driven target scheduler."""
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image

from common import ATTEMPT, CAMERAS, CHEESE, GOALS, LABELS, MOVE_THRESHOLD_M, SAUCE, SOUP, V1, repo_imports, sha

repo_imports()
from native_rgbd_placement import basket_pixels, cloud, color_mask, geometry_evidence, has_deposit
from tableclear.models import ObjectStatus
from tableclear.state import TaskState

LOCATOR_HASH = '05f4b66bae5a260763501bcfc106e1aea2a22e8d11f969968e5fb13f3343636f'


def load_locator():
    path = V1 / 'locate_shape.py'
    assert sha(path) == LOCATOR_HASH
    spec = spec_from_file_location('tableclear_v1_soup_locator', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.locate


@dataclass
class Frame:
    images: tuple
    geometry: dict
    gap: float
    source: object

    @classmethod
    def read(cls, path):
        with np.load(path, allow_pickle=False) as raw:
            rgb = tuple(Image.fromarray(raw[camera][::-1, ::-1].copy()) for camera in CAMERAS)
            geometry = {
                'depths': [raw[f'{camera}_depth'][::-1, ::-1].copy() for camera in CAMERAS],
                'intrinsics': [raw[f'{camera}_intrinsic'] for camera in CAMERAS],
                'extrinsics': [raw[f'{camera}_extrinsic'] for camera in CAMERAS],
                'eef': raw['eef_pos'],
            }
            return cls(rgb, geometry, abs(raw['gripper_qpos']).sum().item(), path)


def sauce_location(frame):
    points = cloud(frame, 0)
    x, y, z = np.moveaxis(points, -1, 0)
    area = (x > -.18) & (x < -.02) & (y > -.01) & (y < .15) & (z > .455) & (z < .541)
    red, green, blue = np.moveaxis(np.asarray(frame.images[0], dtype=np.int16), -1, 0)
    red_band = (red > green + 18) & (red > blue + 18) & (red > 62)
    green_band = color_mask(frame.images[0], LABELS[SAUCE])
    mask = area & green_band & binary_dilation(area & red_band, iterations=5)
    points = points[mask]
    if len(points) < 8 or np.linalg.norm(np.ptp(points[:, :2], axis=0)) > .09:
        return None
    return np.median(points, axis=0).tolist()


class Progress:
    def __init__(self, locate_soup=None):
        self.state = TaskState.from_labels('longclean-v2-scene01', [LABELS[k] for k in GOALS])
        self.ids = dict(zip(GOALS, (row.id for row in self.state.objects), strict=True))
        self.held = None
        self.previous = None
        self.pending_release = None
        self.locate_soup = locate_soup
        self.active = None
        self.tracked_soup = False

    def complete(self, target):
        return self.state.get(self.ids[target]).status == ObjectStatus.IN_BASKET

    def observe(self, frame, target, actions):
        event = None
        if target != self.active:
            self.active = target
            self.held = self.pending_release = None
            self.tracked_soup = False
        if target is None or self.complete(target):
            self.previous = frame
            self.held = self.pending_release = None
            return event
        label = LABELS[target]
        x, y, z = frame.geometry['eef']
        opening = any(action[-1] < -.5 for action in actions)
        near_basket = -.07 < x < .085 and .17 < y < .35
        soup_transport = target == SOUP and y < .16
        if not opening and .039 <= frame.gap <= .069 and .53 < z < .81 and (near_basket or soup_transport):
            pixels = []
            for i, image in enumerate(frame.images):
                xyz = cloud(frame, i) - frame.geometry['eef']
                near = (np.linalg.norm(xyz[..., :2], axis=-1) < .057) & (xyz[..., 2] > -.105) & (xyz[..., 2] < -.012)
                pixels.append(int(np.count_nonzero(near & color_mask(image, label))))
            minimum = {CHEESE: 5, SAUCE: 8, SOUP: 14}[target]
            if soup_transport and max(pixels) >= minimum:
                self.tracked_soup = True
            visible = max(pixels) >= minimum or (target == CHEESE and .039 <= frame.gap <= .049)
            if near_basket and visible and (target != SOUP or self.tracked_soup):
                self.held = (target, frame)
        if self.held and self.held[0] == target and opening and frame.gap >= .052:
            self.pending_release = (target, self.held[1], 3)
            self.held = None
        if self.pending_release and self.pending_release[0] == target:
            _, before, remaining = self.pending_release
            check = geometry_evidence(before, frame, label)
            supported = check and has_deposit(check, upright=True)
            if supported and target == SOUP and self.locate_soup and self.locate_soup(frame.source) is not None:
                supported = False
            if supported and frame.gap >= .067:
                self.state.get(self.ids[target]).status = ObjectStatus.IN_BASKET
                event = {'object': target, 'event': 'in_basket', 'evidence': check, 'grip_gap': frame.gap}
                self.pending_release = None
            else:
                self.pending_release = (target, before, remaining - 1) if remaining > 1 else None
        self.previous = frame
        return event

    def status(self):
        return {name: ('in_basket' if self.complete(name) else 'pending') for name in GOALS}


class Schedule:
    def __init__(self, progress, allow_reorder, before):
        self.progress = progress
        self.allow_reorder = allow_reorder
        self.before = before
        self.order = [k for k in (SOUP, SAUCE, CHEESE) if not progress.complete(k)]
        self.target = None
        self.until = 0
        self.reordered = False

    def change(self, visual_soup, visual_sauce):
        distance = float(np.linalg.norm(np.subtract(visual_soup[:2], self.before[:2]))) if (
            self.before is not None and visual_soup is not None) else None
        audit = {'soup_before_m': self.before, 'soup_now_m': visual_soup, 'distance_m': distance,
                 'sauce_now_m': visual_sauce, 'status': 'unknown' if distance is None or visual_sauce is None else 'observed'}
        if (self.allow_reorder and not self.reordered and distance is not None and distance >= MOVE_THRESHOLD_M
                and visual_sauce is not None and self.order[:2] == [SOUP, SAUCE]
                and not self.progress.complete(SOUP) and not self.progress.complete(SAUCE)):
            previous = list(self.order)
            self.order[:2] = [SAUCE, SOUP]
            self.target = None
            self.reordered = True
            audit.update(before=previous, after=list(self.order))
        return audit

    def next(self, step):
        self.order = [k for k in self.order if not self.progress.complete(k)]
        if not self.order:
            return None
        if self.target not in self.order:
            self.target = self.order[0]
            self.until = step + ATTEMPT
        elif step >= self.until:
            self.order.append(self.order.pop(0))
            self.target = self.order[0]
            self.until = step + ATTEMPT
        return self.target
