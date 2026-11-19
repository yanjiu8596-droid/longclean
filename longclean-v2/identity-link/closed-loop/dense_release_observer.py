"""Sample a pending sauce release more often without changing identity or geometry."""
from dataclasses import replace
from functools import lru_cache

from actual_progress_replay import OBSERVER_SHA, PLACEMENT, load_exact
from common import CHUNK, SAUCE


MAX_RELEASE_AGE = 30
OPEN_GAP = .067


@lru_cache(maxsize=1)
def frozen_module():
    return load_exact("frozen_placement_for_dense_release", PLACEMENT / "observer.py", OBSERVER_SHA)


class DenseReleaseObserver:
    """Preserve original 10-step tracking, adding only off-grid sauce confirmations.

    After one `start`, provide contiguous realized control slices contained in
    each original block. The result uses the original event schema. Off-grid
    frames do not alter transport votes, previous frame, or the clear view;
    only successful pending sauce confirmation marks complete and clears pending.
    """

    def __init__(self):
        self._frozen = frozen_module()
        self._coarse = self._frozen.PlacementObserver()
        self._step = None
        self._actions = []

    @property
    def complete(self):
        return self._coarse.complete

    @property
    def needs_dense(self):
        pending = self._coarse.pending
        return pending is not None and pending.target == SAUCE

    def start(self, frame, step):
        if self._step is not None or type(step) is not int or step < 0 or step % CHUNK:
            raise ValueError("Start once at an original decision boundary")
        self._step = step
        self._coarse.clear_view = replace(frame, source=None)

    def observe(self, frame, actions, step):
        """Consume one slice; submit original 10-step blocks to the frozen observer."""
        if (self._step is None or type(step) is not int or not actions or step != self._step + len(actions)
                or len(self._actions) + len(actions) > CHUNK):
            raise ValueError("Realized control slices must be contiguous and not cross the 10-step grid")
        self._step = step
        self._actions.extend(actions)
        if not step % CHUNK:
            if len(self._actions) != CHUNK:
                raise ValueError("Incomplete original action block")
            result = self._coarse.observe(replace(frame, source=None), self._actions, step)
            self._actions = []
            return result
        if not self.needs_dense:
            return []
        pending = self._coarse.pending
        if not (0 < step - pending.step <= MAX_RELEASE_AGE and frame.gap >= OPEN_GAP):
            return []
        pixels, center = self._frozen.deposit(replace(frame, source=None), pending)
        if center is None:
            return []
        self._coarse.complete.add(SAUCE)
        self._coarse.pending = None
        return [{"object": SAUCE, "event": "in_basket", "release_step": pending.step, "pixels": pixels}]
