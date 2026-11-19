"""Stage 4 progress adapter: one authoritative soup writer, two shadow streams.

The existing planner keeps its ten-step decision boundary. Run the verified
stage-3d observer on both arms at the five-step observation cadence, without
calling the policy or changing any issued actions at the midpoint.
"""
from dataclasses import replace
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
IDENTITY = HERE.parent
ROOT = IDENTITY.parent
for path in (ROOT, IDENTITY):
    sys.path.insert(0, str(path))

from common import SOUP
from stage3d_ablation import factory_plane
from vision import ObjectStatus, Progress


SOUP_WRITER = "stage3d_plane"
STRIDE = 5
_TIMES = ("step", "release_step", "evidence_step")
_TIME_LISTS = ("transport_steps", "baseline_steps")


def absolute_event(event, fork_step):
    """Convert verified relative observer times to action-trace coordinates."""
    result = dict(event)
    for field in _TIMES:
        if field in result:
            result[field] += fork_step
    for field in _TIME_LISTS:
        if field in result:
            result[field] = [sample + fork_step for sample in result[field]]
    if "views" in result:
        result["views"] = [{**view, "depth": [
            {**record, "step": record["step"] + fork_step} for record in view["depth"]
        ]} for view in result["views"]]
    return result


class ClosedLoopProgress(Progress):
    """Keep the frozen legacy planner and non-soup observers unchanged.

    In the control arm the original soup observer owns progress, while the new
    observer serves as a shadow. In the active arm only the new observer can
    mark soup complete; the original soup writer is bypassed. The same
    observer instance is initialized just after the shared fork in both arms.
    """

    def __init__(self, locate_soup=None, *, write_soup=False):
        super().__init__(locate_soup)
        if type(write_soup) is not bool:
            raise TypeError("write_soup must be a boolean arm assignment")
        self.write_soup = write_soup
        self._soup = factory_plane()
        self._fork = None
        self._end = None

    def start_soup(self, frame, step):
        """Seed the sensor-only stream at the post-move fork; no prefix edits."""
        if self._fork is not None or type(step) is not int or step < 0 or step % STRIDE:
            raise ValueError("The soup observer needs one fork on the five-step grid")
        self._fork = self._end = step
        self._observe_soup(frame, [], step)

    def _observe_soup(self, frame, actions, step):
        return self._soup.observe(replace(frame, source=None), actions, step - self._fork)

    def observe_soup(self, frame, actions, step):
        """Observe the realized, contiguous half-block and return audit events.

        A short final slice is still passed to the verified observer, which
        ignores an off-grid sample. No truth, target or file path reaches it.
        """
        if (self._fork is None or type(step) is not int or not 0 < len(actions) <= STRIDE
                or self._end % STRIDE or step != self._end + len(actions)):
            raise ValueError(f"Noncontiguous soup observation at absolute step {step}")
        self._end = step
        result = []
        for event in self._observe_soup(frame, actions, step):
            if event.get("object") != SOUP or event.get("step") != step - self._fork:
                raise RuntimeError("Unexpected object or time from the frozen soup observer")
            applied = False
            if event["event"] == "in_basket" and self.write_soup and not self.complete(SOUP):
                self.state.get(self.ids[SOUP]).status = ObjectStatus.IN_BASKET
                applied = True
            result.append({**absolute_event(event, self._fork), "writer": SOUP_WRITER, "applied": applied})
        return result

    def observe(self, frame, target, actions):
        """Observe legacy progress on the original decision grid, even at prefix."""
        if self.write_soup and target == SOUP:
            # Explicitly suppress the legacy soup writer rather than accepting
            # either detector. Reset any short-lived legacy carry target just
            # as a target change would; next non-soup update is unchanged.
            return super().observe(frame, None, actions)
        return super().observe(frame, target, actions)
