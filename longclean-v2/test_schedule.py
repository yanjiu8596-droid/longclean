"""Test scene-independent scheduling invariants without a GPU."""
from common import CHEESE, SAUCE, SOUP
from vision import Progress, Schedule
from tableclear.models import ObjectStatus


def main():
    tracker = Progress()
    tracker.state.get(tracker.ids[CHEESE]).status = ObjectStatus.IN_BASKET
    before = [-.1, -.15, .475]
    moved = [-.22, -.15, .475]
    sauce = [-.1, .05, .49]
    fixed = Schedule(tracker, False, before)
    assert 'after' not in fixed.change(moved, sauce)
    assert fixed.next(120) == SOUP
    assert fixed.next(250) == SAUCE
    planner = Schedule(tracker, True, before)
    assert planner.change(None, sauce)['status'] == 'unknown'
    assert planner.change(moved, None)['status'] == 'unknown'
    assert 'after' not in planner.change(before, sauce)
    assert planner.next(120) == SOUP
    assert planner.change(moved, sauce)['after'] == [SAUCE, SOUP]
    assert planner.next(130) == SAUCE
    assert 'after' not in planner.change(moved, sauce)
    tracker.state.get(tracker.ids[SAUCE]).status = ObjectStatus.IN_BASKET
    assert planner.next(220) == SOUP
    assert tracker.complete(CHEESE) and CHEESE not in planner.order
    print('SCHEDULE_INVARIANTS_OK')


if __name__ == '__main__':
    main()
