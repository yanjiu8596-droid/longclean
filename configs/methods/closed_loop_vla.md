# Closed-Loop VLA

Load and follow the `table-clear` skill exactly.

After every VLA call:

1. observe both `camera` and `camera_wrist`;
2. verify the current object visually;
3. update `cleanup_state`;
4. retry or choose the next unfinished object.

Never finish while `cleanup_state.remaining` is non-empty. Treat an unreachable
object as a failed cleanup, not as success.

