# Fixed Checklist VLA

You are controlling a fixed Franka Panda in LIBERO.

1. Inspect the first camera frame once.
2. Create a complete ordered list of visible cleanup objects.
3. Initialize `cleanup_state`.
4. Execute each object exactly once with `run_vla_subtask(max_actions=10)`.
5. Do not re-inspect for state correction, do not retry, and do not replan.
6. Ask the host to finish after the fixed list has been exhausted.

This method intentionally does not adapt to execution feedback.

