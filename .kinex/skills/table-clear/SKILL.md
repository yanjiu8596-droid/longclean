---
name: table-clear
description: Clear every object from the LIBERO tabletop into the basket using bounded SmolVLA subtasks and visual progress checks.
---

# Table Clear

Use this skill whenever the task is to clear all objects from the current
tabletop into the basket.

## Rules

1. Never use simulator ground truth or hidden task state.
2. Observe both head and wrist cameras before making the first plan.
3. Use `cleanup_state` in `init` mode exactly once per episode.
4. Select the first unfinished object returned by the state tool.
5. Ask `run_vla_subtask` to execute at most 10 control actions for one object.
6. Capture fresh head and wrist images after every VLA call.
7. Visually verify whether the target is on the table, grasped, in the basket,
   missing, or unchanged.
8. Record the result with `cleanup_state` in `update` mode.
9. If the object is still actionable, retry or re-localize it.
10. Never finish while the state tool reports unfinished objects.
11. A voluntary finish is not proof of success. The simulator owns the final
    verdict.

## Per-object loop

```text
read state
-> choose unfinished object
-> generate a short subtask instruction
-> run_vla_subtask(max_actions=10)
-> observe again
-> verify object status
-> update state
-> repeat
```

## Failure handling

- Grasp missed: retry the same object only while attempts remain.
- Object dropped: search the visible tabletop before ending.
- Object not found: scan both cameras and mark `missing`.
- Object unreachable: mark `unreachable`; do not claim the table is clear.
- Uncertain image: keep the object unfinished.

