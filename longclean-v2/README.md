# LongClean V2

LongClean V2 evaluates three-object cleanup in a derived LIBERO scene 01 task. The experiment uses frozen π₀.₅, a Franka Panda in simulation, native relative OSC, 10-action chunks, and a 780-step episode limit. The observation code uses RGB-D and robot state to update progress and schedule recovery. The simulator records predicate success and grasp state for offline evaluation.

## Source and dependencies

The base TableClear commit is `42a8fa2d01a60e82d3da8707b421f6b6d722aad1`. See `pi05-environment.lock` for the lab's π₀.₅ checkpoint and the RoboEnv, LIBERO, and LeRobot versions. The launchers expect a Linux lab installation with `longclean/`, `roboenv/`, and `workspace/` beneath `TABLECLEAR_LAB`, including the pinned model Python environment in `workspace/.kinex/model-python/`.

The V1 integrated regression assets `locate_shape.py`, `grasp-reference.json`, and `trajectory.json` are required in `workspace/results/lab-backbone-v1/review/scene01-02-08-integrated-regression/` under the same root. Historical runs also reference their original plan, freeze, fixture, and audit files. Restore those from the lab evidence archive using the relative paths expected by the launcher. `archive_v2_evidence.py verify <archive>` checks each member against the embedded manifest.

Set `TABLECLEAR_LAB` to a personal data directory and set `TABLECLEAR_GPU_UUID_2` and `TABLECLEAR_GPU_UUID_3` for the authorized GPU slots used by the archived protocols. The launchers validate the selected slot; follow lab scheduling rules before launching through tmux. For CPU audits that compare recorded CUDA device identifiers, supply the slot identifiers from the archived environment.

The repository uses environment variables for deployment paths and GPU identifiers. `deployment-source-hashes.json` records the original and published SHA-256 values for those files. To validate historical byte-level pins, run against the original source from the local evidence archive.

## Entry points

| Study | Entry points |
|---|---|
| Three-object planning and initial comparison | `prepare.py`, `start.sh`, `run.py`, `audit.py` |
| RGB-D identity and independent visual gates | `identity-link/collect.py`, `identity-link/stage3d_gate.py`, `identity-link/stage3d_ablation.py` |
| Online progress and release timing | `identity-link/closed-loop/dense_online.py`, `dense_online_extra.py`, `merge_dense_online.py` in the same directory |
| Basket retreat and paired recovery | `identity-link/closed-loop/basket_retreat_pair.py`, `basket_retreat_fresh.py`, `near_table_grasp_pair.py` in the same directory |
| Low-visibility release diagnosis | `identity-link/closed-loop/sauce_holdout_replay.py`, `sauce_temporal_replay.py` in the same directory |

For a new planning run, prepare a fresh run directory with `prepare.py --run "$run"`, then start the development phase with `bash start.sh dev "$card" "$uuid" "$run"`. The `fixture` and `pilot` phases consume their preceding results. Audit with `audit.py freeze --run "$run"` and `audit.py summarize --run "$run"` at the corresponding stage. Run GPU launchers with a verified empty device and use the saved exit codes and per-episode artifacts to audit the run.
