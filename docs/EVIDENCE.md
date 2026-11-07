# TableClear evidence

Keep the Git history and run archives together. Each run records its project
commit, scene, seed, official initial-state index, action horizon, frame count,
and simulator verdict. Preserve both successful and unsuccessful episodes.

The compact archive keeps the full result index and milestone logs. For images
and action traces it selects one simulator-confirmed success per scene with the
fewest executed steps, breaking ties by run ID. Use `--final` after the batch
stops to select from the completed results. The archive manifest identifies the
selected runs, capture time and SHA-256 digest of each archived file.

Keep the compact per-episode metrics alongside the full result index. Together
they record the pairing key and initial-state hash, frozen model and control
configuration, available and executed action steps, model calls, timing, and
simulator verdict. The selected videos illustrate individual outcomes. Success
rates come from the result index across matched conditions.

```bash
python scripts/archive_tableclear_evidence.py --run RUN_DIRECTORY --output SNAPSHOT.zip
python scripts/archive_tableclear_evidence.py --run RUN_DIRECTORY --output FINAL.zip --final --replays REPLAY_ROOT
```

For a representative video, replay the original actions in the official LIBERO
environment. Use the prepared EGL runtime and the same pinned dependencies as
the source run. The command verifies state and camera hashes at chunk boundaries
and checks the encoded frame count before publishing `replay.mp4`.

```bash
python scripts/replay_native_video.py \
  --episode-dir EPISODE_DIRECTORY --output REPLAY_ROOT/EXAMPLE \
  --sim-python /root/venvs/libero-cpu2/bin/python \
  --roboenv /root/autodl-tmp/roboenv
```

The video shows the head camera on the left and the wrist camera on the right
at 20 fps. `replay.json` identifies the source episode, original result, replay
commit and hashes of the source trace and video. Treat it as a verified replay
when using it in a portfolio. Keep large experiment files in the evidence
archive and use Git for code, configs and documentation.

Regenerate the diagnostic statistics directly from the archive. The report uses
only conditions with all four horizons present and matching initial-state
hashes. It reports success counts, paired changes against 50 steps and step
differences on cases where both horizons succeeded.

Record model calls as well as simulator steps because shorter action chunks
request observations and inference more often. Save throughput samples with
their workload and concurrency. Use identical workloads for controlled speed
comparisons. Before comparing high-level methods, verify the same effective
action budget and chunk length in the actual episode runner.

```bash
python scripts/summarize_horizon_diagnostics.py --archive FINAL.zip --output-dir EVIDENCE_DIRECTORY
```
