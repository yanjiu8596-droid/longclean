"""Run the native OSC smoke, pilot and formal comparison behind validation gates."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resume_native_diagnostics import stop
from run_overnight_native import EPISODE_SCRIPT, METHODS, SCENES, visual_audit


PROJECT = EPISODE_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from tableclear.config import expand_matrix, load_evaluation_config


def jobs_for(phase):
    if phase == "smoke":
        return [(1, 0, 15, "closed_loop_vla")]
    if phase == "pilot":
        conditions = [(scene, 0, init) for init in (15, 16, 17) for scene in SCENES]
    else:
        config = load_evaluation_config(PROJECT / "configs" / "v1_eval.toml")
        scenes = {scene.id: int(scene.source_task.rsplit("-", 1)[1]) for scene in config.scenes}
        matrix = {(scenes[job.scene_id], job.seed, job.init_state_index, job.method_id) for job in expand_matrix(config)}
        conditions = sorted({job[:3] for job in matrix}, key=lambda job: (job[1], job[2], job[0]))
        if matrix != {(*condition, method) for condition in conditions for method in METHODS}:
            raise ValueError("formal matrix does not match the paired methods")
    return [
        (*condition, METHODS[(index + offset) % len(METHODS)])
        for index, condition in enumerate(conditions) for offset in range(len(METHODS))
    ]


def key(job):
    scene, seed, init, method = job
    return f"{scene:02d}-seed{seed}-init{init:02d}-{method}"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def inspect_episode(job, artifact, code, elapsed, commit):
    try:
        episode = json.loads((artifact / "episode.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        episode = {}
    scene, seed, init, method = job
    completed = (
        code == 0 and isinstance(episode.get("success"), bool)
        and episode.get("method_protocol") == "shared-action-budget-v1"
        and episode.get("step_budget") == 520 and episode.get("chunk_horizon") == 10
        and episode.get("task_id") == f"libero/libero-10-{scene:02d}"
        and episode.get("seed") == seed and episode.get("init_state_index") == init
        and episode.get("method") == method and episode.get("project_commit") == commit
        and isinstance(episode.get("init_state_sha256"), str)
        and episode.get("terminal_reason") in {"simulator_success", "simulator_done", "step_budget", "visual_clear"}
    )
    row = {
        "run": key(job), "artifact": artifact.name, "scene": scene, "seed": seed, "init": init,
        "method": method, "completed": completed, "exit_code": code, "elapsed_s": round(elapsed, 2),
        **{name: episode.get(name) for name in (
            "success", "steps", "step_budget", "chunk_horizon", "init_state_sha256", "bddl_sha256",
            "vla_calls", "visual_calls", "model", "vision", "project_commit", "terminal_reason", "error",
            "supplemental_camera", "cuda_peak_reserved_mib",
        )},
    }
    if not completed:
        row["success"] = None
    if completed and method == "closed_loop_vla":
        row["visual"] = visual_audit(artifact)
        observations = read_jsonl(artifact / "visual.jsonl")
        trace = read_jsonl(artifact / "actions.jsonl")
        row["visual_switches"] = sum(
            observation["stage"] == "progress" and observation["status"] == "in_basket"
            and any(frame["start_step"] == observation["frame"] and frame["target_id"] != observation["target_id"] for frame in trace)
            for observation in observations
        )
    return row


def smoke_passed(row):
    audit = row.get("visual", {})
    return bool(
        row["completed"] and audit["inventory"] == 2 and audit["progress"] > 0
        and audit["unparsed"] == audit["abstentions"] == audit["basket_mismatches"] == 0
        and audit["confirmed_positives"] > 0 and row["visual_switches"] > 0
    )


def paired(rows, count):
    conditions = {}
    for row in rows:
        case = row["scene"], row["seed"], row["init"]
        by_method = conditions.setdefault(case, {})
        if row["method"] in by_method:
            return False
        by_method[row["method"]] = tuple(row[k] for k in (
            "init_state_sha256", "bddl_sha256", "model", "step_budget", "chunk_horizon", "supplemental_camera",
        ))
    return (
        len(conditions) == count
        and all(set(by_method) == set(METHODS) and len(set(by_method.values())) == 1 for by_method in conditions.values())
    )


def run(args):
    import fcntl

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pin = sha256(args.visual_validation.read_bytes()).hexdigest()
    if json.loads(args.visual_validation.read_bytes()).get("gate_passed") is not True:
        raise ValueError("visual validation did not pass")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    summary_path = root / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_bytes())
        if (previous["project_commit"], previous["visual_validation_sha256"]) != (commit, pin):
            raise ValueError("resume requires the original commit and visual validation report")
    summary = {
        "status": "running", "pid": os.getpid(), "project_commit": commit, "visual_validation_sha256": pin,
        "phases": {}, "active": [],
    }
    active = {}

    def save():
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        summary["active"] = [artifact.name for _, artifact, _, _ in active.values()]
        temp = summary_path.with_suffix(".tmp")
        temp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temp.replace(summary_path)

    def launch(job, directory, retry):
        base = key(job)
        artifact = directory / (f"{base}-retry{retry}" if retry else base)
        if artifact.exists() or artifact.with_suffix(".log").exists():
            return False
        command = [
            sys.executable, "-u", str(EPISODE_SCRIPT), "--workspace", str(args.workspace),
            "--roboenv", str(args.roboenv), "--sim-python", str(args.sim_python), "--output", str(artifact),
            "--scene", str(job[0]), "--seed", str(job[1]), "--init-index", str(job[2]), "--method", job[3],
            "--chunk-steps", "10", "--steps", "520", "--fair-methods", "--visual-validation", str(args.visual_validation),
        ]
        with artifact.with_suffix(".log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[job] = (proc, artifact, time.monotonic(), retry)
        return True

    def phase(name, visual_peak_mib=0):
        summary["phase"] = name
        directory = root / name
        directory.mkdir(exist_ok=True)
        index = directory / "results.jsonl"
        planned = jobs_for(name)
        ids = {key(job) for job in planned}
        existing = read_jsonl(index)
        done = {row["run"]: row for row in existing if row["completed"]}
        if not done.keys() <= ids or len(done) != sum(row["completed"] for row in existing):
            raise ValueError(f"unexpected completed entries in {index}")
        pending = deque((job, 0) for job in planned if key(job) not in done)
        errors = sum(not row["completed"] for row in existing)
        degraded = False
        blocked = False

        def save_progress(workers, visual_workers):
            summary["phases"][name] = {
                "completed": len(done), "planned": len(planned), "errors": errors,
                "successes": dict(Counter(row["method"] for row in done.values() if row["success"])),
                "workers": workers, "visual_workers": visual_workers,
                "visual_peak_estimate_mib": visual_peak_mib,
            }
            save()

        save_progress(0, 0)
        while pending or active:
            if name == "smoke" or degraded:
                workers = visual_workers = 1
            else:
                workers, visual_workers = args.workers, args.visual_workers
                limit = root / "worker-limit.json"
                if limit.exists():
                    limits = json.loads(limit.read_bytes())
                    workers = max(1, min(8, int(limits.get("workers", workers))))
                    visual_workers = max(1, min(workers, 2, int(limits.get("visual_workers", visual_workers))))
            for _ in range(len(pending)):
                if len(active) >= workers or blocked:
                    break
                job, retry = pending.popleft()
                active_visual = sum(j[3] == "closed_loop_vla" for j in active)
                is_visual = job[3] == "closed_loop_vla"
                estimated = (active_visual + is_visual) * visual_peak_mib + (
                    len(active) - active_visual + (not is_visual)
                ) * args.baseline_peak_mib
                if (is_visual and active_visual >= visual_workers) or (active and estimated > args.gpu_budget_mib):
                    pending.append((job, retry))
                    continue
                while not launch(job, directory, retry):
                    retry += 1
            for job, (process, artifact, started, retry) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[job]
                row = inspect_episode(job, artifact, code, time.monotonic() - started, commit)
                with index.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({"phase": name, **row}), flush=True)
                if row["completed"]:
                    done[row["run"]] = row
                else:
                    errors += 1
                    degraded = True
                    if retry == 0:
                        pending.appendleft((job, 1))
                    else:
                        blocked = True
            save_progress(workers, visual_workers)
            if blocked and not active:
                break
            if active:
                time.sleep(2)
        return list(done.values()), len(done) == len(planned)

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        smoke, complete = phase("smoke")
        if not complete or not smoke_passed(smoke[0]):
            summary["status"] = "smoke_gate_failed"
            return
        # Add headroom for the CUDA context, EGL and non-PyTorch allocations.
        visual_peak_mib = (smoke[0].get("cuda_peak_reserved_mib") or args.gpu_budget_mib) + 1024
        pilot, complete = phase("pilot", visual_peak_mib)
        audits = [row["visual"] for row in pilot if "visual" in row]
        if (
            not complete or not paired(pilot, 9)
            or any(audit["false_positives"] or audit["unparsed"] for audit in audits)
            or not any(row.get("visual_switches") for row in pilot)
        ):
            summary["status"] = "pilot_gate_failed"
            return
        formal, complete = phase("formal", visual_peak_mib)
        summary["status"] = "completed" if complete and paired(formal, 45) else "formal_incomplete"
    except KeyboardInterrupt:
        summary["status"] = "stopped"
    except Exception:
        summary["status"] = "runtime_error"
        raise
    finally:
        for process, *_ in active.values():
            stop(process)
        active.clear()
        save()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "workspace", "roboenv", "sim-python", "visual-validation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--visual-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gpu-budget-mib", type=int, default=22500)
    parser.add_argument("--baseline-peak-mib", type=int, default=2800)
    run(parser.parse_args())
