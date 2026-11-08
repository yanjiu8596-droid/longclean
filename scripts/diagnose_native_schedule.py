"""Run paired prompt and allocation diagnostics away from the formal matrix."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from run_native_osc import PROJECT
from run_overnight_native import SCENES, visual_audit


VARIANTS = (
    ("direct", "direct_vla", ()),
    ("fixed_original", "fixed_checklist_vla", ()),
    ("fixed_compact", "fixed_checklist_vla", ("--diagnostic-compact-prompt",)),
    ("closed_original", "closed_loop_vla", ()),
    ("closed_compact", "closed_loop_vla", ("--diagnostic-compact-prompt",)),
    ("closed_compact_long", "closed_loop_vla", ("--diagnostic-compact-prompt", "--diagnostic-attempt-chunks", "26")),
)
INDICES = (18, 20, 23, 27, 33, 46)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    jobs = [(scene, init, name, method, flags) for init in INDICES for scene in SCENES for name, method, flags in VARIANTS]
    (root / "plan.json").write_text(json.dumps({"project_commit": commit, "seed": 0, "budget": 520, "chunk": 10,
        "visual_validation_sha256": sha256(args.visual_validation.read_bytes()).hexdigest(), "jobs": jobs}, indent=2), encoding="utf-8")

    def episode(job):
        scene, init, name, method, flags = job
        for suffix in ("", "-retry1"):
            output = root / f"{scene:02d}-init{init:02d}-{name}{suffix}"
            command = [sys.executable, "-u", str(PROJECT / "scripts" / "run_native_osc.py"), "--output", str(output),
                       "--workspace", str(args.workspace), "--sim-python", str(args.sim_python), "--roboenv", str(args.roboenv),
                       "--scene", str(scene), "--init-index", str(init), "--seed", "0", "--steps", "520", "--chunk-steps", "10",
                       "--method", method, "--fair-methods", "--visual-validation", str(args.visual_validation), *flags]
            with output.with_suffix(".log").open("x", encoding="utf-8") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            path = output / "episode.json"
            record = json.loads(path.read_bytes()) if path.is_file() else {}
            if process.returncode == 0 and type(record.get("success")) is bool and record["project_commit"] == commit:
                actions = (output / "actions.jsonl").read_bytes()
                peak = max(sum(g["satisfied"] for g in row["audit_goals"]) for row in map(json.loads, actions.splitlines()))
                return {"scene": scene, "init": init, "variant": name, "artifact": output.name, "completed": True,
                        **{key: record[key] for key in ("init_state_sha256", "bddl_sha256", "success", "steps", "vla_calls", "visual_calls", "terminal_reason")},
                        "peak_official_targets": peak, "actions_sha256": sha256(actions).hexdigest(),
                        **({"visual_audit": visual_audit(output)} if method == "closed_loop_vla" else {})}
        return {"scene": scene, "init": init, "variant": name, "completed": False, "exit_code": process.returncode,
                "artifact": output.name}

    rows = []
    for method_type, workers in ((False, 6), (True, 2)):
        queue = [job for job in jobs if (job[3] == "closed_loop_vla") == method_type]
        with ThreadPoolExecutor(max_workers=workers) as pool, (root / "results.jsonl").open("a", encoding="utf-8") as index:
            for row in pool.map(episode, queue):
                rows.append(row)
                index.write(json.dumps(row) + "\n")
                index.flush()
                print(f"{len(rows)}/{len(jobs)} {row['artifact']} complete={row['completed']}", flush=True)
    paired = []
    for scene in SCENES:
        for init in INDICES:
            group = [row for row in rows if (row["scene"], row["init"]) == (scene, init) and row["completed"]]
            if len(group) == len(VARIANTS) and len({(row["init_state_sha256"], row["bddl_sha256"]) for row in group}) == 1:
                paired.extend(group)
    report = {"status": "completed" if len(paired) == len(jobs) else "incomplete",
              "planned": len(jobs), "paired_conditions": len(paired) // len(VARIANTS),
              "completed": sum(row["completed"] for row in rows), "paired_successes": Counter(row["variant"] for row in paired if row["success"]),
              "paired_peaks": {name: Counter(row["peak_official_targets"] for row in paired if row["variant"] == name)
                               for name, *_ in VARIANTS}}
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "workspace", "sim-python", "roboenv", "visual-validation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
