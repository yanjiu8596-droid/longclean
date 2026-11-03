from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

from .aggregate import aggregate_episode_records, write_summary_csv
from .config import expand_matrix, load_evaluation_config
from .controller import CleanupController
from .events import read_jsonl
from .models import CleanupVerdict, VerdictStatus
from .state import NoActionableObjectError, TaskState


def _doctor(data_root: Path) -> int:
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}")
    print(f"data_root: {data_root}")
    print(f"data_root exists: {data_root.exists()}")

    warnings = 0
    if platform.system() != "Linux":
        print("warning: real Kinex/RoboEnv/MuJoCo runs require Ubuntu")
        warnings += 1
    if not data_root.exists():
        print("warning: data_root does not exist or is not mounted")
        warnings += 1
    else:
        usage = shutil.disk_usage(data_root)
        free_gib = usage.free / (1024**3)
        print(f"free space: {free_gib:.1f} GiB")
        if free_gib < 8:
            print("warning: fewer than 8 GiB remain on the data disk")
            warnings += 1

    expected = {
        "HF_HOME": data_root / "cache" / "huggingface",
        "UV_CACHE_DIR": data_root / "cache" / "uv",
        "ROBOENV_ASSET_CACHE": data_root / "roboenv-assets",
        "TMPDIR": data_root / "tmp",
    }
    for key, recommended in expected.items():
        actual = os.environ.get(key)
        if actual is None:
            print(f"warning: {key} is unset; recommended {recommended}")
            warnings += 1
        else:
            print(f"{key}: {actual}")

    return 0 if warnings == 0 else 1


def _print_matrix(config_path: Path) -> int:
    config = load_evaluation_config(config_path)
    jobs = expand_matrix(config)
    print(f"project: {config.project_name}")
    print(f"scenes: {len(config.scenes)}")
    print(f"methods: {len(config.methods)}")
    print(f"seeds: {list(config.seeds)}")
    print(f"episodes per seed: {config.episodes_per_seed}")
    print(f"total episodes: {len(jobs)}")
    print("first job:", jobs[0])
    print("last job:", jobs[-1])
    return 0


def _demo() -> int:
    state = TaskState.from_labels(
        task_id="tableclear-demo",
        labels=("red mug", "blue bowl", "food box"),
    )
    controller = CleanupController(state, max_attempts=2, max_actions=10)

    while not controller.is_cleared():
        try:
            subtask = controller.next_subtask()
        except NoActionableObjectError:
            break
        print(f"execute: {subtask.instruction}")
        controller.apply_verdict(
            CleanupVerdict(
                target_id=subtask.target_id,
                status=VerdictStatus.IN_BASKET,
                progress=1.0,
                confidence=1.0,
                evidence="deterministic local demo",
            )
        )

    print(f"cleared: {controller.is_cleared()}")
    print(f"remaining: {list(state.remaining_ids())}")
    return 0 if controller.is_cleared() else 1


def _aggregate(inputs: list[Path], output: Path) -> int:
    records = []
    for path in inputs:
        records.extend(read_jsonl(path))
    summaries = aggregate_episode_records(records)
    if not summaries:
        print("no episode_end records found", file=sys.stderr)
        return 1
    written = write_summary_csv(summaries, output)
    print(f"wrote {len(summaries)} method summaries to {written}")
    for summary in summaries:
        print(
            f"{summary.method}: success={summary.successes}/{summary.episodes}, "
            f"leftover={summary.mean_leftover:.2f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tableclear")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check local runtime and storage")
    doctor.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp")),
    )

    matrix = subparsers.add_parser("matrix", help="expand and print a V1 matrix")
    matrix.add_argument("--config", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="summarize one or more episode JSONL files",
    )
    aggregate.add_argument("--input", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("demo", help="run the platform-independent cleanup demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.data_root)
    if args.command == "matrix":
        return _print_matrix(args.config)
    if args.command == "aggregate":
        return _aggregate(args.input, args.output)
    if args.command == "demo":
        return _demo()
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())

