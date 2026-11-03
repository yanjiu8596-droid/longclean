"""Run RoboEnv's audited LIBERO installer without its 30-minute subprocess cap.

Use only after the standard `roboenv prepare libero` command times out while
large wheels are still being extracted. The source audit, dependency lock,
environment isolation, checks and registration remain RoboEnv-owned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from roboenv.provisioning import setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/root/autodl-tmp/roboenv"),
    )
    args = parser.parse_args()

    original_run = setup.subprocess.run

    def run_without_timeout(*run_args, **run_kwargs):
        run_kwargs.pop("timeout", None)
        return original_run(*run_args, **run_kwargs)

    setup.subprocess.run = run_without_timeout
    setup.install(
        "libero",
        args.prefix,
        project=args.project,
        progress=lambda message: print(f"[prepare] {message}", flush=True),
    )
    print("LIBERO interpreter installed and registered.")


if __name__ == "__main__":
    main()

