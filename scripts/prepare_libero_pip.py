"""RoboEnv-equivalent LIBERO installer using direct pip installation.

This fallback keeps RoboEnv's audit, lock, import probes and registration, but
replaces uv's cache extraction stage with direct pip installs. It is intended
for hosts where uv repeatedly spends more than an hour unpacking large wheels.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboenv.provisioning import setup


MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


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

    def run_with_direct_pip(command, *run_args, **run_kwargs):
        command = list(command)
        run_kwargs.pop("timeout", None)

        if command[:2] == ["uv", "venv"] and "--seed" not in command:
            command.insert(2, "--seed")
            environment = dict(run_kwargs.get("env") or os.environ)
            environment["UV_DEFAULT_INDEX"] = MIRROR
            run_kwargs["env"] = environment
        elif command[:3] == ["uv", "pip", "install"]:
            python = command[command.index("--python") + 1]
            remaining = command[5:]
            wheelhouse = os.environ.get("TABLE_CLEAR_WHEELHOUSE")
            rewritten = []
            skip = False
            for value in remaining:
                if skip:
                    skip = False
                    continue
                if value in {"--index-url", "--extra-index-url"}:
                    if not wheelhouse:
                        rewritten.append(value)
                        rewritten.append(
                            MIRROR
                            if value == "--index-url"
                            else "https://download.pytorch.org/whl/cpu"
                        )
                    skip = True
                else:
                    rewritten.append(value)
            command = [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--prefer-binary",
                "--timeout",
                "120",
                "--retries",
                "5",
                *rewritten,
            ]
            if wheelhouse:
                command[command.index("--retries") + 2 : command.index("--retries") + 2] = [
                    "--no-index",
                    "--find-links",
                    wheelhouse,
                ]
        elif command[:3] == ["uv", "pip", "check"]:
            python = command[command.index("--python") + 1]
            command = [python, "-m", "pip", "check"]

        return original_run(command, *run_args, **run_kwargs)

    setup.subprocess.run = run_with_direct_pip
    setup.install(
        "libero",
        args.prefix,
        project=args.project,
        progress=lambda message: print(f"[prepare] {message}", flush=True),
    )
    print("LIBERO interpreter installed and registered.")


if __name__ == "__main__":
    main()

