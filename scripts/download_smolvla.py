from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
BASE_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BASE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/tableclear-workspace/models/smolvla_libero"),
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path(
            "/root/autodl-tmp/tableclear-workspace/models/"
            "smolvlm2_500m_video_instruct"
        ),
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=args.output,
    )
    snapshot_download(
        repo_id=BASE_REPO_ID,
        revision=BASE_REVISION,
        local_dir=args.base_output,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )
    print(f"Downloaded {REPO_ID}@{REVISION} to {args.output}")
    print(f"Downloaded {BASE_REPO_ID}@{BASE_REVISION} to {args.base_output}")


if __name__ == "__main__":
    main()

