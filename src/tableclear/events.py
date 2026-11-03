from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()


class JsonlEventWriter:
    """Minimal append-only JSONL writer for reproducible episode events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def __enter__(self) -> JsonlEventWriter:
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("event writer is not open")
        record = {
            "timestamp_utc": timestamp_utc(),
            "event_type": event_type,
            **dict(payload),
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL record at {path}:{line_number} is not an object"
                )
            yield value

