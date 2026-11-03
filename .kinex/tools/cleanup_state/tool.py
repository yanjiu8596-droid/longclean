from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CleanupStateError(ValueError):
    pass


def execute(request: dict[str, Any], workspace: Path) -> dict[str, Any]:
    action = request.get("action")
    state_path = workspace / ".tableclear" / "state.json"

    if action == "init":
        task_id = request.get("task_id")
        objects = request.get("objects")
        if not isinstance(task_id, str) or not task_id.strip():
            raise CleanupStateError("task_id is required for init")
        if not isinstance(objects, list) or not objects:
            raise CleanupStateError("objects must be a non-empty list for init")
        records = {}
        for index, label in enumerate(objects, start=1):
            if not isinstance(label, str) or not label.strip():
                raise CleanupStateError(
                    "every object label must be a non-empty string"
                )
            object_id = f"object_{index}"
            records[object_id] = {
                "id": object_id,
                "label": label,
                "status": "pending",
                "attempts": 0,
                "last_seen_region": None,
            }
        state: dict[str, Any] = {
            "task_id": task_id,
            "objects": records,
            "events": [],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_atomic(state_path, state)
        return {
            "task_id": task_id,
            "objects": list(records.values()),
            "remaining": list(records),
            "in_basket": [],
        }

    state = _read(state_path)
    if action == "get":
        return {
            **state,
            **_state_summary(state),
        }

    if action == "update":
        target_id = request.get("target_id")
        status = request.get("status")
        if target_id not in state["objects"]:
            raise CleanupStateError(f"unknown target_id: {target_id}")
        allowed = {
            "pending",
            "grasped",
            "in_basket",
            "missing",
            "lost_grip",
            "blocked",
            "unreachable",
        }
        if status not in allowed:
            raise CleanupStateError(f"invalid status: {status}")
        record = state["objects"][target_id]
        record["status"] = status
        if "location" in request:
            record["last_seen_region"] = request["location"]
        record["attempts"] = int(record.get("attempts", 0)) + 1
        state["events"].append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "target_id": target_id,
                "status": status,
                "location": request.get("location"),
                "evidence": request.get("evidence", ""),
            }
        )
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_atomic(state_path, state)
        summary = _state_summary(state)
        return {"updated": record, **summary}

    raise CleanupStateError(f"unsupported action: {action}")


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CleanupStateError("cleanup state is not initialized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupStateError(f"cannot read cleanup state: {exc}") from exc
    if not isinstance(value, dict):
        raise CleanupStateError("cleanup state must be a JSON object")
    return value


def _state_summary(state: dict[str, Any]) -> dict[str, list[str]]:
    objects = state.get("objects")
    if not isinstance(objects, dict):
        raise CleanupStateError("cleanup state objects must be an object")
    remaining = [
        obj_id
        for obj_id, record in objects.items()
        if record.get("status") != "in_basket"
    ]
    in_basket = [
        obj_id
        for obj_id, record in objects.items()
        if record.get("status") == "in_basket"
    ]
    return {"remaining": remaining, "in_basket": in_basket}


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    from kinex import args, fail, finish

    workspace = Path(os.environ.get("KINEX_WORKSPACE", os.getcwd()))
    try:
        result = execute(dict(args()), workspace)
    except CleanupStateError as exc:
        fail(str(exc))
    else:
        finish(result)


if __name__ == "__main__":
    main()


