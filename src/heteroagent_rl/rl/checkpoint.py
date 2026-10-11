from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


RUN_FILES = (
    "episodes.checkpoint.jsonl",
    "trajectories.jsonl",
    "summary.json",
    "run_config.json",
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_by_task(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == sum(1 for _ in path.open("r", encoding="utf-8")):
                    break
                raise
            task_id = row.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"Checkpoint row {line_number} is missing task_id.")
            rows[task_id] = row
    return rows


def prepare_run_directory(
    output_dir: Path,
    *,
    run_config: dict[str, Any],
    fresh: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        for name in RUN_FILES:
            path = output_dir / name
            if path.exists():
                path.unlink()

    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError(
                "Existing checkpoint configuration does not match this run. "
                "Use --fresh to start a new run in this output directory."
            )
    else:
        atomic_write_json(config_path, run_config)

    return output_dir / "episodes.checkpoint.jsonl"
