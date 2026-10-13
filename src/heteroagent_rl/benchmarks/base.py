from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    prompt: str
    entry_point: str
