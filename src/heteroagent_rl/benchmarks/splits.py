from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.6
    dev_fraction: float = 0.2
    seed: str = "heteroagent-rl-v1"

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0.0 <= self.dev_fraction < 1.0:
            raise ValueError("dev_fraction must be between 0 and 1")
        if self.train_fraction + self.dev_fraction >= 1.0:
            raise ValueError("train_fraction + dev_fraction must be less than 1")


def partition_task_ids(
    task_ids: list[str],
    *,
    config: SplitConfig | None = None,
) -> dict[str, list[str]]:
    """Return deterministic, disjoint train/dev/test task IDs.

    IDs are ordered by a stable SHA-256 key derived from the split seed and task
    ID, then sliced into exact-size partitions.
    """
    cfg = config or SplitConfig()
    cfg.validate()

    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{cfg.seed}|{task_id}".encode("utf-8")
        ).hexdigest(),
    )
    n = len(ordered)
    train_end = int(n * cfg.train_fraction)
    dev_end = train_end + int(n * cfg.dev_fraction)

    return {
        "train": ordered[:train_end],
        "dev": ordered[train_end:dev_end],
        "test": ordered[dev_end:],
    }


def select_split(
    task_ids: list[str],
    split: str,
    *,
    config: SplitConfig | None = None,
) -> list[str]:
    if split == "all":
        return list(task_ids)
    partitions = partition_task_ids(task_ids, config=config)
    try:
        return partitions[split]
    except KeyError as exc:
        raise ValueError("split must be one of all, train, dev, test") from exc
