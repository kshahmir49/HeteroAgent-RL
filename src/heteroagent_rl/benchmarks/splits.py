from __future__ import annotations

import hashlib
from dataclasses import dataclass


HUMANEVAL_LEGACY_MAX_ID = 24


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.65
    dev_fraction: float = 0.175
    seed: str = "heteroagent-rl-clean-v1"

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0.0 <= self.dev_fraction < 1.0:
            raise ValueError("dev_fraction must be between 0 and 1")
        if self.train_fraction + self.dev_fraction >= 1.0:
            raise ValueError("train_fraction + dev_fraction must be less than 1")


def _humaneval_index(task_id: str) -> int | None:
    if not task_id.startswith("HumanEval/"):
        return None
    try:
        return int(task_id.split("/", 1)[1])
    except ValueError:
        return None


def legacy_task_ids(task_ids: list[str], *, benchmark: str | None = None) -> list[str]:
    if benchmark != "humaneval":
        return []
    output: list[str] = []
    for task_id in task_ids:
        index = _humaneval_index(task_id)
        if index is not None and index <= HUMANEVAL_LEGACY_MAX_ID:
            output.append(task_id)
    return output


def clean_task_ids(task_ids: list[str], *, benchmark: str | None = None) -> list[str]:
    legacy = set(legacy_task_ids(task_ids, benchmark=benchmark))
    return [task_id for task_id in task_ids if task_id not in legacy]


def partition_task_ids(
    task_ids: list[str],
    *,
    config: SplitConfig | None = None,
) -> dict[str, list[str]]:
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
    benchmark: str | None = None,
    config: SplitConfig | None = None,
) -> list[str]:
    if split == "all":
        return list(task_ids)
    if split == "legacy_dev":
        return legacy_task_ids(task_ids, benchmark=benchmark)

    partitions = partition_task_ids(
        clean_task_ids(task_ids, benchmark=benchmark),
        config=config,
    )
    try:
        return partitions[split]
    except KeyError as exc:
        raise ValueError(
            "split must be one of all, legacy_dev, train, dev, test"
        ) from exc
