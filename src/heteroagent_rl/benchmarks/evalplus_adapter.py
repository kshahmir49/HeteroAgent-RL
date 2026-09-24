from __future__ import annotations

import hashlib
from typing import Any

from heteroagent_rl.benchmarks.base import CodingTask


def _require_evalplus():
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError(
            'EvalPlus is not installed. Run pip install -e ".[bench]" first.'
        ) from exc

    return get_human_eval_plus, get_mbpp_plus


def load_evalplus_problems(
    benchmark: str,
    *,
    limit: int | None = None,
) -> tuple[list[CodingTask], dict[str, dict[str, Any]]]:
    get_human_eval_plus, get_mbpp_plus = _require_evalplus()

    if benchmark == "humaneval":
        problems = get_human_eval_plus()
    elif benchmark == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    def task_number(task_id: str) -> int:
        try:
            return int(task_id.split("/")[-1])
        except ValueError:
            return 0

    ordered_ids = sorted(problems, key=task_number)
    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    selected = {task_id: problems[task_id] for task_id in ordered_ids}
    tasks = [
        CodingTask(
            task_id=task_id,
            prompt=selected[task_id]["prompt"],
            entry_point=selected[task_id]["entry_point"],
        )
        for task_id in ordered_ids
    ]
    return tasks, selected


def subset_hash(benchmark: str, task_ids: list[str]) -> str:
    raw = benchmark + "|" + "|".join(task_ids)
    return "heteroagent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def score_evalplus_subset(
    benchmark: str,
    problems: dict[str, dict[str, Any]],
    solutions: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Score a selected subset with EvalPlus internals.

    EvalPlus recommends sandboxed execution for untrusted model-generated code.
    Callers should require explicit user opt-in before invoking this function.
    """
    try:
        from evalplus.evaluate import check_correctness, get_groundtruth
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    except ImportError as exc:
        raise RuntimeError(
            'EvalPlus is not installed. Run pip install -e ".[bench]" first.'
        ) from exc

    task_ids = list(problems)
    hashcode = subset_hash(benchmark, task_ids)
    output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS if benchmark == "mbpp" else []
    expected = get_groundtruth(problems, hashcode, output_not_none)

    scored: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        result = check_correctness(
            benchmark,
            0,
            problems[task_id],
            solutions[task_id],
            expected[task_id],
            base_only=False,
            fast_check=True,
            identifier=f"{task_id}:0",
        )
        scored[task_id] = result

    return scored
