from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, str]:
    samples: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples[row["task_id"]] = row["solution"]
    return samples


def subset_hash(benchmark: str, task_ids: list[str]) -> str:
    raw = benchmark + "|" + "|".join(task_ids)
    return "heteroagent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    args = parse_args()

    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    with Path(args.problems).open("rb") as handle:
        problems = pickle.load(handle)

    samples = load_samples(Path(args.samples))
    task_ids = list(problems)

    missing = [task_id for task_id in task_ids if task_id not in samples]
    if missing:
        raise RuntimeError(f"Missing generated solutions for tasks: {missing}")

    output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS if args.benchmark == "mbpp" else []
    expected = get_groundtruth(
        problems,
        subset_hash(args.benchmark, task_ids),
        output_not_none,
    )

    scored: dict[str, dict[str, str]] = {}
    for task_id in task_ids:
        result = check_correctness(
            args.benchmark,
            0,
            problems[task_id],
            samples[task_id],
            expected[task_id],
            base_only=False,
            fast_check=True,
            identifier=f"{task_id}:0",
        )
        scored[task_id] = {
            "base_status": result["base"][0],
            "plus_status": result["plus"][0],
        }

    Path(args.output).write_text(
        json.dumps(scored, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
