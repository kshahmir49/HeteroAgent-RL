from __future__ import annotations

import argparse
import json
from pathlib import Path

from heteroagent_rl.rl.router import (
    load_jsonl,
    pair_trajectory_rows,
    save_router_policy,
    train_router_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a contextual Executor-vs-Specialist router from cached trajectories."
    )
    parser.add_argument(
        "--executor-trajectories",
        default="runs/controllers/humaneval/train/executor_only/trajectories.jsonl",
    )
    parser.add_argument(
        "--specialist-trajectories",
        default="runs/controllers/humaneval/train/specialist_only/trajectories.jsonl",
    )
    parser.add_argument(
        "--output",
        default="runs/router/humaneval/contextual_tree.json",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executor_rows = load_jsonl(args.executor_trajectories)
    specialist_rows = load_jsonl(args.specialist_trajectories)
    examples = pair_trajectory_rows(executor_rows, specialist_rows)
    payload, metadata = train_router_policy(examples, seed=args.seed)
    save_router_policy(payload, args.output)

    report = {
        "policy_path": str(Path(args.output)),
        **metadata,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
