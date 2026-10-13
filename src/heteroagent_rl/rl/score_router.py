from __future__ import annotations

import argparse
import json
from pathlib import Path

from heteroagent_rl.rl.router import (
    EXECUTOR_LABEL,
    RouterPolicy,
    load_jsonl,
    pair_trajectory_rows,
    task_feature_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a contextual router against paired cached trajectories."
    )
    parser.add_argument(
        "--policy",
        default="runs/router/humaneval/contextual_tree.json",
    )
    parser.add_argument(
        "--executor-trajectories",
        default="runs/controllers/humaneval/dev/executor_only/trajectories.jsonl",
    )
    parser.add_argument(
        "--specialist-trajectories",
        default="runs/controllers/humaneval/dev/specialist_only/trajectories.jsonl",
    )
    parser.add_argument(
        "--output",
        default="runs/router/humaneval/dev_score.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = RouterPolicy.load(args.policy)
    executor_rows = load_jsonl(args.executor_trajectories)
    specialist_rows = load_jsonl(args.specialist_trajectories)
    examples = pair_trajectory_rows(executor_rows, specialist_rows)

    routed_success = 0
    routed_base = 0
    routed_reward = 0.0
    executor_routes = 0
    specialist_routes = 0
    oracle_success = 0

    decisions: list[dict[str, object]] = []
    executor_by_id = {str(row["task_id"]): row for row in executor_rows}
    specialist_by_id = {str(row["task_id"]): row for row in specialist_rows}

    for row in examples:
        task_id = str(row["task_id"])
        raw_features = executor_by_id[task_id]["task_features"]
        observation = task_feature_observation(raw_features)
        label = policy.predict_label(observation)

        if label == EXECUTOR_LABEL:
            chosen = executor_by_id[task_id]
            route = "executor"
            executor_routes += 1
        else:
            chosen = specialist_by_id[task_id]
            route = "specialist"
            specialist_routes += 1

        success = bool(chosen["strict_success"])
        base_ok = chosen.get("base_status") == "pass"
        reward = float(chosen["policy_reward"])
        oracle = bool(
            executor_by_id[task_id]["strict_success"]
            or specialist_by_id[task_id]["strict_success"]
        )

        routed_success += int(success)
        routed_base += int(base_ok)
        routed_reward += reward
        oracle_success += int(oracle)
        decisions.append(
            {
                "task_id": task_id,
                "route": route,
                "strict_success": success,
                "base_pass": base_ok,
                "policy_reward": reward,
            }
        )

    n = len(examples)
    report = {
        "num_tasks": n,
        "executor_routes": executor_routes,
        "specialist_routes": specialist_routes,
        "base_pass_rate": routed_base / n,
        "strict_pass_rate": routed_success / n,
        "avg_policy_reward": routed_reward / n,
        "executor_strict_rate": sum(
            bool(row["strict_success"]) for row in executor_rows
        ) / n,
        "specialist_strict_rate": sum(
            bool(row["strict_success"]) for row in specialist_rows
        ) / n,
        "oracle_strict_rate": oracle_success / n,
        "decisions": decisions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
