from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.repairer import build_repairer
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.benchmarks.evalplus_adapter import load_evalplus_problems
from heteroagent_rl.benchmarks.splits import SplitConfig, select_split
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.clients.ollama import OllamaClient
from heteroagent_rl.rl.controllers import CONTROLLERS, build_controller, run_controller_episode
from heteroagent_rl.rl.env import CandidateFeedback, HeteroAgentEnv
from heteroagent_rl.rl.reward import RewardConfig, terminal_reward
from heteroagent_rl.sandbox.runner import (
    run_evalplus_sandbox,
    run_public_examples_sandbox,
)


DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic controller baselines through the RL environment."
    )
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="all")
    parser.add_argument("--split-seed", default="heteroagent-rl-v1")
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="full_pipeline")
    parser.add_argument("--output-dir", default="runs/controllers")
    parser.add_argument("--backend", choices=["ollama", "mock", "openai"], default="ollama")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--planner-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--executor-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--verifier-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--repair-model", default=None)
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("HETEROAGENT_OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(os.environ.get("HETEROAGENT_NUM_CTX", "4096")),
    )
    parser.add_argument(
        "--sandbox-backend",
        choices=["auto", "apple", "docker"],
        default="auto",
    )
    parser.add_argument("--sandbox-timeout", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--call-penalty", type=float, default=0.01)
    parser.add_argument("--token-penalty-per-1k", type=float, default=0.001)
    parser.add_argument("--latency-penalty-per-s", type=float, default=0.001)
    parser.add_argument("--tool-call-penalty", type=float, default=0.005)
    parser.add_argument("--tool-latency-penalty-per-s", type=float, default=0.001)
    return parser.parse_args()


def build_client(args: argparse.Namespace):
    backend = "mock" if args.mock else args.backend
    if backend == "mock":
        return MockLLMClient()
    if backend == "ollama":
        return OllamaClient(base_url=args.ollama_url, num_ctx=args.num_ctx)

    from heteroagent_rl.clients.openai_compatible import OpenAICompatibleClient

    api_key = os.environ.get("HETEROAGENT_API_KEY")
    base_url = os.environ.get("HETEROAGENT_BASE_URL")
    if not api_key:
        raise SystemExit("Set HETEROAGENT_API_KEY when using --backend openai.")
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")

    all_tasks, all_problems = load_evalplus_problems(args.benchmark, limit=None)
    split_config = SplitConfig(seed=args.split_seed)
    selected_ids = select_split(
        [task.task_id for task in all_tasks],
        args.split,
        config=split_config,
    )
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    selected_set = set(selected_ids)
    tasks = [task for task in all_tasks if task.task_id in selected_set]
    tasks.sort(key=lambda task: selected_ids.index(task.task_id))
    raw_problems = {task_id: all_problems[task_id] for task_id in selected_ids}
    client = build_client(args)
    planner = build_planner(client, args.planner_model)
    executor = build_executor(client, args.executor_model)
    verifier = build_verifier(client, args.verifier_model)
    repairer = build_repairer(client, args.repair_model or args.executor_model)
    reward_config = RewardConfig(
        call_penalty=args.call_penalty,
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
        tool_call_penalty=args.tool_call_penalty,
        tool_latency_penalty_per_s=args.tool_latency_penalty_per_s,
    )

    rows: list[dict] = []
    solutions: dict[str, str] = {}

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.task_id}")
        problem = raw_problems[task.task_id]

        def public_feedback(_task: str, candidate: str, *, _problem=problem, _task_id=task.task_id):
            _, scored = run_public_examples_sandbox(
                {_task_id: _problem},
                {_task_id: candidate},
                backend=args.sandbox_backend,
                timeout_s=args.sandbox_timeout,
            )
            result = scored[_task_id]
            return CandidateFeedback(
                status=result["status"],
                report=result.get("report", ""),
                attempted=int(result.get("attempted", 0)),
                failed=int(result.get("failed", 0)),
            )

        env = HeteroAgentEnv(
            planner=planner,
            executor=executor,
            verifier=verifier,
            repairer=repairer,
            public_feedback_fn=public_feedback,
            terminal_score_fn=None,
            reward_config=reward_config,
            max_steps=args.max_steps,
        )
        controller = build_controller(args.controller)
        episode = run_controller_episode(env, controller, task.prompt)
        episode["task_id"] = task.task_id
        episode["entry_point"] = task.entry_point
        rows.append(episode)
        solutions[task.task_id] = str(episode["candidate"])

    sandbox_backend, scored = run_evalplus_sandbox(
        args.benchmark,
        raw_problems,
        solutions,
        backend=args.sandbox_backend,
        timeout_s=args.sandbox_timeout,
    )

    base_pass = 0
    plus_pass = 0
    total_policy_reward = 0.0

    for row in rows:
        result = scored[row["task_id"]]
        base_ok = result["base_status"] == "pass"
        plus_ok = base_ok and result["plus_status"] == "pass"
        row["base_status"] = result["base_status"]
        row["plus_status"] = result["plus_status"]
        row["strict_success"] = plus_ok
        terminal = terminal_reward(
            success=float(plus_ok),
            config=reward_config,
        )
        row["terminal_reward"] = terminal
        row["policy_reward"] = float(row["environment_reward"]) + terminal
        total_policy_reward += row["policy_reward"]
        base_pass += int(base_ok)
        plus_pass += int(plus_ok)

    output_dir = Path(args.output_dir) / args.benchmark / args.controller
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(trajectory_path, rows)

    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    total_latency = sum(float(row["total_latency_s"]) for row in rows)
    total_calls = sum(int(row["llm_calls"]) for row in rows)
    total_tool_calls = sum(int(row["tool_calls"]) for row in rows)
    total_tool_latency = sum(float(row["total_tool_latency_s"]) for row in rows)

    summary = {
        "benchmark": args.benchmark,
        "controller": args.controller,
        "num_tasks": len(tasks),
        "split": args.split,
        "split_seed": args.split_seed,
        "backend": "mock" if args.mock else args.backend,
        "planner_model": args.planner_model,
        "executor_model": args.executor_model,
        "verifier_model": args.verifier_model,
        "repair_model": args.repair_model or args.executor_model,
        "sandbox_backend": sandbox_backend,
        "base_pass_rate": base_pass / len(tasks),
        "plus_pass_rate": plus_pass / len(tasks),
        "total_llm_calls": total_calls,
        "avg_llm_calls": total_calls / len(tasks),
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": total_tool_calls / len(tasks),
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / len(tasks),
        "total_latency_s": total_latency,
        "avg_latency_s": total_latency / len(tasks),
        "total_tool_latency_s": total_tool_latency,
        "avg_tool_latency_s": total_tool_latency / len(tasks),
        "total_policy_reward": total_policy_reward,
        "avg_policy_reward": total_policy_reward / len(tasks),
        "reward_config": {
            "success_reward": reward_config.success_reward,
            "failure_reward": reward_config.failure_reward,
            "call_penalty": reward_config.call_penalty,
            "token_penalty_per_1k": reward_config.token_penalty_per_1k,
            "latency_penalty_per_s": reward_config.latency_penalty_per_s,
            "tool_call_penalty": reward_config.tool_call_penalty,
            "tool_latency_penalty_per_s": reward_config.tool_latency_penalty_per_s,
            "invalid_action_penalty": reward_config.invalid_action_penalty,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Trajectories written to {trajectory_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
