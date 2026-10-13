from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from heteroagent_rl.agents.executor import build_executor, build_specialist
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.repairer import build_repairer
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.benchmarks.evalplus_adapter import load_evalplus_problems
from heteroagent_rl.benchmarks.splits import SplitConfig, select_split
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.clients.ollama import OllamaClient
from heteroagent_rl.rl.checkpoint import (
    append_jsonl_durable,
    atomic_write_json,
    load_jsonl_by_task,
    prepare_run_directory,
)
from heteroagent_rl.rl.controllers import CONTROLLERS, build_controller, run_controller_episode
from heteroagent_rl.rl.env import CandidateFeedback, HeteroAgentEnv
from heteroagent_rl.rl.reward import RewardConfig, terminal_reward
from heteroagent_rl.sandbox.runner import (
    run_evalplus_sandbox,
    run_public_examples_sandbox,
)


DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"
DEFAULT_SPECIALIST_MODEL = "qwen2.5-coder:7b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic controller baselines through the RL environment."
    )
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["all", "legacy_dev", "train", "dev", "test"], default="all")
    parser.add_argument("--split-seed", default="heteroagent-rl-clean-v1")
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="full_pipeline")
    parser.add_argument("--output-dir", default="runs/controllers")
    parser.add_argument("--backend", choices=["ollama", "mock", "openai"], default="ollama")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--planner-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--executor-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--specialist-model", default=DEFAULT_SPECIALIST_MODEL)
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any checkpoint for this split/controller and restart from task 1.",
    )
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


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def resource_totals(rows: list[dict]) -> dict[str, float | int]:
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    total_latency = sum(float(row["total_latency_s"]) for row in rows)
    total_calls = sum(int(row["llm_calls"]) for row in rows)
    total_tool_calls = sum(int(row["tool_calls"]) for row in rows)
    total_tool_latency = sum(float(row["total_tool_latency_s"]) for row in rows)
    return {
        "total_llm_calls": total_calls,
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "total_latency_s": total_latency,
        "total_tool_latency_s": total_tool_latency,
    }


def progress_summary(
    *,
    args: argparse.Namespace,
    selected_ids: list[str],
    rows: list[dict],
    reward_config: RewardConfig,
) -> dict:
    totals = resource_totals(rows)
    completed = len(rows)
    return {
        "status": "in_progress",
        "benchmark": args.benchmark,
        "controller": args.controller,
        "split": args.split,
        "split_seed": args.split_seed,
        "num_tasks": len(selected_ids),
        "completed_tasks": completed,
        "remaining_tasks": len(selected_ids) - completed,
        "backend": "mock" if args.mock else args.backend,
        "planner_model": args.planner_model,
        "executor_model": args.executor_model,
        "specialist_model": args.specialist_model,
        "verifier_model": args.verifier_model,
        "repair_model": args.repair_model or args.executor_model,
        **totals,
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
        benchmark=args.benchmark,
        config=split_config,
    )
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    selected_set = set(selected_ids)
    tasks = [task for task in all_tasks if task.task_id in selected_set]
    tasks.sort(key=lambda task: selected_ids.index(task.task_id))
    raw_problems = {task_id: all_problems[task_id] for task_id in selected_ids}

    reward_config = RewardConfig(
        call_penalty=args.call_penalty,
        token_penalty_per_1k=args.token_penalty_per_1k,
        latency_penalty_per_s=args.latency_penalty_per_s,
        tool_call_penalty=args.tool_call_penalty,
        tool_latency_penalty_per_s=args.tool_latency_penalty_per_s,
    )

    run_config = {
        "benchmark": args.benchmark,
        "split": args.split,
        "split_seed": args.split_seed,
        "selected_ids": selected_ids,
        "controller": args.controller,
        "backend": "mock" if args.mock else args.backend,
        "planner_model": args.planner_model,
        "executor_model": args.executor_model,
        "specialist_model": args.specialist_model,
        "verifier_model": args.verifier_model,
        "repair_model": args.repair_model or args.executor_model,
        "num_ctx": args.num_ctx,
        "sandbox_backend": args.sandbox_backend,
        "sandbox_timeout": args.sandbox_timeout,
        "max_steps": args.max_steps,
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

    output_dir = Path(args.output_dir) / args.benchmark / args.split / args.controller
    checkpoint_path = prepare_run_directory(
        output_dir,
        run_config=run_config,
        fresh=args.fresh,
    )
    summary_path = output_dir / "summary.json"
    trajectory_path = output_dir / "trajectories.jsonl"

    checkpoint_rows = load_jsonl_by_task(checkpoint_path)
    unknown_checkpoint_ids = set(checkpoint_rows) - selected_set
    if unknown_checkpoint_ids:
        raise RuntimeError(
            "Checkpoint contains task IDs outside the selected split. "
            "Use --fresh to reset this run."
        )

    rows_by_task = {
        task_id: checkpoint_rows[task_id]
        for task_id in selected_ids
        if task_id in checkpoint_rows
    }

    if rows_by_task:
        print(
            f"Resuming {args.controller} on {args.split} with "
            f"{len(rows_by_task)}/{len(selected_ids)} tasks already checkpointed."
        )

    client = build_client(args)
    planner = build_planner(client, args.planner_model)
    executor = build_executor(client, args.executor_model)
    specialist = build_specialist(client, args.specialist_model)
    verifier = build_verifier(client, args.verifier_model)
    repairer = build_repairer(client, args.repair_model or args.executor_model)

    for index, task in enumerate(tasks, start=1):
        if task.task_id in rows_by_task:
            print(f"[{index}/{len(tasks)}] {task.task_id} — checkpointed, skipping")
            continue

        print(f"[{index}/{len(tasks)}] {task.task_id}")
        problem = raw_problems[task.task_id]

        def public_feedback(
            _task: str,
            candidate: str,
            *,
            _problem=problem,
            _task_id=task.task_id,
        ):
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
            specialist=specialist,
            public_feedback_fn=public_feedback,
            terminal_score_fn=None,
            reward_config=reward_config,
            max_steps=args.max_steps,
        )
        controller = build_controller(args.controller)
        episode = run_controller_episode(
            env,
            controller,
            task.prompt,
            entry_point=task.entry_point,
        )
        episode["task_id"] = task.task_id
        episode["entry_point"] = task.entry_point

        append_jsonl_durable(checkpoint_path, episode)
        rows_by_task[task.task_id] = episode

        completed_rows = [
            rows_by_task[task_id]
            for task_id in selected_ids
            if task_id in rows_by_task
        ]
        atomic_write_json(
            summary_path,
            progress_summary(
                args=args,
                selected_ids=selected_ids,
                rows=completed_rows,
                reward_config=reward_config,
            ),
        )

    rows = [rows_by_task[task_id] for task_id in selected_ids]
    solutions = {row["task_id"]: str(row["candidate"]) for row in rows}

    print("All episodes checkpointed. Running terminal EvalPlus scoring.")
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

    write_jsonl_atomic(trajectory_path, rows)

    totals = resource_totals(rows)
    total_calls = int(totals["total_llm_calls"])
    total_tool_calls = int(totals["total_tool_calls"])
    total_tokens = int(totals["total_tokens"])
    total_latency = float(totals["total_latency_s"])
    total_tool_latency = float(totals["total_tool_latency_s"])

    summary = {
        "status": "complete",
        "benchmark": args.benchmark,
        "controller": args.controller,
        "num_tasks": len(tasks),
        "completed_tasks": len(tasks),
        "split": args.split,
        "split_seed": args.split_seed,
        "backend": "mock" if args.mock else args.backend,
        "planner_model": args.planner_model,
        "executor_model": args.executor_model,
        "specialist_model": args.specialist_model,
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
        "reward_config": run_config["reward_config"],
    }
    atomic_write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Checkpoint written to {checkpoint_path}")
    print(f"Trajectories written to {trajectory_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
