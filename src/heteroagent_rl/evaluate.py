from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.benchmarks.evalplus_adapter import load_evalplus_problems
from heteroagent_rl.code_utils import extract_code, parse_verdict
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.clients.ollama import OllamaClient
from heteroagent_rl.metrics import summarize_verifier_calibration
from heteroagent_rl.preflight import preflight_solution
from heteroagent_rl.repair import repair_known_preflight_issues
from heteroagent_rl.sandbox.runner import run_evalplus_sandbox
from heteroagent_rl.workflow.fixed import FixedWorkflow


DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HeteroAgent-RL on EvalPlus coding benchmarks."
    )
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--backend",
        choices=["ollama", "mock", "openai"],
        default="ollama",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--planner-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--executor-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--verifier-model", default=DEFAULT_LOCAL_MODEL)
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
        "--run-tests",
        action="store_true",
        help="Run EvalPlus correctness tests inside an isolated container backend.",
    )
    parser.add_argument(
        "--sandbox-backend",
        choices=["auto", "apple", "docker"],
        default="auto",
        help="Isolation backend used only with --run-tests.",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=float,
        default=300.0,
        help="Maximum wall-clock seconds for the whole sandbox evaluation.",
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    tasks, raw_problems = load_evalplus_problems(
        args.benchmark,
        limit=args.limit,
    )
    client = build_client(args)
    workflow = FixedWorkflow(
        planner=build_planner(client, args.planner_model),
        executor=build_executor(client, args.executor_model),
        verifier=build_verifier(client, args.verifier_model),
    )

    output_dir = Path(args.output_dir) / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    trajectories: list[dict] = []
    solutions: dict[str, str] = {}

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.task_id}")
        result = workflow.run(task.prompt)
        raw_solution = extract_code(result.final_answer)
        verifier_verdict = parse_verdict(result.steps[-1].response)
        repair = repair_known_preflight_issues(raw_solution)
        solution = repair.repaired_code
        preflight = repair.repaired_preflight or preflight_solution(solution)

        solutions[task.task_id] = solution
        samples.append(
            {
                "task_id": task.task_id,
                "solution": solution,
            }
        )
        trajectories.append(
            {
                "task_id": task.task_id,
                "entry_point": task.entry_point,
                "verifier_verdict": verifier_verdict,
                "raw_model_solution": raw_solution,
                "system_solution": solution,
                "repair": repair.to_dict(),
                "preflight": preflight.to_dict(),
                **result.to_dict(),
            }
        )

    sample_path = output_dir / "samples.jsonl"
    trajectory_path = output_dir / "trajectories.jsonl"
    summary_path = output_dir / "summary.json"

    write_jsonl(sample_path, samples)
    write_jsonl(trajectory_path, trajectories)

    total_tokens = sum(row["total_tokens"] for row in trajectories)
    total_latency = sum(row["total_latency_s"] for row in trajectories)

    summary: dict = {
        "benchmark": args.benchmark,
        "num_tasks": len(tasks),
        "backend": "mock" if args.mock else args.backend,
        "planner_model": args.planner_model,
        "executor_model": args.executor_model,
        "verifier_model": args.verifier_model,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / len(tasks),
        "total_latency_s": total_latency,
        "avg_latency_s": total_latency / len(tasks),
        "verifier_pass_rate": (
            sum(row["verifier_verdict"] == "PASS" for row in trajectories) / len(tasks)
        ),
        "raw_preflight_pass_rate": (
            sum(
                bool(row["repair"]["raw_preflight"]["ok"])
                for row in trajectories
                if row["repair"]["raw_preflight"] is not None
            )
            / len(tasks)
        ),
        "preflight_pass_rate": (
            sum(row["preflight"]["ok"] for row in trajectories) / len(tasks)
        ),
        "repair_rate": (
            sum(row["repair"]["applied"] for row in trajectories) / len(tasks)
        ),
        "tests_executed": False,
    }

    if args.run_tests:
        sandbox_backend, scored = run_evalplus_sandbox(
            args.benchmark,
            raw_problems,
            solutions,
            backend=args.sandbox_backend,
            timeout_s=args.sandbox_timeout,
        )

        base_pass = 0
        strict_plus_pass = 0

        for row in trajectories:
            result = scored[row["task_id"]]
            base_status = result["base_status"]
            plus_status = result["plus_status"]
            row["base_status"] = base_status
            row["plus_status"] = plus_status

            base_ok = base_status == "pass"
            strict_plus_ok = base_ok and plus_status == "pass"
            base_pass += int(base_ok)
            strict_plus_pass += int(strict_plus_ok)

        verifier_metrics = summarize_verifier_calibration(trajectories)

        write_jsonl(trajectory_path, trajectories)
        summary.update(
            {
                "tests_executed": True,
                "sandbox_backend": sandbox_backend,
                "base_pass_rate": base_pass / len(tasks),
                "plus_pass_rate": strict_plus_pass / len(tasks),
                **verifier_metrics,
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Samples written to {sample_path}")
    print(f"Trajectories written to {trajectory_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
