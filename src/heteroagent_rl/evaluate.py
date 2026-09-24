from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.repairer import build_repairer
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.benchmarks.evalplus_adapter import load_evalplus_problems
from heteroagent_rl.code_utils import extract_code, parse_verdict
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.clients.ollama import OllamaClient
from heteroagent_rl.metrics import summarize_verifier_calibration
from heteroagent_rl.preflight import preflight_solution
from heteroagent_rl.repair import repair_known_preflight_issues
from heteroagent_rl.sandbox.runner import (
    run_evalplus_sandbox,
    run_public_examples_sandbox,
)
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
        "--repair-model",
        default=None,
        help="Model for one-shot public-example repair. Defaults to executor model.",
    )
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
        "--public-repair",
        action="store_true",
        help=(
            "Run only prompt-visible examples in the sandbox and make at most one "
            "LLM repair attempt when those examples fail."
        ),
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run held-out EvalPlus correctness tests inside an isolated container.",
    )
    parser.add_argument(
        "--sandbox-backend",
        choices=["auto", "apple", "docker"],
        default="auto",
        help="Isolation backend used for public examples and held-out tests.",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=float,
        default=300.0,
        help="Maximum wall-clock seconds for each sandbox batch.",
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


def _public_repair_prompt(task: str, candidate: str, report: str) -> str:
    return f"""Original task

{task}

Candidate solution

```python
{candidate}
```

Public example failure report

{report or "One or more public examples failed."}

Repair the candidate using only the original task and the public example failure report."""


def apply_public_example_repairs(
    *,
    args: argparse.Namespace,
    client,
    raw_problems: dict,
    solutions: dict[str, str],
    trajectories: list[dict],
) -> tuple[str, dict[str, str]]:
    sandbox_backend, before = run_public_examples_sandbox(
        raw_problems,
        solutions,
        backend=args.sandbox_backend,
        timeout_s=args.sandbox_timeout,
    )

    repair_model = args.repair_model or args.executor_model
    repairer = build_repairer(client, repair_model)
    task_to_row = {row["task_id"]: row for row in trajectories}
    repaired_ids: list[str] = []

    for task_id, public_result in before.items():
        row = task_to_row[task_id]
        row["public_examples_before"] = public_result
        row["public_repair_applied"] = False

        if public_result["status"] != "fail":
            row["public_examples_after"] = public_result
            continue

        repair_step = repairer.run(
            _public_repair_prompt(
                row["task"],
                solutions[task_id],
                public_result.get("report", ""),
            )
        )
        repaired_candidate = extract_code(repair_step.response)
        mechanical = repair_known_preflight_issues(repaired_candidate)
        final_solution = mechanical.repaired_code

        row["public_repair_applied"] = True
        row["public_repair_step"] = asdict(repair_step)
        row["public_repair_solution"] = repaired_candidate
        row["post_llm_repair_mechanical_repair"] = mechanical.to_dict()
        row["system_solution"] = final_solution
        row["total_tokens"] += repair_step.input_tokens + repair_step.output_tokens
        row["total_latency_s"] += repair_step.latency_s

        solutions[task_id] = final_solution
        repaired_ids.append(task_id)

    if repaired_ids:
        repaired_problems = {task_id: raw_problems[task_id] for task_id in repaired_ids}
        repaired_solutions = {task_id: solutions[task_id] for task_id in repaired_ids}
        _, after = run_public_examples_sandbox(
            repaired_problems,
            repaired_solutions,
            backend=sandbox_backend,
            timeout_s=args.sandbox_timeout,
        )
        for task_id in repaired_ids:
            task_to_row[task_id]["public_examples_after"] = after[task_id]

    return sandbox_backend, solutions


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

    public_sandbox_backend = None
    if args.public_repair:
        public_sandbox_backend, solutions = apply_public_example_repairs(
            args=args,
            client=client,
            raw_problems=raw_problems,
            solutions=solutions,
            trajectories=trajectories,
        )

    samples = [
        {"task_id": task.task_id, "solution": solutions[task.task_id]}
        for task in tasks
    ]

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
        "repair_model": args.repair_model or args.executor_model,
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
        "public_repair_enabled": args.public_repair,
        "tests_executed": False,
    }

    if args.public_repair:
        public_rows = [
            row
            for row in trajectories
            if row["public_examples_before"]["status"] != "no_examples"
        ]
        repaired_rows = [row for row in trajectories if row["public_repair_applied"]]
        summary.update(
            {
                "public_sandbox_backend": public_sandbox_backend,
                "public_example_tasks": len(public_rows),
                "public_examples_initial_pass_rate": (
                    sum(row["public_examples_before"]["status"] == "pass" for row in public_rows)
                    / len(public_rows)
                    if public_rows
                    else 0.0
                ),
                "llm_repair_rate": len(repaired_rows) / len(tasks),
                "llm_repair_success_rate": (
                    sum(row["public_examples_after"]["status"] == "pass" for row in repaired_rows)
                    / len(repaired_rows)
                    if repaired_rows
                    else 0.0
                ),
                "public_examples_final_pass_rate": (
                    sum(row["public_examples_after"]["status"] == "pass" for row in public_rows)
                    / len(public_rows)
                    if public_rows
                    else 0.0
                ),
            }
        )

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

        calibration_rows = [
            row for row in trajectories if not row.get("public_repair_applied", False)
        ]
        verifier_metrics = summarize_verifier_calibration(calibration_rows)

        write_jsonl(trajectory_path, trajectories)
        summary.update(
            {
                "tests_executed": True,
                "sandbox_backend": sandbox_backend,
                "base_pass_rate": base_pass / len(tasks),
                "plus_pass_rate": strict_plus_pass / len(tasks),
                "verifier_calibration_tasks": len(calibration_rows),
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
