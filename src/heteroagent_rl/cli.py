from __future__ import annotations

import argparse
import os

from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.clients.ollama import OllamaClient
from heteroagent_rl.workflow.fixed import FixedWorkflow


DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--backend",
        choices=["ollama", "mock", "openai"],
        default="ollama",
        help="LLM backend. Ollama is the free local default.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Shortcut for --backend mock.",
    )
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
    return parser.parse_args()


def build_client(args: argparse.Namespace):
    backend = "mock" if args.mock else args.backend

    if backend == "mock":
        return MockLLMClient()

    if backend == "ollama":
        return OllamaClient(
            base_url=args.ollama_url,
            num_ctx=args.num_ctx,
        )

    from heteroagent_rl.clients.openai_compatible import OpenAICompatibleClient

    api_key = os.environ.get("HETEROAGENT_API_KEY")
    base_url = os.environ.get("HETEROAGENT_BASE_URL")

    if not api_key:
        raise SystemExit("Set HETEROAGENT_API_KEY when using --backend openai.")

    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
    )


def main() -> None:
    args = parse_args()
    client = build_client(args)

    workflow = FixedWorkflow(
        planner=build_planner(client, args.planner_model),
        executor=build_executor(client, args.executor_model),
        verifier=build_verifier(client, args.verifier_model),
    )

    result = workflow.run(args.task)
    print(result.to_json())


if __name__ == "__main__":
    main()
