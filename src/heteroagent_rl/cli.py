from __future__ import annotations

import argparse
import os

from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.workflow.fixed import FixedWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--planner-model", default="planner-model")
    parser.add_argument("--executor-model", default="executor-model")
    parser.add_argument("--verifier-model", default="verifier-model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mock:
        client = MockLLMClient()
    else:
        from heteroagent_rl.clients.openai_compatible import OpenAICompatibleClient

        api_key = os.environ.get("HETEROAGENT_API_KEY")
        base_url = os.environ.get("HETEROAGENT_BASE_URL")

        if not api_key:
            raise SystemExit("Set HETEROAGENT_API_KEY or run with --mock.")

        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
        )

    workflow = FixedWorkflow(
        planner=build_planner(client, args.planner_model),
        executor=build_executor(client, args.executor_model),
        verifier=build_verifier(client, args.verifier_model),
    )

    result = workflow.run(args.task)
    print(result.to_json())


if __name__ == "__main__":
    main()
