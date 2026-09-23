from heteroagent_rl.agents.executor import build_executor
from heteroagent_rl.agents.planner import build_planner
from heteroagent_rl.agents.verifier import build_verifier
from heteroagent_rl.clients.mock import MockLLMClient
from heteroagent_rl.workflow.fixed import FixedWorkflow


def test_fixed_workflow_records_three_steps():
    client = MockLLMClient()

    workflow = FixedWorkflow(
        planner=build_planner(client, "mock-planner"),
        executor=build_executor(client, "mock-executor"),
        verifier=build_verifier(client, "mock-verifier"),
    )

    result = workflow.run(
        "Write a Python function that returns the nth Fibonacci number."
    )

    assert [step.role for step in result.steps] == [
        "planner",
        "executor",
        "verifier",
    ]
    assert result.total_tokens == 150
    assert "def fibonacci" in result.final_answer
    assert result.steps[-1].response.startswith("VERDICT: PASS")
