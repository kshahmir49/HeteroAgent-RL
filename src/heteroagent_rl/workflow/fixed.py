from __future__ import annotations

from dataclasses import dataclass

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.schema import WorkflowResult


@dataclass
class FixedWorkflow:
    planner: Agent
    executor: Agent
    verifier: Agent

    def run(self, task: str) -> WorkflowResult:
        steps = []

        planner_prompt = f"""Original task

{task}

Produce a concise plan for the Executor."""
        plan = self.planner.run(planner_prompt)
        steps.append(plan)

        executor_prompt = f"""Original task

{task}

Planner guidance

{plan.response}

Produce the candidate solution."""
        candidate = self.executor.run(executor_prompt)
        steps.append(candidate)

        # The verifier intentionally does not receive the planner output.
        # This reduces duplicated context and makes verification more independent.
        verifier_prompt = f"""Original task

{task}

Candidate solution

{candidate.response}

Verify the candidate independently."""
        verification = self.verifier.run(verifier_prompt)
        steps.append(verification)

        return WorkflowResult(
            task=task,
            final_answer=candidate.response,
            steps=steps,
        )
