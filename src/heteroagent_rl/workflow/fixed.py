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

Produce a plan for the Executor."""
        plan = self.planner.run(planner_prompt)
        steps.append(plan)

        executor_prompt = f"""Original task

{task}

Planner output

{plan.response}

Produce the candidate solution."""
        candidate = self.executor.run(executor_prompt)
        steps.append(candidate)

        verifier_prompt = f"""Original task

{task}

Planner output

{plan.response}

Candidate solution

{candidate.response}

Verify the candidate."""
        verification = self.verifier.run(verifier_prompt)
        steps.append(verification)

        return WorkflowResult(
            task=task,
            final_answer=candidate.response,
            steps=steps,
        )
