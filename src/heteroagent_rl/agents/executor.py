from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


EXECUTOR_SYSTEM_PROMPT = """You are the Executor in a multi-agent problem-solving system.
Use the Planner's guidance to solve the task. Produce a concrete candidate answer.
For coding tasks, prefer complete executable code. Be concise and do not merely restate
the plan."""


def build_executor(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="executor",
        model=model,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        client=client,
    )
