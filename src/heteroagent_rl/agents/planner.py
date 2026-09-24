from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


PLANNER_SYSTEM_PROMPT = """You are the Planner in a multi-agent problem-solving system.
Create a short actionable plan for another agent.

Rules
- Do not write the final answer.
- For coding tasks, do not write code.
- Use at most 6 bullets.
- Keep the response under 180 words.
- Include only constraints, algorithmic choices, edge cases, and likely failure modes.
- Do not restate the task unless needed for clarity."""


def build_planner(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="planner",
        model=model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        client=client,
    )
