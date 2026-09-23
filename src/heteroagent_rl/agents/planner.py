from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


PLANNER_SYSTEM_PROMPT = """You are the Planner in a multi-agent problem-solving system.
Your job is to decompose the task, identify constraints, propose a concise plan, and flag
likely failure modes. Do not write a polished final answer unless planning itself is the task.
Make your plan actionable for another agent."""


def build_planner(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="planner",
        model=model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        client=client,
    )
