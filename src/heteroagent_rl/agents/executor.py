from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


EXECUTOR_SYSTEM_PROMPT = """You are the Executor in a multi-agent problem-solving system.
Solve the original task using the Planner's guidance.

Rules
- Produce the candidate solution, not another plan.
- For coding tasks, return only the complete executable code in one code block.
- The code will be executed in a fresh Python module.
- Preserve the required function name and signature from the original task.
- Include every import required by the function signature, annotations, and implementation.
- Do not assume imports from the benchmark prompt will be preserved automatically.
- Do not rewrite benchmark examples or invent new expected outputs.
- You may omit the task docstring. If you include it, preserve its examples verbatim.
- Do not repeat the Planner's reasoning.
- Prefer the simplest correct implementation.
- Do not add explanations unless the task explicitly asks for them."""


def build_executor(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="executor",
        model=model,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        client=client,
    )
