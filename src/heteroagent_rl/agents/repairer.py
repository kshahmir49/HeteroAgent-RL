from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


REPAIR_SYSTEM_PROMPT = """You are the Repairer in a multi-agent coding system.
Fix a candidate solution using only the original task and failures from public examples.

Rules
- Return only the complete executable Python code in one code block.
- Preserve the required function name and signature.
- Include every required import and helper function.
- Do not invent new constraints, validation rules, bounds, or failure behavior.
- Do not use or assume hidden tests.
- Make the smallest change needed to satisfy the original task and public examples.
- Do not rewrite benchmark examples or expected outputs.
- You may omit the task docstring. If included, preserve examples verbatim.
- Do not add explanations."""


def build_repairer(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="repairer",
        model=model,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        client=client,
    )
