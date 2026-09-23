from __future__ import annotations

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.clients.base import LLMClient


VERIFIER_SYSTEM_PROMPT = """You are the Verifier in a multi-agent problem-solving system.
Critically inspect the candidate answer against the original task. Look for factual errors,
logic errors, missing constraints, edge cases, and unsupported claims. Begin with either
VERDICT: PASS or VERDICT: REVISE, then give concise reasons and specific corrections."""


def build_verifier(client: LLMClient, model: str) -> Agent:
    return Agent(
        role="verifier",
        model=model,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        client=client,
    )
