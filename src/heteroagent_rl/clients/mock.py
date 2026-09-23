from __future__ import annotations

from heteroagent_rl.clients.base import LLMClient
from heteroagent_rl.schema import LLMResult, LLMUsage


class MockLLMClient(LLMClient):
    """Deterministic fake client used for unit tests and pipeline development."""

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        system = system_prompt.strip().lower()

        if system.startswith("you are the planner"):
            text = (
                "1. Identify the required inputs and outputs.\n"
                "2. Choose the simplest correct algorithm.\n"
                "3. Ask the executor to implement it.\n"
                "4. Verify edge cases and complexity."
            )
        elif system.startswith("you are the executor"):
            text = (
                "Candidate solution\n\n"
                "```python\n"
                "def fibonacci(n: int) -> int:\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
                "```"
            )
        elif system.startswith("you are the verifier"):
            text = (
                "VERDICT: PASS\n"
                "The solution handles n=0, uses O(n) time and O(1) auxiliary space, "
                "and rejects negative inputs."
            )
        else:
            text = "Mock response"

        return LLMResult(
            text=text,
            model=model,
            latency_s=0.001,
            usage=LLMUsage(input_tokens=20, output_tokens=30),
        )
