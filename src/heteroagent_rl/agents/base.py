from __future__ import annotations

from dataclasses import dataclass

from heteroagent_rl.clients.base import LLMClient
from heteroagent_rl.schema import StepRecord


@dataclass
class Agent:
    role: str
    model: str
    system_prompt: str
    client: LLMClient
    temperature: float = 0.0

    def run(self, prompt: str) -> StepRecord:
        result = self.client.generate(
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            temperature=self.temperature,
        )

        return StepRecord(
            role=self.role,
            prompt=prompt,
            response=result.text,
            model=result.model,
            latency_s=result.latency_s,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
