from __future__ import annotations

import time
from openai import OpenAI

from heteroagent_rl.clients.base import LLMClient
from heteroagent_rl.schema import LLMResult, LLMUsage


class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI or any OpenAI-compatible endpoint such as vLLM."""

    def __init__(self, *, api_key: str, base_url: str | None = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        latency = time.perf_counter() - start
        choice = response.choices[0]
        usage = response.usage

        return LLMResult(
            text=choice.message.content or "",
            model=model,
            latency_s=latency,
            usage=LLMUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            raw=None,
        )
