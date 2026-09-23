from __future__ import annotations

import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from heteroagent_rl.clients.base import LLMClient
from heteroagent_rl.schema import LLMResult, LLMUsage


class OllamaClient(LLMClient):
    """Minimal client for a local Ollama server.

    The implementation deliberately uses only the Python standard library so the
    default local backend adds no extra runtime dependency.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        keep_alive: str = "5m",
        timeout_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }

        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start Ollama and make sure the local "
                f"server is reachable at {self.base_url}."
            ) from exc

        latency = time.perf_counter() - start
        message = body.get("message") or {}

        return LLMResult(
            text=message.get("content", ""),
            model=body.get("model", model),
            latency_s=latency,
            usage=LLMUsage(
                input_tokens=int(body.get("prompt_eval_count", 0) or 0),
                output_tokens=int(body.get("eval_count", 0) or 0),
            ),
            raw=body,
        )
