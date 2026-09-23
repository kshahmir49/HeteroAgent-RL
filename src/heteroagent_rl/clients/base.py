from __future__ import annotations

from abc import ABC, abstractmethod
from heteroagent_rl.schema import LLMResult


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        raise NotImplementedError
