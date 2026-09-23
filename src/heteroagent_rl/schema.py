from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict[str, Any] | None = None


@dataclass
class StepRecord:
    role: str
    prompt: str
    response: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class WorkflowResult:
    task: str
    final_answer: str
    steps: list[StepRecord]

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def total_latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "total_latency_s": self.total_latency_s,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
