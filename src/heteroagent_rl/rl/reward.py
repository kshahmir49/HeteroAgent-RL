from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    success_reward: float = 1.0
    failure_reward: float = 0.0
    call_penalty: float = 0.01
    token_penalty_per_1k: float = 0.001
    latency_penalty_per_s: float = 0.001
    tool_call_penalty: float = 0.005
    tool_latency_penalty_per_s: float = 0.001
    invalid_action_penalty: float = 0.05


def llm_step_cost(*, tokens: int, latency_s: float, config: RewardConfig) -> float:
    return (
        config.call_penalty
        + config.token_penalty_per_1k * (tokens / 1000.0)
        + config.latency_penalty_per_s * latency_s
    )


def tool_step_cost(*, latency_s: float, config: RewardConfig) -> float:
    return (
        config.tool_call_penalty
        + config.tool_latency_penalty_per_s * latency_s
    )


def terminal_reward(*, success: float, config: RewardConfig) -> float:
    clipped = min(1.0, max(0.0, float(success)))
    return config.failure_reward + clipped * (
        config.success_reward - config.failure_reward
    )
