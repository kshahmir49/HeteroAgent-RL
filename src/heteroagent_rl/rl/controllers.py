from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.env import HeteroAgentEnv, StepTransition


class Controller(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def select_action(
        self,
        observation: dict[str, float | int | bool],
        action_mask: list[int],
    ) -> ControllerAction:
        ...


@dataclass
class ExecutorOnlyController:
    name: str = "executor_only"

    def reset(self) -> None:
        pass

    def select_action(self, observation, action_mask) -> ControllerAction:
        if not observation["has_candidate"]:
            return ControllerAction.CALL_EXECUTOR
        return ControllerAction.STOP


@dataclass
class PlannerExecutorController:
    name: str = "planner_executor"

    def reset(self) -> None:
        pass

    def select_action(self, observation, action_mask) -> ControllerAction:
        if not observation["has_plan"]:
            return ControllerAction.CALL_PLANNER
        if not observation["has_candidate"]:
            return ControllerAction.CALL_EXECUTOR
        return ControllerAction.STOP


@dataclass
class PublicRepairController:
    name: str = "public_repair"
    _repair_attempted: bool = False

    def reset(self) -> None:
        self._repair_attempted = False

    def select_action(self, observation, action_mask) -> ControllerAction:
        if not observation["has_plan"]:
            return ControllerAction.CALL_PLANNER
        if not observation["has_candidate"]:
            return ControllerAction.CALL_EXECUTOR
        if not observation["public_examples_available"]:
            return ControllerAction.RUN_PUBLIC_TESTS
        if observation["public_examples_fail"] and not self._repair_attempted:
            self._repair_attempted = True
            return ControllerAction.CALL_REPAIR
        return ControllerAction.STOP


@dataclass
class FullPipelineController:
    name: str = "full_pipeline"
    _repair_attempted: bool = False

    def reset(self) -> None:
        self._repair_attempted = False

    def select_action(self, observation, action_mask) -> ControllerAction:
        if not observation["has_plan"]:
            return ControllerAction.CALL_PLANNER
        if not observation["has_candidate"]:
            return ControllerAction.CALL_EXECUTOR
        if not observation["public_examples_available"]:
            return ControllerAction.RUN_PUBLIC_TESTS
        if observation["public_examples_fail"] and not self._repair_attempted:
            self._repair_attempted = True
            return ControllerAction.CALL_REPAIR
        if not observation["has_verifier_feedback"]:
            return ControllerAction.CALL_VERIFIER
        return ControllerAction.STOP


CONTROLLERS = {
    "executor_only": ExecutorOnlyController,
    "planner_executor": PlannerExecutorController,
    "public_repair": PublicRepairController,
    "full_pipeline": FullPipelineController,
}


def build_controller(name: str) -> Controller:
    try:
        return CONTROLLERS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(CONTROLLERS))
        raise ValueError(f"Unknown controller '{name}'. Choose from {choices}.") from exc


def run_controller_episode(
    env: HeteroAgentEnv,
    controller: Controller,
    task: str,
) -> dict[str, object]:
    observation = env.reset(task)
    controller.reset()

    transitions: list[StepTransition] = []
    actions: list[str] = []
    total_reward = 0.0

    while True:
        action = controller.select_action(observation, env.action_mask())
        transition = env.step(action)
        transitions.append(transition)
        actions.append(action.name)
        total_reward += transition.reward
        observation = transition.observation
        if transition.terminated:
            break

    state = env.state
    if state is None:
        raise RuntimeError("Environment terminated without state.")

    return {
        "controller": controller.name,
        "actions": actions,
        "environment_reward": total_reward,
        "candidate": state.candidate or "",
        "llm_calls": state.llm_calls,
        "tool_calls": state.tool_calls,
        "total_tokens": state.total_tokens,
        "total_latency_s": state.total_latency_s,
        "total_tool_latency_s": state.total_tool_latency_s,
        "mechanical_repairs": state.mechanical_repairs,
        "verifier_verdict": state.verifier_verdict,
        "public_examples_status": (
            state.public_feedback.status if state.public_feedback is not None else None
        ),
        "steps": [
            {
                "reward": transition.reward,
                "terminated": transition.terminated,
                "info": transition.info,
                "observation": transition.observation,
            }
            for transition in transitions
        ],
    }
