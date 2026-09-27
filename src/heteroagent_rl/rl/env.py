from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from heteroagent_rl.agents.base import Agent
from heteroagent_rl.code_utils import extract_code, parse_verdict
from heteroagent_rl.preflight import preflight_solution
from heteroagent_rl.repair import repair_known_preflight_issues
from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.reward import RewardConfig, llm_step_cost, terminal_reward
from heteroagent_rl.schema import StepRecord


@dataclass(frozen=True)
class CandidateFeedback:
    status: str
    report: str = ""
    attempted: int = 0
    failed: int = 0

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class RLState:
    task: str
    plan: str | None = None
    candidate: str | None = None
    verifier_feedback: str | None = None
    verifier_verdict: str = "UNKNOWN"
    public_feedback: CandidateFeedback | None = None
    preflight_ok: bool | None = None
    mechanical_repairs: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    step_count: int = 0
    done: bool = False
    terminal_success: float | None = None
    history: list[StepRecord] = field(default_factory=list)


@dataclass(frozen=True)
class StepTransition:
    observation: dict[str, float | int | bool]
    reward: float
    terminated: bool
    info: dict[str, object]


FeedbackFn = Callable[[str, str], CandidateFeedback]
TerminalScoreFn = Callable[[str, str], float | bool]


class HeteroAgentEnv:
    """Framework-independent RL environment for agent orchestration.

    Public feedback may inspect prompt-visible examples. The terminal scorer is
    called only when an episode stops and should be the only component with
    access to held-out evaluation.
    """

    def __init__(
        self,
        *,
        planner: Agent,
        executor: Agent,
        verifier: Agent,
        repairer: Agent,
        public_feedback_fn: FeedbackFn | None = None,
        terminal_score_fn: TerminalScoreFn | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.repairer = repairer
        self.public_feedback_fn = public_feedback_fn
        self.terminal_score_fn = terminal_score_fn
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.state: RLState | None = None

    @property
    def action_size(self) -> int:
        return len(ControllerAction)

    def reset(self, task: str) -> dict[str, float | int | bool]:
        self.state = RLState(task=task)
        return self.observation()

    def _require_state(self) -> RLState:
        if self.state is None:
            raise RuntimeError("Call reset(task) before step(action).")
        return self.state

    def valid_actions(self) -> set[ControllerAction]:
        state = self._require_state()
        if state.done:
            return set()

        actions = {
            ControllerAction.CALL_PLANNER,
            ControllerAction.CALL_EXECUTOR,
            ControllerAction.STOP,
        }
        if state.candidate is not None:
            actions.add(ControllerAction.CALL_VERIFIER)
        if (
            state.candidate is not None
            and state.public_feedback is not None
            and state.public_feedback.status == "fail"
        ):
            actions.add(ControllerAction.CALL_REPAIR)
        return actions

    def action_mask(self) -> list[int]:
        valid = self.valid_actions()
        return [int(action in valid) for action in ControllerAction]

    def observation(self) -> dict[str, float | int | bool]:
        state = self._require_state()
        public = state.public_feedback
        return {
            "has_plan": state.plan is not None,
            "has_candidate": state.candidate is not None,
            "has_verifier_feedback": state.verifier_feedback is not None,
            "verifier_pass": state.verifier_verdict == "PASS",
            "verifier_revise": state.verifier_verdict == "REVISE",
            "public_examples_available": public is not None,
            "public_examples_pass": bool(public and public.passed),
            "public_examples_fail": bool(public and public.status == "fail"),
            "preflight_ok": bool(state.preflight_ok),
            "mechanical_repairs": state.mechanical_repairs,
            "llm_calls": state.llm_calls,
            "total_tokens": state.total_tokens,
            "total_latency_s": state.total_latency_s,
            "step_count": state.step_count,
            "remaining_steps": max(0, self.max_steps - state.step_count),
        }

    def _record_step(self, step: StepRecord) -> float:
        state = self._require_state()
        state.history.append(step)
        state.llm_calls += 1
        tokens = step.input_tokens + step.output_tokens
        state.total_tokens += tokens
        state.total_latency_s += step.latency_s
        return llm_step_cost(
            tokens=tokens,
            latency_s=step.latency_s,
            config=self.reward_config,
        )

    def _inspect_candidate(self, raw_candidate: str) -> None:
        state = self._require_state()
        mechanical = repair_known_preflight_issues(raw_candidate)
        if mechanical.applied:
            state.mechanical_repairs += 1
        state.candidate = mechanical.repaired_code
        state.preflight_ok = preflight_solution(state.candidate).ok
        state.verifier_feedback = None
        state.verifier_verdict = "UNKNOWN"

        if self.public_feedback_fn is None:
            state.public_feedback = None
        else:
            state.public_feedback = self.public_feedback_fn(
                state.task,
                state.candidate,
            )

    def _planner_prompt(self) -> str:
        state = self._require_state()
        return f"""Original task

{state.task}

Produce a concise plan for the Executor."""

    def _executor_prompt(self) -> str:
        state = self._require_state()
        guidance = state.plan or "No Planner guidance is available. Solve from the task directly."
        return f"""Original task

{state.task}

Planner guidance

{guidance}

Produce the candidate solution."""

    def _verifier_prompt(self) -> str:
        state = self._require_state()
        return f"""Original task

{state.task}

Candidate solution

{state.candidate}

Verify the candidate independently."""

    def _repair_prompt(self) -> str:
        state = self._require_state()
        report = state.public_feedback.report if state.public_feedback else ""
        return f"""Original task

{state.task}

Candidate solution

```python
{state.candidate}
```

Public example failure report

{report or "One or more public examples failed."}

Repair the candidate using only the original task and the public example failure report."""

    def _score_terminal(self) -> tuple[float, dict[str, object]]:
        state = self._require_state()
        success = 0.0
        if state.candidate is not None and self.terminal_score_fn is not None:
            success = float(self.terminal_score_fn(state.task, state.candidate))
        state.terminal_success = success
        state.done = True
        return terminal_reward(success=success, config=self.reward_config), {
            "terminal_success": success
        }

    def step(self, action: ControllerAction | int) -> StepTransition:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode is already terminated.")

        action = ControllerAction(action)
        if action not in self.valid_actions():
            state.step_count += 1
            reward = -self.reward_config.invalid_action_penalty
            info: dict[str, object] = {
                "action": action.name,
                "invalid_action": True,
            }
            if state.step_count >= self.max_steps:
                terminal, terminal_info = self._score_terminal()
                reward += terminal
                info.update(terminal_info)
            return StepTransition(self.observation(), reward, state.done, info)

        reward = 0.0
        info = {"action": action.name, "invalid_action": False}

        if action == ControllerAction.STOP:
            reward, terminal_info = self._score_terminal()
            info.update(terminal_info)
            return StepTransition(self.observation(), reward, True, info)

        if action == ControllerAction.CALL_PLANNER:
            step = self.planner.run(self._planner_prompt())
            state.plan = step.response
            reward -= self._record_step(step)

        elif action == ControllerAction.CALL_EXECUTOR:
            step = self.executor.run(self._executor_prompt())
            reward -= self._record_step(step)
            self._inspect_candidate(extract_code(step.response))

        elif action == ControllerAction.CALL_VERIFIER:
            step = self.verifier.run(self._verifier_prompt())
            reward -= self._record_step(step)
            state.verifier_feedback = step.response
            state.verifier_verdict = parse_verdict(step.response)

        elif action == ControllerAction.CALL_REPAIR:
            step = self.repairer.run(self._repair_prompt())
            reward -= self._record_step(step)
            self._inspect_candidate(extract_code(step.response))

        state.step_count += 1
        if state.step_count >= self.max_steps:
            terminal, terminal_info = self._score_terminal()
            reward += terminal
            info.update(terminal_info)

        info["action_mask"] = self.action_mask() if not state.done else [0] * self.action_size
        return StepTransition(self.observation(), reward, state.done, info)
