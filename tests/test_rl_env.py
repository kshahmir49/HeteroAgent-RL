from __future__ import annotations

from dataclasses import dataclass

import pytest

from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.env import CandidateFeedback, HeteroAgentEnv
from heteroagent_rl.rl.reward import RewardConfig
from heteroagent_rl.schema import StepRecord


@dataclass
class FakeAgent:
    role: str
    responses: list[str]
    calls: int = 0

    def run(self, prompt: str) -> StepRecord:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return StepRecord(
            role=self.role,
            prompt=prompt,
            response=response,
            model="fake",
            latency_s=0.5,
            input_tokens=100,
            output_tokens=50,
        )


def build_env(
    *,
    public_feedback_fn=None,
    terminal_score_fn=None,
    max_steps=10,
    specialist=None,
):
    return HeteroAgentEnv(
        planner=FakeAgent("planner", ["Use direct arithmetic."]),
        executor=FakeAgent(
            "executor",
            ["def solve(x: int) -> int:\n    return x + 1"],
        ),
        verifier=FakeAgent(
            "verifier",
            ["VERDICT: PASS\nLooks correct."],
        ),
        repairer=FakeAgent(
            "repairer",
            ["def solve(x: int) -> int:\n    return x * 2"],
        ),
        specialist=specialist,
        public_feedback_fn=public_feedback_fn,
        terminal_score_fn=terminal_score_fn,
        reward_config=RewardConfig(
            success_reward=1.0,
            failure_reward=0.0,
            call_penalty=0.01,
            token_penalty_per_1k=0.0,
            latency_penalty_per_s=0.0,
            tool_call_penalty=0.02,
            tool_latency_penalty_per_s=0.0,
            invalid_action_penalty=0.05,
        ),
        max_steps=max_steps,
    )


def test_reset_and_action_mask_require_candidate_for_tools_and_verifier():
    env = build_env()
    obs = env.reset("Implement solve.")

    assert not obs["has_candidate"]
    assert env.action_mask() == [1, 1, 0, 0, 0, 1, 0]

    transition = env.step(ControllerAction.CALL_EXECUTOR)

    assert transition.reward == pytest.approx(-0.01)
    assert transition.observation["has_candidate"]
    assert not transition.observation["task_has_public_examples"]
    assert env.action_mask() == [1, 1, 0, 1, 0, 1, 0]


def test_terminal_scorer_runs_only_on_stop():
    calls = []

    def terminal(task: str, candidate: str) -> float:
        calls.append((task, candidate))
        return 1.0

    env = build_env(terminal_score_fn=terminal)
    env.reset("Implement solve.")

    env.step(ControllerAction.CALL_EXECUTOR)
    assert calls == []

    transition = env.step(ControllerAction.STOP)

    assert transition.terminated
    assert transition.reward == pytest.approx(1.0)
    assert transition.info["terminal_success"] == 1.0
    assert len(calls) == 1


def test_public_tests_are_explicit_and_unlock_repair():
    def public_feedback(task: str, candidate: str) -> CandidateFeedback:
        if "x * 2" in candidate:
            return CandidateFeedback(status="pass", attempted=1, failed=0, latency_s=0.25)
        return CandidateFeedback(
            status="fail",
            report="Expected 4 but got 3.",
            attempted=1,
            failed=1,
            latency_s=0.25,
        )

    env = build_env(public_feedback_fn=public_feedback)
    env.reset(
        """def solve(x: int) -> int:
    \"\"\"
    >>> solve(2)
    4
    \"\"\"
""",
        entry_point="solve",
    )

    generated = env.step(ControllerAction.CALL_EXECUTOR)
    assert not generated.observation["public_examples_available"]
    assert ControllerAction.RUN_PUBLIC_TESTS in env.valid_actions()
    assert ControllerAction.CALL_REPAIR not in env.valid_actions()

    tested = env.step(ControllerAction.RUN_PUBLIC_TESTS)
    assert tested.reward == pytest.approx(-0.02)
    assert tested.observation["public_examples_fail"]
    assert tested.observation["tool_calls"] == 1
    assert tested.observation["total_tool_latency_s"] == pytest.approx(0.25)
    assert ControllerAction.CALL_REPAIR in env.valid_actions()

    repaired = env.step(ControllerAction.CALL_REPAIR)
    assert not repaired.observation["public_examples_available"]
    assert ControllerAction.RUN_PUBLIC_TESTS in env.valid_actions()

    retested = env.step(ControllerAction.RUN_PUBLIC_TESTS)
    assert retested.observation["public_examples_pass"]
    assert retested.observation["tool_calls"] == 2
    assert ControllerAction.CALL_REPAIR not in env.valid_actions()


def test_invalid_repair_action_is_penalized_without_llm_call():
    env = build_env()
    env.reset("Implement solve.")

    transition = env.step(ControllerAction.CALL_REPAIR)

    assert transition.reward == pytest.approx(-0.05)
    assert transition.info["invalid_action"] is True
    assert env.state is not None
    assert env.state.llm_calls == 0


def test_step_budget_terminates_and_scores_current_candidate():
    env = build_env(
        terminal_score_fn=lambda task, candidate: 1.0,
        max_steps=1,
    )
    env.reset("Implement solve.")

    transition = env.step(ControllerAction.CALL_EXECUTOR)

    assert transition.terminated
    assert transition.info["terminal_success"] == 1.0
    assert transition.reward == pytest.approx(0.99)


def test_verifier_updates_observation_but_not_terminal_truth():
    env = build_env(terminal_score_fn=lambda task, candidate: 0.0)
    env.reset("Implement solve.")
    env.step(ControllerAction.CALL_EXECUTOR)

    verification = env.step(ControllerAction.CALL_VERIFIER)

    assert verification.observation["verifier_pass"]
    assert not verification.terminated

    stopped = env.step(ControllerAction.STOP)

    assert stopped.info["terminal_success"] == 0.0
    assert stopped.reward == pytest.approx(0.0)


def test_public_test_action_masked_when_task_has_no_examples():
    env = build_env(
        public_feedback_fn=lambda task, candidate: CandidateFeedback(status="no_examples")
    )
    env.reset("def solve(x: int) -> int:\n    \"\"\"Return x.\"\"\"\n", entry_point="solve")
    env.step(ControllerAction.CALL_EXECUTOR)

    assert not env.observation()["task_has_public_examples"]
    assert ControllerAction.RUN_PUBLIC_TESTS not in env.valid_actions()


def test_specialist_is_explicit_action_and_marks_candidate_source():
    specialist = FakeAgent(
        "specialist",
        ["def solve(x: int) -> int:\n    return x * 3"],
    )
    env = build_env(specialist=specialist)
    initial = env.reset("def solve(x: int) -> int:\n    return x\n", entry_point="solve")

    assert not initial["candidate_is_specialist"]
    assert ControllerAction.CALL_SPECIALIST in env.valid_actions()

    transition = env.step(ControllerAction.CALL_SPECIALIST)

    assert transition.observation["has_candidate"]
    assert transition.observation["candidate_is_specialist"]
    assert env.state is not None
    assert env.state.candidate_source == "specialist"
    assert env.state.llm_calls == 1


def test_specialist_action_is_invalid_when_not_configured():
    env = build_env()
    env.reset("Implement solve.")

    transition = env.step(ControllerAction.CALL_SPECIALIST)

    assert transition.info["invalid_action"] is True
    assert transition.reward == pytest.approx(-0.05)
