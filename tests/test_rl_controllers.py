from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.controllers import (
    ExecutorOnlyController,
    FullPipelineController,
    PlannerExecutorController,
    PublicRepairController,
)


def obs(**overrides):
    base = {
        "has_plan": False,
        "has_candidate": False,
        "has_verifier_feedback": False,
        "verifier_pass": False,
        "verifier_revise": False,
        "public_examples_available": False,
        "public_examples_pass": False,
        "public_examples_fail": False,
        "preflight_ok": False,
        "mechanical_repairs": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "total_tokens": 0,
        "total_latency_s": 0.0,
        "total_tool_latency_s": 0.0,
        "step_count": 0,
        "remaining_steps": 10,
    }
    base.update(overrides)
    return base


def test_executor_only_calls_executor_then_stops():
    controller = ExecutorOnlyController()
    controller.reset()

    assert controller.select_action(obs(), [1, 1, 0, 0, 0, 1]) == ControllerAction.CALL_EXECUTOR
    assert controller.select_action(obs(has_candidate=True), [1] * 6) == ControllerAction.STOP


def test_planner_executor_uses_plan_before_executor():
    controller = PlannerExecutorController()
    controller.reset()

    assert controller.select_action(obs(), [1, 1, 0, 0, 0, 1]) == ControllerAction.CALL_PLANNER
    assert controller.select_action(obs(has_plan=True), [1] * 6) == ControllerAction.CALL_EXECUTOR
    assert controller.select_action(
        obs(has_plan=True, has_candidate=True), [1] * 6
    ) == ControllerAction.STOP


def test_public_repair_explicitly_tests_repairs_and_retests():
    controller = PublicRepairController()
    controller.reset()

    assert controller.select_action(obs(), [1] * 6) == ControllerAction.CALL_PLANNER
    assert controller.select_action(obs(has_plan=True), [1] * 6) == ControllerAction.CALL_EXECUTOR
    assert controller.select_action(
        obs(has_plan=True, has_candidate=True), [1] * 6
    ) == ControllerAction.RUN_PUBLIC_TESTS
    assert controller.select_action(
        obs(
            has_plan=True,
            has_candidate=True,
            public_examples_available=True,
            public_examples_fail=True,
        ),
        [1] * 6,
    ) == ControllerAction.CALL_REPAIR
    assert controller.select_action(
        obs(has_plan=True, has_candidate=True), [1] * 6
    ) == ControllerAction.RUN_PUBLIC_TESTS
    assert controller.select_action(
        obs(
            has_plan=True,
            has_candidate=True,
            public_examples_available=True,
            public_examples_pass=True,
        ),
        [1] * 6,
    ) == ControllerAction.STOP


def test_full_pipeline_tests_then_verifies():
    controller = FullPipelineController()
    controller.reset()

    assert controller.select_action(obs(), [1] * 6) == ControllerAction.CALL_PLANNER
    assert controller.select_action(obs(has_plan=True), [1] * 6) == ControllerAction.CALL_EXECUTOR
    assert controller.select_action(
        obs(has_plan=True, has_candidate=True), [1] * 6
    ) == ControllerAction.RUN_PUBLIC_TESTS
    assert controller.select_action(
        obs(
            has_plan=True,
            has_candidate=True,
            public_examples_available=True,
            public_examples_pass=True,
        ),
        [1] * 6,
    ) == ControllerAction.CALL_VERIFIER
    assert controller.select_action(
        obs(
            has_plan=True,
            has_candidate=True,
            public_examples_available=True,
            public_examples_pass=True,
            has_verifier_feedback=True,
        ),
        [1] * 6,
    ) == ControllerAction.STOP
