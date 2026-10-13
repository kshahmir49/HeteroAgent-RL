from __future__ import annotations

import json

import pytest

from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.router import (
    ROUTER_FEATURE_NAMES,
    RouterPolicy,
    feature_vector,
    pair_trajectory_rows,
)


def task_features():
    return {
        "prompt_chars": 100,
        "prompt_lines": 10,
        "prompt_words": 20,
        "function_count": 1,
        "helper_count": 0,
        "import_count": 0,
        "public_example_count": 2,
        "argument_count": 1,
        "annotation_count": 0,
        "has_varargs": False,
        "has_kwargs": False,
        "parse_ok": True,
    }


def trajectory(task_id, *, source, strict, reward):
    return {
        "task_id": task_id,
        "candidate_source": source,
        "task_features": task_features(),
        "base_status": "pass" if strict else "fail",
        "strict_success": strict,
        "policy_reward": reward,
    }


def test_pair_trajectory_rows_uses_reward_for_preferred_route():
    executor = [
        trajectory("HumanEval/25", source="executor", strict=True, reward=0.8),
        trajectory("HumanEval/26", source="executor", strict=False, reward=-0.1),
    ]
    specialist = [
        trajectory("HumanEval/25", source="specialist", strict=True, reward=0.7),
        trajectory("HumanEval/26", source="specialist", strict=True, reward=0.9),
    ]

    paired = pair_trajectory_rows(executor, specialist)

    assert paired[0]["preferred_label"] == 0
    assert paired[1]["preferred_label"] == 1
    assert len(paired[0]["features"]) == len(ROUTER_FEATURE_NAMES)


def test_pair_trajectory_rows_rejects_mismatched_task_sets():
    executor = [trajectory("HumanEval/25", source="executor", strict=True, reward=0.8)]
    specialist = [trajectory("HumanEval/26", source="specialist", strict=True, reward=0.9)]

    with pytest.raises(ValueError):
        pair_trajectory_rows(executor, specialist)


def test_serialized_router_policy_predicts_actions(tmp_path):
    payload = {
        "policy_type": "decision_tree_router",
        "feature_schema_version": 1,
        "feature_names": ROUTER_FEATURE_NAMES,
        "nodes": [
            {
                "feature_index": 0,
                "threshold": 0.05,
                "left": 1,
                "right": 2,
                "prediction": None,
            },
            {"prediction": 0},
            {"prediction": 1},
        ],
        "metadata": {},
    }
    path = tmp_path / "router.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    policy = RouterPolicy.load(path)

    low = {name: 0.0 for name in ROUTER_FEATURE_NAMES}
    high = dict(low)
    high["task_length_norm"] = 0.2

    assert policy.predict_action(low) == ControllerAction.CALL_EXECUTOR
    assert policy.predict_action(high) == ControllerAction.CALL_SPECIALIST
