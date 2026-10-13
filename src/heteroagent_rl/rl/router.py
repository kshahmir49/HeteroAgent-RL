from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from heteroagent_rl.rl.actions import ControllerAction
from heteroagent_rl.rl.task_features import TaskFeatures


ROUTER_FEATURE_NAMES = [
    "task_length_norm",
    "task_lines_norm",
    "task_words_norm",
    "task_function_count_norm",
    "task_helper_count_norm",
    "task_import_count_norm",
    "task_public_example_count_norm",
    "task_argument_count_norm",
    "task_annotation_count_norm",
    "task_has_public_examples",
    "task_has_helpers",
    "task_has_imports",
    "task_has_varargs",
    "task_has_kwargs",
    "task_parse_ok",
]

EXECUTOR_LABEL = 0
SPECIALIST_LABEL = 1
LABEL_TO_ACTION = {
    EXECUTOR_LABEL: ControllerAction.CALL_EXECUTOR,
    SPECIALIST_LABEL: ControllerAction.CALL_SPECIALIST,
}


def _numeric(value: float | int | bool) -> float:
    return float(value)


def task_feature_observation(raw: dict[str, Any]) -> dict[str, float | int | bool]:
    return TaskFeatures(**raw).to_observation()


def feature_vector(observation: dict[str, float | int | bool]) -> list[float]:
    return [_numeric(observation[name]) for name in ROUTER_FEATURE_NAMES]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pair_trajectory_rows(
    executor_rows: Iterable[dict[str, Any]],
    specialist_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    executor = {str(row["task_id"]): row for row in executor_rows}
    specialist = {str(row["task_id"]): row for row in specialist_rows}
    if set(executor) != set(specialist):
        missing_executor = sorted(set(specialist) - set(executor))
        missing_specialist = sorted(set(executor) - set(specialist))
        raise ValueError(
            "Executor and specialist trajectories must contain identical task IDs. "
            f"Missing executor IDs {missing_executor}; "
            f"missing specialist IDs {missing_specialist}."
        )

    paired: list[dict[str, Any]] = []
    for task_id in sorted(executor):
        executor_row = executor[task_id]
        specialist_row = specialist[task_id]
        executor_features = executor_row.get("task_features") or {}
        specialist_features = specialist_row.get("task_features") or {}
        if executor_features != specialist_features:
            raise ValueError(f"Task features differ between trajectories for {task_id}.")

        executor_reward = float(executor_row["policy_reward"])
        specialist_reward = float(specialist_row["policy_reward"])
        preferred = (
            SPECIALIST_LABEL
            if specialist_reward > executor_reward
            else EXECUTOR_LABEL
        )
        observation = task_feature_observation(executor_features)
        paired.append(
            {
                "task_id": task_id,
                "features": feature_vector(observation),
                "executor_strict_success": bool(executor_row["strict_success"]),
                "specialist_strict_success": bool(specialist_row["strict_success"]),
                "executor_base_pass": executor_row.get("base_status") == "pass",
                "specialist_base_pass": specialist_row.get("base_status") == "pass",
                "executor_reward": executor_reward,
                "specialist_reward": specialist_reward,
                "preferred_label": preferred,
            }
        )
    return paired


@dataclass(frozen=True)
class RouterPolicy:
    feature_names: list[str]
    nodes: list[dict[str, Any]]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RouterPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("policy_type") != "decision_tree_router":
            raise ValueError("Unsupported router policy type.")
        feature_names = list(payload["feature_names"])
        if feature_names != ROUTER_FEATURE_NAMES:
            raise ValueError("Router feature schema does not match this code version.")
        return cls(
            feature_names=feature_names,
            nodes=list(payload["nodes"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def predict_label(self, observation: dict[str, float | int | bool]) -> int:
        values = feature_vector(observation)
        node_index = 0
        while True:
            node = self.nodes[node_index]
            prediction = node.get("prediction")
            if prediction is not None:
                return int(prediction)
            feature_index = int(node["feature_index"])
            threshold = float(node["threshold"])
            if values[feature_index] <= threshold:
                node_index = int(node["left"])
            else:
                node_index = int(node["right"])

    def predict_action(
        self,
        observation: dict[str, float | int | bool],
    ) -> ControllerAction:
        return LABEL_TO_ACTION[self.predict_label(observation)]


def policy_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serialize_tree(classifier: Any) -> list[dict[str, Any]]:
    tree = classifier.tree_
    classes = [int(value) for value in classifier.classes_]
    nodes: list[dict[str, Any]] = []
    for index in range(tree.node_count):
        left = int(tree.children_left[index])
        right = int(tree.children_right[index])
        if left == right:
            counts = tree.value[index][0]
            prediction = classes[max(range(len(counts)), key=lambda i: counts[i])]
            nodes.append({"prediction": int(prediction)})
        else:
            nodes.append(
                {
                    "feature_index": int(tree.feature[index]),
                    "threshold": float(tree.threshold[index]),
                    "left": left,
                    "right": right,
                    "prediction": None,
                }
            )
    return nodes


def _folds(labels: list[int], *, seed: int) -> list[tuple[list[int], list[int]]]:
    from sklearn.model_selection import KFold, StratifiedKFold

    counts = {label: labels.count(label) for label in set(labels)}
    minimum = min(counts.values()) if counts else 0
    n_splits = min(5, minimum)
    indices = list(range(len(labels)))
    if n_splits >= 2:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return [
            (list(train), list(valid))
            for train, valid in splitter.split(indices, labels)
        ]

    splitter = KFold(
        n_splits=min(5, len(labels)),
        shuffle=True,
        random_state=seed,
    )
    return [(list(train), list(valid)) for train, valid in splitter.split(indices)]


def train_router_policy(
    examples: list[dict[str, Any]],
    *,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(examples) < 4:
        raise ValueError("At least four paired training examples are required.")

    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Router training requires scikit-learn. "
            "Install with pip install -e '.[router]'."
        ) from exc

    x = [list(map(float, row["features"])) for row in examples]
    y = [int(row["preferred_label"]) for row in examples]
    folds = _folds(y, seed=seed)

    candidates = [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (1, 2, 3, 4)
        for leaf in (2, 4, 8)
    ]

    scored: list[dict[str, Any]] = []
    for params in candidates:
        fold_rewards: list[float] = []
        fold_strict: list[float] = []
        for train_idx, valid_idx in folds:
            classifier = DecisionTreeClassifier(
                random_state=seed,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
            )
            classifier.fit([x[i] for i in train_idx], [y[i] for i in train_idx])
            predictions = classifier.predict([x[i] for i in valid_idx])

            rewards: list[float] = []
            strict: list[float] = []
            for prediction, index in zip(predictions, valid_idx):
                row = examples[index]
                if int(prediction) == SPECIALIST_LABEL:
                    rewards.append(float(row["specialist_reward"]))
                    strict.append(float(row["specialist_strict_success"]))
                else:
                    rewards.append(float(row["executor_reward"]))
                    strict.append(float(row["executor_strict_success"]))
            fold_rewards.append(sum(rewards) / len(rewards))
            fold_strict.append(sum(strict) / len(strict))

        scored.append(
            {
                **params,
                "cv_mean_reward": sum(fold_rewards) / len(fold_rewards),
                "cv_mean_strict_success": sum(fold_strict) / len(fold_strict),
            }
        )

    best = max(
        scored,
        key=lambda row: (
            row["cv_mean_reward"],
            row["cv_mean_strict_success"],
            -row["max_depth"],
            row["min_samples_leaf"],
        ),
    )

    classifier = DecisionTreeClassifier(
        random_state=seed,
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
    )
    classifier.fit(x, y)

    executor_wins = sum(row["preferred_label"] == EXECUTOR_LABEL for row in examples)
    specialist_wins = len(examples) - executor_wins
    oracle_success = sum(
        row["executor_strict_success"] or row["specialist_strict_success"]
        for row in examples
    )

    metadata = {
        "seed": seed,
        "num_training_tasks": len(examples),
        "executor_preferred": executor_wins,
        "specialist_preferred": specialist_wins,
        "selected_params": {
            "max_depth": int(best["max_depth"]),
            "min_samples_leaf": int(best["min_samples_leaf"]),
        },
        "cv_mean_reward": float(best["cv_mean_reward"]),
        "cv_mean_strict_success": float(best["cv_mean_strict_success"]),
        "train_executor_strict_rate": sum(
            row["executor_strict_success"] for row in examples
        ) / len(examples),
        "train_specialist_strict_rate": sum(
            row["specialist_strict_success"] for row in examples
        ) / len(examples),
        "train_oracle_strict_rate": oracle_success / len(examples),
        "candidate_scores": scored,
    }
    payload = {
        "policy_type": "decision_tree_router",
        "feature_schema_version": 1,
        "feature_names": ROUTER_FEATURE_NAMES,
        "nodes": _serialize_tree(classifier),
        "metadata": metadata,
    }
    return payload, metadata


def save_router_policy(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
