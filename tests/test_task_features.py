from heteroagent_rl.rl.task_features import extract_task_features


def test_extracts_humaneval_style_task_features():
    task = """from typing import List

def helper(x: int) -> int:
    return x + 1

def solve(xs: List[int], scale: int = 1) -> int:
    \"\"\"Return a scaled sum.
    >>> solve([1, 2])
    3
    >>> solve([1, 2], 2)
    6
    \"\"\"
"""

    features = extract_task_features(task, entry_point="solve")

    assert features.parse_ok
    assert features.function_count == 2
    assert features.helper_count == 1
    assert features.import_count == 1
    assert features.public_example_count == 2
    assert features.argument_count == 2
    assert features.annotation_count == 3
    assert features.has_public_examples
    assert features.has_helpers
    assert features.has_imports


def test_uses_entry_point_not_helper_docstring_examples():
    task = """def helper(x):
    \"\"\"
    >>> helper(1)
    2
    \"\"\"
    return x + 1

def solve(x):
    \"\"\"Return x unchanged.\"\"\"
"""

    features = extract_task_features(task, entry_point="solve")

    assert features.public_example_count == 0
    assert not features.has_public_examples


def test_syntax_fallback_still_counts_visible_doctests():
    task = """Implement solve

>>> solve(1)
2
"""

    features = extract_task_features(task, entry_point="solve")

    assert not features.parse_ok
    assert features.public_example_count == 1


def test_observation_features_are_bounded():
    task = "def solve(x: int) -> int:\n    \"\"\"Return x.\"\"\"\n"
    observation = extract_task_features(task, "solve").to_observation()

    numeric = [
        value
        for value in observation.values()
        if isinstance(value, float)
    ]

    assert numeric
    assert all(0.0 <= value <= 1.0 for value in numeric)
