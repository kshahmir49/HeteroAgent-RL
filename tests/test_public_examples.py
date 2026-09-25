from heteroagent_rl.sandbox.worker import run_public_examples


def test_public_examples_pass_for_standalone_solution():
    problem = {
        "task_id": "HumanEval/test",
        "entry_point": "double",
        "prompt": """def double(x: int) -> int:
    \"\"\"Return twice x.
    >>> double(2)
    4
    >>> double(0)
    0
    \"\"\"
""",
    }
    solution = """def double(x: int) -> int:
    return x * 2
"""

    result = run_public_examples(problem, solution)

    assert result["status"] == "pass"
    assert result["attempted"] == 2
    assert result["failed"] == 0


def test_public_examples_catch_wrong_behavior():
    problem = {
        "task_id": "HumanEval/test",
        "entry_point": "double",
        "prompt": """def double(x: int) -> int:
    \"\"\"Return twice x.
    >>> double(2)
    4
    \"\"\"
""",
    }
    solution = """def double(x: int) -> int:
    return x + 1
"""

    result = run_public_examples(problem, solution)

    assert result["status"] == "fail"
    assert result["failed"] == 1
    assert "Expected" in result["report"]


def test_public_examples_do_not_inject_prompt_helpers():
    problem = {
        "task_id": "HumanEval/test",
        "entry_point": "uses_helper",
        "prompt": """def helper(x: int) -> int:
    return x + 1

def uses_helper(x: int) -> int:
    \"\"\"Use the helper.
    >>> uses_helper(1)
    2
    \"\"\"
""",
    }
    solution = """def uses_helper(x: int) -> int:
    return helper(x)
"""

    result = run_public_examples(problem, solution)

    assert result["status"] == "fail"
    assert "NameError" in result["report"]
