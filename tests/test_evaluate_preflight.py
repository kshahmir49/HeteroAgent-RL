from heteroagent_rl.preflight import preflight_solution


def test_humaneval_style_missing_typing_import_is_detected():
    code = """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    return False
"""
    result = preflight_solution(code)

    assert not result.ok
    assert result.likely_missing_typing_imports == ["List"]


def test_humaneval_style_complete_solution_passes():
    code = """from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    return False
"""
    result = preflight_solution(code)

    assert result.ok
