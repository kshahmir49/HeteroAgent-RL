from heteroagent_rl.preflight import preflight_solution


def test_preflight_accepts_complete_typing_import():
    code = """from typing import List

def f(xs: List[int]) -> List[int]:
    return xs
"""
    result = preflight_solution(code)
    assert result.ok
    assert result.unresolved_annotation_names == []


def test_preflight_catches_missing_list_import():
    code = """def f(xs: List[int]) -> bool:
    return bool(xs)
"""
    result = preflight_solution(code)
    assert not result.ok
    assert result.unresolved_annotation_names == ["List"]
    assert result.likely_missing_typing_imports == ["List"]


def test_preflight_reports_syntax_error():
    result = preflight_solution("def f(:\n    pass")
    assert not result.ok
    assert result.syntax_error is not None
