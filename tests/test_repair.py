from heteroagent_rl.repair import repair_known_preflight_issues


def test_repairs_missing_list_import():
    code = """def f(xs: List[int]) -> List[int]:
    return xs
"""
    result = repair_known_preflight_issues(code)

    assert result.applied
    assert result.repair_type == "missing_typing_import"
    assert result.details == {"imports_added": ["List"]}
    assert result.repaired_code.startswith("from typing import List\n")
    assert result.repaired_preflight is not None
    assert result.repaired_preflight.ok


def test_repairs_multiple_typing_imports_deterministically():
    code = """def f(xs: List[int]) -> Optional[int]:
    return xs[0] if xs else None
"""
    result = repair_known_preflight_issues(code)

    assert result.applied
    assert result.repaired_code.startswith("from typing import List, Optional\n")


def test_preserves_future_import_position():
    code = """from __future__ import annotations

def f(xs: List[int]) -> List[int]:
    return xs
"""
    result = repair_known_preflight_issues(code)

    assert result.applied
    assert result.repaired_code.splitlines()[:2] == [
        "from __future__ import annotations",
        "from typing import List",
    ]


def test_does_not_repair_unknown_unresolved_name():
    code = """def f(xs: CustomType) -> List[int]:
    return []
"""
    result = repair_known_preflight_issues(code)

    assert not result.applied
    assert result.repaired_code == code


def test_does_not_touch_syntax_errors():
    code = "def f(:\n    pass"
    result = repair_known_preflight_issues(code)

    assert not result.applied
    assert result.repaired_code == code
