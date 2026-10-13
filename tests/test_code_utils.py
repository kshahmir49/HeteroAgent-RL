from heteroagent_rl.code_utils import extract_code, parse_verdict


def test_extract_code_from_fence():
    text = """Here is the solution.

```python
def add(a, b):
    return a + b
```
"""
    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_extract_code_without_fence():
    assert extract_code("def f():\n    return 1") == "def f():\n    return 1"


def test_parse_verdict():
    assert parse_verdict("VERDICT: PASS\nLooks good") == "PASS"
    assert parse_verdict("VERDICT: REVISE\nBug") == "REVISE"
    assert parse_verdict("Maybe") == "UNKNOWN"
