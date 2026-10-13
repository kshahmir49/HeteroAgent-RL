from __future__ import annotations

import re


_FENCED_CODE = re.compile(
    r"```(?:python)?\s*\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_code(text: str) -> str:
    """Return code from the first fenced block, or the raw response if unfenced."""
    match = _FENCED_CODE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_verdict(text: str) -> str:
    first_line = text.strip().splitlines()[0].upper() if text.strip() else ""
    if "VERDICT: PASS" in first_line:
        return "PASS"
    if "VERDICT: REVISE" in first_line:
        return "REVISE"
    return "UNKNOWN"
