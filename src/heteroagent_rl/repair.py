from __future__ import annotations

import ast
from dataclasses import dataclass, field

from heteroagent_rl.preflight import PreflightResult, preflight_solution


@dataclass
class RepairResult:
    raw_code: str
    repaired_code: str
    applied: bool
    repair_type: str | None = None
    details: dict[str, object] = field(default_factory=dict)
    raw_preflight: PreflightResult | None = None
    repaired_preflight: PreflightResult | None = None

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "repair_type": self.repair_type,
            "details": self.details,
            "raw_preflight": (
                self.raw_preflight.to_dict() if self.raw_preflight is not None else None
            ),
            "repaired_preflight": (
                self.repaired_preflight.to_dict()
                if self.repaired_preflight is not None
                else None
            ),
        }


def _typing_import_insert_index(code: str) -> int:
    """Return a zero-based line index after any module docstring/future imports."""
    tree = ast.parse(code)
    index = 0

    if tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            index = first.end_lineno or first.lineno

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            index = max(index, node.end_lineno or node.lineno)

    return index


def repair_known_preflight_issues(code: str) -> RepairResult:
    """Apply only deterministic repairs with an unambiguous static interpretation."""
    raw_preflight = preflight_solution(code)

    if raw_preflight.ok or raw_preflight.syntax_error is not None:
        return RepairResult(
            raw_code=code,
            repaired_code=code,
            applied=False,
            raw_preflight=raw_preflight,
            repaired_preflight=raw_preflight,
        )

    unresolved = set(raw_preflight.unresolved_annotation_names)
    missing_typing = set(raw_preflight.likely_missing_typing_imports)

    if not missing_typing or unresolved != missing_typing:
        return RepairResult(
            raw_code=code,
            repaired_code=code,
            applied=False,
            raw_preflight=raw_preflight,
            repaired_preflight=raw_preflight,
        )

    names = sorted(missing_typing)
    import_line = f"from typing import {', '.join(names)}"

    lines = code.splitlines()
    insert_at = _typing_import_insert_index(code)
    lines.insert(insert_at, import_line)
    repaired = "\n".join(lines)

    if code.endswith("\n"):
        repaired += "\n"

    repaired_preflight = preflight_solution(repaired)

    return RepairResult(
        raw_code=code,
        repaired_code=repaired,
        applied=True,
        repair_type="missing_typing_import",
        details={"imports_added": names},
        raw_preflight=raw_preflight,
        repaired_preflight=repaired_preflight,
    )
