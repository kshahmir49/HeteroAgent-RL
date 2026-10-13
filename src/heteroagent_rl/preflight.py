from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field


BUILTIN_NAMES = set(dir(builtins))
TYPING_NAMES = {
    "Any",
    "Callable",
    "ClassVar",
    "Dict",
    "Final",
    "FrozenSet",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "Mapping",
    "MutableMapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "Union",
}


@dataclass
class PreflightResult:
    ok: bool
    syntax_error: str | None = None
    unresolved_annotation_names: list[str] = field(default_factory=list)
    likely_missing_typing_imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "syntax_error": self.syntax_error,
            "unresolved_annotation_names": self.unresolved_annotation_names,
            "likely_missing_typing_imports": self.likely_missing_typing_imports,
        }


def _bound_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

    return names


def _annotation_nodes(tree: ast.AST) -> list[ast.AST]:
    annotations: list[ast.AST] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                args.append(node.args.vararg)
            if node.args.kwarg is not None:
                args.append(node.args.kwarg)

            for arg in args:
                if arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)

        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)

    return annotations


def preflight_solution(code: str) -> PreflightResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        message = f"{exc.msg} at line {exc.lineno}"
        return PreflightResult(ok=False, syntax_error=message)

    available = BUILTIN_NAMES | _bound_module_names(tree)
    unresolved: set[str] = set()

    for annotation in _annotation_nodes(tree):
        for node in ast.walk(annotation):
            if isinstance(node, ast.Name) and node.id not in available:
                unresolved.add(node.id)

    missing_typing = sorted(unresolved & TYPING_NAMES)
    unresolved_sorted = sorted(unresolved)

    return PreflightResult(
        ok=not unresolved_sorted,
        unresolved_annotation_names=unresolved_sorted,
        likely_missing_typing_imports=missing_typing,
    )
