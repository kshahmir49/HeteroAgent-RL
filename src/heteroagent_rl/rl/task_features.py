from __future__ import annotations

import ast
import doctest
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskFeatures:
    prompt_chars: int
    prompt_lines: int
    prompt_words: int
    function_count: int
    helper_count: int
    import_count: int
    public_example_count: int
    argument_count: int
    annotation_count: int
    has_varargs: bool
    has_kwargs: bool
    parse_ok: bool

    @property
    def has_public_examples(self) -> bool:
        return self.public_example_count > 0

    @property
    def has_helpers(self) -> bool:
        return self.helper_count > 0

    @property
    def has_imports(self) -> bool:
        return self.import_count > 0

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    def to_observation(self) -> dict[str, float | int | bool]:
        return {
            "task_length_norm": min(self.prompt_chars / 4000.0, 1.0),
            "task_lines_norm": min(self.prompt_lines / 100.0, 1.0),
            "task_words_norm": min(self.prompt_words / 800.0, 1.0),
            "task_function_count_norm": min(self.function_count / 6.0, 1.0),
            "task_helper_count_norm": min(self.helper_count / 5.0, 1.0),
            "task_import_count_norm": min(self.import_count / 5.0, 1.0),
            "task_public_example_count_norm": min(self.public_example_count / 8.0, 1.0),
            "task_argument_count_norm": min(self.argument_count / 10.0, 1.0),
            "task_annotation_count_norm": min(self.annotation_count / 12.0, 1.0),
            "task_has_public_examples": self.has_public_examples,
            "task_has_helpers": self.has_helpers,
            "task_has_imports": self.has_imports,
            "task_has_varargs": self.has_varargs,
            "task_has_kwargs": self.has_kwargs,
            "task_parse_ok": self.parse_ok,
        }


def _function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, bool, bool]:
    count = (
        len(node.args.posonlyargs)
        + len(node.args.args)
        + len(node.args.kwonlyargs)
    )
    if node.args.vararg is not None:
        count += 1
    if node.args.kwarg is not None:
        count += 1
    return count, node.args.vararg is not None, node.args.kwarg is not None


def _annotation_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    count = 0
    args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)

    count += sum(arg.annotation is not None for arg in args)
    count += int(node.returns is not None)
    return count


def _doctest_count(text: str) -> int:
    if not text:
        return 0
    try:
        return len(doctest.DocTestParser().get_examples(text))
    except ValueError:
        return len(re.findall(r"(?m)^\s*>>>\s+", text))


def extract_task_features(task: str, entry_point: str | None = None) -> TaskFeatures:
    prompt_chars = len(task)
    prompt_lines = len(task.splitlines())
    prompt_words = len(re.findall(r"\S+", task))

    try:
        tree = ast.parse(task)
    except SyntaxError:
        return TaskFeatures(
            prompt_chars=prompt_chars,
            prompt_lines=prompt_lines,
            prompt_words=prompt_words,
            function_count=0,
            helper_count=0,
            import_count=0,
            public_example_count=len(re.findall(r"(?m)^\s*>>>\s+", task)),
            argument_count=0,
            annotation_count=0,
            has_varargs=False,
            has_kwargs=False,
            parse_ok=False,
        )

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    target = None
    if entry_point is not None:
        target = next((node for node in functions if node.name == entry_point), None)
    if target is None and functions:
        target = functions[-1]

    helper_count = len(functions)
    if target is not None:
        helper_count -= 1

    public_example_count = 0
    if target is not None:
        public_example_count = _doctest_count(ast.get_docstring(target, clean=True) or "")
    elif functions:
        public_example_count = sum(
            _doctest_count(ast.get_docstring(node, clean=True) or "")
            for node in functions
        )

    argument_count = 0
    has_varargs = False
    has_kwargs = False
    annotation_count = 0
    if target is not None:
        argument_count, has_varargs, has_kwargs = _function_arguments(target)
        annotation_count = _annotation_count(target)

    return TaskFeatures(
        prompt_chars=prompt_chars,
        prompt_lines=prompt_lines,
        prompt_words=prompt_words,
        function_count=len(functions),
        helper_count=max(0, helper_count),
        import_count=len(imports),
        public_example_count=public_example_count,
        argument_count=argument_count,
        annotation_count=annotation_count,
        has_varargs=has_varargs,
        has_kwargs=has_kwargs,
        parse_ok=True,
    )
