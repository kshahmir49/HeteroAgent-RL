from __future__ import annotations

import argparse
import ast
import doctest
import hashlib
import io
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evalplus", "public"], default="evalplus")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"])
    parser.add_argument("--problems", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, str]:
    samples: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples[row["task_id"]] = row["solution"]
    return samples


def subset_hash(benchmark: str, task_ids: list[str]) -> str:
    raw = benchmark + "|" + "|".join(task_ids)
    return "heteroagent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def support_code(prompt: str, entry_point: str) -> str:
    """Keep imports/helpers from the public prompt but drop the target function."""
    tree = ast.parse(prompt)
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            continue
        kept.append(node)
    module = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def run_public_examples(problem: dict, solution: str) -> dict:
    parser = doctest.DocTestParser()
    examples = parser.get_examples(problem["prompt"])
    if not examples:
        return {
            "status": "no_examples",
            "attempted": 0,
            "failed": 0,
            "report": "",
        }

    globs: dict = {}
    try:
        # The system contract requires a standalone module. Do not inject helpers
        # or imports from the benchmark prompt here.
        exec(solution, globs)
    except BaseException as exc:
        return {
            "status": "fail",
            "attempted": len(examples),
            "failed": len(examples),
            "report": f"Candidate could not load: {type(exc).__name__}: {exc}",
        }

    doc = doctest.DocTest(
        examples=examples,
        globs=globs,
        name=problem["task_id"],
        filename=problem["task_id"],
        lineno=0,
        docstring=problem["prompt"],
    )
    output = io.StringIO()
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    result = runner.run(doc, out=output.write, clear_globs=False)
    report = output.getvalue()
    if len(report) > 4000:
        report = report[:4000] + "\n[report truncated]"

    return {
        "status": "pass" if result.failed == 0 else "fail",
        "attempted": result.attempted,
        "failed": result.failed,
        "report": report,
    }


def run_evalplus(benchmark: str, problems: dict, samples: dict) -> dict:
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    task_ids = list(problems)
    output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS if benchmark == "mbpp" else []
    expected = get_groundtruth(
        problems,
        subset_hash(benchmark, task_ids),
        output_not_none,
    )

    scored: dict[str, dict[str, str]] = {}
    for task_id in task_ids:
        result = check_correctness(
            benchmark,
            0,
            problems[task_id],
            samples[task_id],
            expected[task_id],
            base_only=False,
            fast_check=True,
            identifier=f"{task_id}:0",
        )
        scored[task_id] = {
            "base_status": result["base"][0],
            "plus_status": result["plus"][0],
        }
    return scored


def main() -> None:
    args = parse_args()

    with Path(args.problems).open("rb") as handle:
        problems = pickle.load(handle)

    samples = load_samples(Path(args.samples))
    task_ids = list(problems)
    missing = [task_id for task_id in task_ids if task_id not in samples]
    if missing:
        raise RuntimeError(f"Missing generated solutions for tasks: {missing}")

    if args.mode == "public":
        scored = {
            task_id: run_public_examples(problems[task_id], samples[task_id])
            for task_id in task_ids
        }
    else:
        if not args.benchmark:
            raise RuntimeError("--benchmark is required in evalplus mode")
        scored = run_evalplus(args.benchmark, problems, samples)

    Path(args.output).write_text(
        json.dumps(scored, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
