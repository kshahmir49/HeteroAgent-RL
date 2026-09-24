from __future__ import annotations

import json
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EVALPLUS_IMAGE = "ganler/evalplus:v0.3.1"
APPLE_EVALPLUS_IMAGE = "docker.io/ganler/evalplus:v0.3.1"


class SandboxExecutionError(RuntimeError):
    pass


def _command_works(command: list[str], timeout_s: float = 5.0) -> bool:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def available_backends() -> dict[str, bool]:
    docker = bool(shutil.which("docker")) and _command_works(["docker", "info"])
    apple = (
        platform.system() == "Darwin"
        and bool(shutil.which("container"))
        and _command_works(["container", "system", "status"])
    )
    return {
        "apple": apple,
        "docker": docker,
    }


def resolve_backend(requested: str = "auto") -> str:
    if requested not in {"auto", "apple", "docker"}:
        raise ValueError(f"Unknown sandbox backend: {requested}")

    availability = available_backends()

    if requested != "auto":
        if not availability[requested]:
            raise SandboxExecutionError(
                f"Sandbox backend '{requested}' is not available or not running."
            )
        return requested

    if availability["apple"]:
        return "apple"
    if availability["docker"]:
        return "docker"

    raise SandboxExecutionError(
        "No supported sandbox is available. Install/start Apple container on "
        "supported macOS or Docker, then rerun the sandbox doctor."
    )


def _docker_command(input_dir: Path, output_dir: Path, benchmark: str | None, mode: str = "evalplus") -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "128",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp",
        "-v",
        f"{input_dir}:/input:ro",
        "-v",
        f"{output_dir}:/output:rw",
        EVALPLUS_IMAGE,
        "python",
        "/input/worker.py",
        "--mode",
        mode,
        *(["--benchmark", benchmark] if benchmark else []),
        "--problems",
        "/input/problems.pkl",
        "--samples",
        "/input/samples.jsonl",
        "--output",
        "/output/results.json",
    ]


def _apple_command(input_dir: Path, output_dir: Path, benchmark: str | None, mode: str = "evalplus") -> list[str]:
    return [
        "container",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--rosetta",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--tmpfs",
        "/tmp:size=512M,mode=1777",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp",
        "--volume",
        f"{input_dir}:/input:ro",
        "--volume",
        f"{output_dir}:/output",
        APPLE_EVALPLUS_IMAGE,
        "python",
        "/input/worker.py",
        "--mode",
        mode,
        *(["--benchmark", benchmark] if benchmark else []),
        "--problems",
        "/input/problems.pkl",
        "--samples",
        "/input/samples.jsonl",
        "--output",
        "/output/results.json",
    ]


def _write_bundle(
    input_dir: Path,
    problems: dict[str, dict[str, Any]],
    solutions: dict[str, str],
) -> None:
    worker_source = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")
    (input_dir / "worker.py").write_text(worker_source, encoding="utf-8")

    with (input_dir / "problems.pkl").open("wb") as handle:
        pickle.dump(problems, handle)

    with (input_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in problems:
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "solution": solutions[task_id],
                    }
                )
                + "\n"
            )


def run_evalplus_sandbox(
    benchmark: str,
    problems: dict[str, dict[str, Any]],
    solutions: dict[str, str],
    *,
    backend: str = "auto",
    timeout_s: float = 300.0,
) -> tuple[str, dict[str, dict[str, str]]]:
    """Execute EvalPlus inside a container boundary.

    The container has no network, a read-only root filesystem, a temporary /tmp,
    read-only benchmark inputs, and a small writable output directory. Only the
    final normalized score JSON is copied back into the host process.
    """
    resolved = resolve_backend(backend)

    with tempfile.TemporaryDirectory(prefix="heteroagent-sandbox-input-") as in_tmp:
        with tempfile.TemporaryDirectory(prefix="heteroagent-sandbox-output-") as out_tmp:
            input_dir = Path(in_tmp).resolve()
            output_dir = Path(out_tmp).resolve()
            _write_bundle(input_dir, problems, solutions)

            if resolved == "docker":
                command = _docker_command(input_dir, output_dir, benchmark, "evalplus")
            else:
                command = _apple_command(input_dir, output_dir, benchmark, "evalplus")

            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxExecutionError(
                    f"Sandbox evaluation exceeded {timeout_s:.0f} seconds."
                ) from exc

            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                detail = stderr or stdout or "No error output was produced."
                raise SandboxExecutionError(
                    f"Sandbox evaluation failed with exit code {result.returncode}. "
                    f"{detail}"
                )

            result_path = output_dir / "results.json"
            if not result_path.exists():
                raise SandboxExecutionError(
                    "Sandbox finished without producing /output/results.json."
                )

            scored = json.loads(result_path.read_text(encoding="utf-8"))

    return resolved, scored


def run_public_examples_sandbox(
    problems: dict[str, dict[str, Any]],
    solutions: dict[str, str],
    *,
    backend: str = "auto",
    timeout_s: float = 120.0,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Run only examples visible in the benchmark prompt inside the container."""
    resolved = resolve_backend(backend)

    with tempfile.TemporaryDirectory(prefix="heteroagent-public-input-") as in_tmp:
        with tempfile.TemporaryDirectory(prefix="heteroagent-public-output-") as out_tmp:
            input_dir = Path(in_tmp).resolve()
            output_dir = Path(out_tmp).resolve()
            _write_bundle(input_dir, problems, solutions)

            if resolved == "docker":
                command = _docker_command(input_dir, output_dir, None, "public")
            else:
                command = _apple_command(input_dir, output_dir, None, "public")

            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxExecutionError(
                    f"Public-example sandbox exceeded {timeout_s:.0f} seconds."
                ) from exc

            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "No error output."
                raise SandboxExecutionError(
                    f"Public-example sandbox failed with exit code {result.returncode}. {detail}"
                )

            result_path = output_dir / "results.json"
            if not result_path.exists():
                raise SandboxExecutionError(
                    "Public-example sandbox finished without producing results.json."
                )

            scored = json.loads(result_path.read_text(encoding="utf-8"))

    return resolved, scored
