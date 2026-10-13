from __future__ import annotations

import json
import platform
import shutil
import subprocess

from heteroagent_rl.sandbox.runner import (
    APPLE_EVALPLUS_IMAGE,
    EVALPLUS_IMAGE,
    available_backends,
)


def _version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip()
    return text or None


def main() -> None:
    report = {
        "platform": platform.platform(),
        "apple_container_binary": shutil.which("container"),
        "apple_container_version": _version(["container", "--version"])
        if shutil.which("container")
        else None,
        "docker_binary": shutil.which("docker"),
        "docker_version": _version(["docker", "--version"])
        if shutil.which("docker")
        else None,
        "available": available_backends(),
        "evalplus_image": EVALPLUS_IMAGE,
        "apple_evalplus_image": APPLE_EVALPLUS_IMAGE,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
