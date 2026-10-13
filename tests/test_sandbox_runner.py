from pathlib import Path

from heteroagent_rl.sandbox.runner import _apple_command, _docker_command


def test_docker_command_has_isolation_controls():
    command = _docker_command(Path("/tmp/input"), Path("/tmp/output"), "humaneval")

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "--pids-limit" in command
    assert "/tmp/input:/input:ro" in command
    assert "/tmp/output:/output:rw" in command


def test_apple_command_has_vm_isolation_controls():
    command = _apple_command(Path("/tmp/input"), Path("/tmp/output"), "humaneval")

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--rosetta" in command
    assert "/tmp/input:/input:ro" in command
    assert "/tmp/output:/output" in command


def test_resolve_backend_retries_transient_failure(monkeypatch):
    import heteroagent_rl.sandbox.runner as runner

    calls = {"count": 0}

    def fake_available_backends():
        calls["count"] += 1
        return {
            "apple": calls["count"] >= 2,
            "docker": False,
        }

    monkeypatch.setattr(runner, "available_backends", fake_available_backends)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    assert runner.resolve_backend("auto", retries=2) == "apple"
    assert calls["count"] == 2


def test_resolve_backend_fails_after_retry_budget(monkeypatch):
    import heteroagent_rl.sandbox.runner as runner

    calls = {"count": 0}

    def fake_available_backends():
        calls["count"] += 1
        return {"apple": False, "docker": False}

    monkeypatch.setattr(runner, "available_backends", fake_available_backends)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    try:
        runner.resolve_backend("auto", retries=2)
    except runner.SandboxExecutionError:
        pass
    else:
        raise AssertionError("Expected SandboxExecutionError")

    assert calls["count"] == 3
