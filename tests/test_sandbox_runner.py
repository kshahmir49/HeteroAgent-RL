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
