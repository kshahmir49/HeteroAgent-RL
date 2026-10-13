import json
from pathlib import Path

import pytest

from heteroagent_rl.rl.checkpoint import (
    append_jsonl_durable,
    atomic_write_json,
    load_jsonl_by_task,
    prepare_run_directory,
)


def test_checkpoint_round_trip_and_last_row_wins(tmp_path: Path):
    path = tmp_path / "episodes.checkpoint.jsonl"

    append_jsonl_durable(path, {"task_id": "HumanEval/1", "candidate": "a"})
    append_jsonl_durable(path, {"task_id": "HumanEval/2", "candidate": "b"})
    append_jsonl_durable(path, {"task_id": "HumanEval/1", "candidate": "c"})

    rows = load_jsonl_by_task(path)

    assert rows["HumanEval/1"]["candidate"] == "c"
    assert rows["HumanEval/2"]["candidate"] == "b"


def test_prepare_run_directory_rejects_config_mismatch(tmp_path: Path):
    run_dir = tmp_path / "run"

    checkpoint = prepare_run_directory(
        run_dir,
        run_config={"controller": "full_pipeline", "split": "dev"},
    )

    assert checkpoint.name == "episodes.checkpoint.jsonl"

    with pytest.raises(RuntimeError):
        prepare_run_directory(
            run_dir,
            run_config={"controller": "executor_only", "split": "dev"},
        )


def test_fresh_run_clears_previous_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    prepare_run_directory(
        run_dir,
        run_config={"controller": "full_pipeline"},
    )
    append_jsonl_durable(
        run_dir / "episodes.checkpoint.jsonl",
        {"task_id": "HumanEval/1"},
    )
    atomic_write_json(run_dir / "summary.json", {"status": "in_progress"})

    prepare_run_directory(
        run_dir,
        run_config={"controller": "full_pipeline"},
        fresh=True,
    )

    assert not (run_dir / "episodes.checkpoint.jsonl").exists()
    assert not (run_dir / "summary.json").exists()
    assert json.loads((run_dir / "run_config.json").read_text()) == {
        "controller": "full_pipeline"
    }


def test_loader_ignores_truncated_final_line(tmp_path: Path):
    path = tmp_path / "episodes.checkpoint.jsonl"
    path.write_text(
        '{"task_id": "HumanEval/1", "candidate": "ok"}\n'
        '{"task_id": "HumanEval/2"',
        encoding="utf-8",
    )

    rows = load_jsonl_by_task(path)

    assert list(rows) == ["HumanEval/1"]
