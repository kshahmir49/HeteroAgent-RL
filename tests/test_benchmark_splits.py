from heteroagent_rl.benchmarks.splits import (
    SplitConfig,
    clean_task_ids,
    legacy_task_ids,
    partition_task_ids,
    select_split,
)


def test_partitions_are_deterministic_disjoint_and_complete():
    ids = [f"HumanEval/{i}" for i in range(25, 164)]
    config = SplitConfig(seed="fixed")

    first = partition_task_ids(ids, config=config)
    second = partition_task_ids(ids, config=config)

    assert first == second
    assert len(first["train"]) == 90
    assert len(first["dev"]) == 24
    assert len(first["test"]) == 25

    train = set(first["train"])
    dev = set(first["dev"])
    test = set(first["test"])

    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)
    assert train | dev | test == set(ids)


def test_humaneval_legacy_tasks_are_reserved_from_clean_pool():
    ids = [f"HumanEval/{i}" for i in range(164)]

    legacy = legacy_task_ids(ids, benchmark="humaneval")
    clean = clean_task_ids(ids, benchmark="humaneval")

    assert legacy == [f"HumanEval/{i}" for i in range(25)]
    assert set(legacy).isdisjoint(clean)
    assert clean == [f"HumanEval/{i}" for i in range(25, 164)]


def test_select_clean_splits_never_include_legacy_humaneval_ids():
    ids = [f"HumanEval/{i}" for i in range(164)]
    config = SplitConfig(seed="fixed")

    for split in ("train", "dev", "test"):
        selected = select_split(
            ids,
            split,
            benchmark="humaneval",
            config=config,
        )
        assert all(int(task_id.split("/")[1]) >= 25 for task_id in selected)


def test_select_legacy_dev_returns_only_reserved_humaneval_tasks():
    ids = [f"HumanEval/{i}" for i in range(164)]

    selected = select_split(ids, "legacy_dev", benchmark="humaneval")

    assert selected == [f"HumanEval/{i}" for i in range(25)]


def test_non_humaneval_benchmarks_are_not_filtered():
    ids = [f"Mbpp/{i}" for i in range(20)]
    config = SplitConfig(seed="fixed")

    clean = clean_task_ids(ids, benchmark="mbpp")
    partitions = partition_task_ids(clean, config=config)

    assert clean == ids
    assert sum(len(values) for values in partitions.values()) == len(ids)


def test_select_all_preserves_input_order():
    ids = ["HumanEval/2", "HumanEval/1"]
    assert select_split(ids, "all", benchmark="humaneval") == ids
