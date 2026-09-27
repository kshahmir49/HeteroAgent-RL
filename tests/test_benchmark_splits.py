from heteroagent_rl.benchmarks.splits import SplitConfig, partition_task_ids, select_split


def test_partitions_are_deterministic_disjoint_and_complete():
    ids = [f"HumanEval/{i}" for i in range(25)]
    config = SplitConfig(seed="fixed")

    first = partition_task_ids(ids, config=config)
    second = partition_task_ids(ids, config=config)

    assert first == second
    assert len(first["train"]) == 15
    assert len(first["dev"]) == 5
    assert len(first["test"]) == 5

    train = set(first["train"])
    dev = set(first["dev"])
    test = set(first["test"])

    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)
    assert train | dev | test == set(ids)


def test_select_all_preserves_input_order():
    ids = ["HumanEval/2", "HumanEval/1"]
    assert select_split(ids, "all") == ids
