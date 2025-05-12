import pytest
from fairpyx import Instance, divide
from fairpyx.algorithms.markakis_psomas import algorithm1_worst_case_allocation

def test_example_1():
    instance = Instance(
        valuations={"A": {"1": 6, "2": 3, "3": 1}, "B": {"1": 2, "2": 5, "3": 5}}
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    assert set(alloc.bundles["A"]) == {"1", "3"}
    assert set(alloc.bundles["B"]) == {"2"}

def test_example_2():
    instance = Instance(
        valuations={
            "A": {"1": 7, "2": 2, "3": 1, "4": 1},
            "B": {"1": 3, "2": 6, "3": 1, "4": 2},
            "C": {"1": 2, "2": 3, "3": 5, "4": 5},
        }
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    for agent in instance.valuations:
        value = sum(instance.valuations[agent][item] for item in alloc.bundles[agent])
        assert value >= instance.worst_case_value(agent)

def test_empty_instance():
    instance = Instance(valuations={})
    with pytest.raises(ValueError):
        list(divide(algorithm=algorithm1_worst_case_allocation, instance=instance))

def test_single_agent_all_items():
    instance = Instance(valuations={"A": {"1": 10, "2": 5}})
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    assert set(alloc.bundles["A"]) == {"1", "2"}

def test_large_random_instance():
    import random
    random.seed(42)
    agents = [f"A{i}" for i in range(10)]
    items = [str(i) for i in range(15)]
    valuations = {
        agent: {item: random.randint(1, 10) for item in items}
        for agent in agents
    }
    instance = Instance(valuations)
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    for agent in agents:
        value = sum(valuations[agent][item] for item in alloc.bundles[agent])
        assert value >= instance.worst_case_value(agent)

def test_with_capacities():
    from collections import Counter
    instance = Instance(
        valuations={
            "Alice": {"c1": 8, "c2": 6, "c3": 10},
            "Bob": {"c1": 8, "c2": 10, "c3": 6},
            "Chana": {"c1": 6, "c2": 8, "c3": 10},
            "Dana": {"c1": 6, "c2": 8, "c3": 10}
        },
        agent_capacities={"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3},
        item_capacities={"c1": 2, "c2": 3, "c3": 4}
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    for agent, cap in instance.agent_capacities.items():
        assert len(alloc.bundles[agent]) <= cap
    counter = Counter(item for bundle in alloc.bundles.values() for item in bundle)
    for item, cap in instance.item_capacities.items():
        assert counter[item] <= cap
