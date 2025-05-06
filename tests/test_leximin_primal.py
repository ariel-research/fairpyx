import pytest
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.leximin_primal import leximin_primal


def test_empty_instance():
    # Test the behavior when no agents, no items, and no capacities are provided.
    inst = Instance(valuations={}, agent_capacities={}, item_capacities={})
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert alloc.bundles == {}


def test_invalid_no_items():
    # Test when agents exist but no items are available at all (zero capacity).
    inst = Instance(valuations={"1": {}}, agent_capacities={"1": 1}, item_capacities={})
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert alloc.bundles == {"1": {}}


def test_large_instance():
    # Stress test: 50 agents with 10 item types, all agents want everything.
    valuations = {str(i): {chr(97 + j): 1 for j in range(10)} for i in range(1, 51)}
    agent_capacities = {str(i): 1 for i in range(1, 51)}
    item_capacities = {chr(97 + j): 5 for j in range(10)}
    inst = Instance(valuations, agent_capacities, item_capacities)
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    # Check output is in the expected structure
    assert isinstance(alloc.bundles, dict)
    assert all(isinstance(bundle, dict) for bundle in alloc.bundles.values())


def test_compare_naive_matching():
    # Compare to a simple perfect match allocation: two agents, two items.
    inst = Instance(
        valuations={"1": {"a": 1}, "2": {"b": 1}},
        agent_capacities={"1": 1, "2": 1},
        item_capacities={"a": 1, "b": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    # Each agent should get their only desired item
    assert set(alloc.bundles.keys()) == {"1", "2"}
    assert all(len(b) <= 1 for b in alloc.bundles.values())


def test_fairness_condition():
    # Test that the probabilities are nearly equal when agents want the same thing.
    inst = Instance(
        valuations={"1": {"a": 1}, "2": {"a": 1}},
        agent_capacities={"1": 1, "2": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    # Verify that both agents received approximately the same probability
    probs = [sum(bundle.values()) for bundle in alloc.bundles.values()]
    assert abs(probs[0] - probs[1]) < 1e-6


def test_partial_satisfiability():
    # Only 2 units of item are available for 5 agents. Only 2 items can be assigned.
    inst = Instance(
        valuations={str(i): {"a": 1} for i in range(1, 6)},
        agent_capacities={str(i): 1 for i in range(1, 6)},
        item_capacities={"a": 2}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    total = sum(len(b) for b in alloc.bundles.values())
    assert total <= 2


def test_agents_with_no_compatible_items():
    # Agent 2 cannot receive any item because they have no valuation
    inst = Instance(
        valuations={"1": {"a": 1}, "2": {}},
        agent_capacities={"1": 1, "2": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert alloc.bundles["2"] == {}


# ========= My Extension of tests to check in part C ============

def test_duplicate_preferences():
    # Agent has the same item listed multiple times — this should have no effect
    inst = Instance(
        valuations={"1": {"a": 1, "a": 1}},  # duplicate keys ignored in Python dict
        agent_capacities={"1": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert sum(len(b) for b in alloc.bundles.values()) <= 1


def test_agent_with_zero_capacity():
    # Agent exists but cannot be assigned any items due to zero capacity
    inst = Instance(
        valuations={"1": {"a": 1}},
        agent_capacities={"1": 0},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert alloc.bundles["1"] == {}


def test_item_with_zero_capacity():
    # Item appears in valuations but has zero available units
    inst = Instance(
        valuations={"1": {"a": 1}},
        agent_capacities={"1": 1},
        item_capacities={"a": 0}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    assert sum(len(b) for b in alloc.bundles.values()) == 0


def test_unbalanced_demand_and_supply():
    # Total demand exceeds total supply, but some agents should still get items
    inst = Instance(
        valuations={str(i): {"a": 1} for i in range(1, 6)},
        agent_capacities={str(i): 1 for i in range(1, 6)},
        item_capacities={"a": 3}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    total = sum(len(b) for b in alloc.bundles.values())
    assert total == 3


def test_all_agents_same_preferences():
    # All agents want the same single item — should be split fairly
    inst = Instance(
        valuations={str(i): {"x": 1} for i in range(1, 5)},
        agent_capacities={str(i): 1 for i in range(1, 5)},
        item_capacities={"x": 2}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    total = sum(len(b) for b in alloc.bundles.values())
    assert total <= 2  # Only 2 units can be allocated

    # All bundles should contain either 0 or 1 item max
    assert all(len(b) <= 1 for b in alloc.bundles.values())
