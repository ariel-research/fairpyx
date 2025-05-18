"""
An implementation of the algorithm in:
"Leximin Allocations in the Real World", by D. Kurokawa, A. D. Procaccia, and N. Shah (2018), https://doi.org/10.1145/3274641

Programmer: Lior Trachtman
Date: 2025-05-05
"""

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

def test_example_1_simple_allocations():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"a": 1, "b": 1}},  # F1, F2, F3
        agent_capacities={1: 1, 2: 1, 3: 1}, # d1 = d2 = d3 = 1
        item_capacities={"a": 1, "b": 1}  # Ca = Cb = 1
    )
    alloc = AllocationBuilder(instance) # Initialize allocator
    leximin_primal(alloc) # Execute algorithm

    # Expected allocations: all possible deterministic allocations that the randomized result may include.
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {}}, 1 / 3),      # Agents 1 and 2 receive facilities, agent 3 gets nothing
        ({1: {"a": 1}, 3: {"b": 1}, 2: {}}, 1 / 3),      # Agents 1 and 3 receive facilities, agent 2 gets nothing
        ({3: {"a": 1}, 2: {"b": 1}, 1: {}}, 1 / 3)       # Agents 2 and 3 receive facilities, agent 1 gets nothing
    ]

    # Actual allocations returned by the algorithm
    actual_allocations_with_prob = alloc.distribution  # list of (dict, probability)

    # Check that all expected allocations appear with approximately correct probabilities
    for expected_alloc, expected_prob in expected_allocations_with_prob:
        found = False
        for actual_alloc, actual_prob in actual_allocations_with_prob:
            if actual_alloc == expected_alloc:
                found = True
                assert abs(actual_prob - expected_prob) < 1e-6, (
                    f"Allocation {actual_alloc} has wrong probability: got {actual_prob}, expected {expected_prob}"
                )
                break
        assert found, f"Missing expected allocation: {expected_alloc}"

def test_example_2_with_all_branches_allocations():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}, 4: {"a": 1, "b": 1}},  # F1, F2, F3, F4
        agent_capacities={1: 1, 2: 1, 3: 1, 4: 1},  # d1 = d2 = d3 = d4 = 1
        item_capacities={"a": 1, "b": 2}  # Ca = 1, Cb = 2
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    # Expected allocations: all deterministic allocations that could appear in the randomized distribution.
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}, 4: {}}, 1 / 3),      # Agents 1, 2, 3 receive facilities, 4 gets nothing
        ({4: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}, 1: {}}, 1 / 3),      # Agents 2, 3, 4 receive facilities, 1 gets nothing
        ({1: {"a": 1}, 4: {"b": 1}, 2: {"b": 1}, 3: {}}, 1 / 6),      # Agents 1, 2, 4 receive facilities, 3 gets nothing
        ({1: {"a": 1}, 3: {"b": 1}, 4: {"b": 1}, 2: {}}, 1 / 6),      # Agents 1, 3, 4 receive facilities, 2 gets nothing
    ]

    # Actual allocations returned by the algorithm
    actual_allocations_with_prob = alloc.distribution  # list of (dict, probability)

    # Check that all expected allocations appear in the distribution with correct probabilities
    for expected_alloc, expected_prob in expected_allocations_with_prob:
        found = False
        for actual_alloc, actual_prob in actual_allocations_with_prob:
            if actual_alloc == expected_alloc:
                found = True
                assert abs(actual_prob - expected_prob) < 1e-6, (
                    f"Allocation {actual_alloc} has wrong probability: got {actual_prob}, expected {expected_prob}"
                )
                break
        assert found, f"Missing expected allocation: {expected_alloc}"


def test_example_3_perfect_allocation():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}},  # F1, F2, F3
        agent_capacities={1: 1, 2: 1, 3: 1},  # d1 = d2 = d3 = 1
        item_capacities={"a": 1, "b": 2}  # Ca = 1, Cb = 2
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    # Expected allocations: both b facilities are used and each agent receives one facility
    # Only one deterministic allocation exists with probability 1.0
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}}, 1.0),  # Agent 1 gets a, agents 2 and 3 get b
    ]

    # Actual allocations returned by the algorithm
    actual_allocations_with_prob = alloc.distribution  # list of (dict, probability)

    # Check that all expected allocations appear with approximately correct probabilities
    for expected_alloc, expected_prob in expected_allocations_with_prob:
        found = False
        for actual_alloc, actual_prob in actual_allocations_with_prob:
            if actual_alloc == expected_alloc:
                found = True
                assert abs(actual_prob - expected_prob) < 1e-6, (
                    f"Allocation {actual_alloc} has wrong probability: got {actual_prob}, expected {expected_prob}"
                )
                break
        assert found, f"Missing expected allocation: {expected_alloc}"


def test_example_4_bad_input_allocation():
    instance = Instance(
        valuations={i: {"a": 1} for i in range(1, 11)},  # All agents want facility "a"
        agent_capacities={i: 1 for i in range(1, 11)},  # di = 1 for all i
        item_capacities={"a": 1}  # Only one unit of "a" available
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    # Expected allocations: each agent gets facility "a" in a separate allocation, others get nothing
    expected_allocations_with_prob = [
        ({1: {"a": 1}, **{i: {} for i in range(2, 11)}}, 1 / 10),
        ({2: {"a": 1}, **{i: {} for i in list(range(1, 2)) + list(range(3, 11))}}, 1 / 10),
        ({3: {"a": 1}, **{i: {} for i in list(range(1, 3)) + list(range(4, 11))}}, 1 / 10),
        ({4: {"a": 1}, **{i: {} for i in list(range(1, 4)) + list(range(5, 11))}}, 1 / 10),
        ({5: {"a": 1}, **{i: {} for i in list(range(1, 5)) + list(range(6, 11))}}, 1 / 10),
        ({6: {"a": 1}, **{i: {} for i in list(range(1, 6)) + list(range(7, 11))}}, 1 / 10),
        ({7: {"a": 1}, **{i: {} for i in list(range(1, 7)) + list(range(8, 11))}}, 1 / 10),
        ({8: {"a": 1}, **{i: {} for i in list(range(1, 8)) + list(range(9, 11))}}, 1 / 10),
        ({9: {"a": 1}, **{i: {} for i in list(range(1, 9)) + [10]}}, 1 / 10),
        ({10: {"a": 1}, **{i: {} for i in range(1, 10)}}, 1 / 10),
    ]

    # Actual allocations returned by the algorithm
    actual_allocations_with_prob = alloc.distribution  # list of (dict, probability)

    # Verify that all expected allocations appear with correct probabilities
    for expected_alloc, expected_prob in expected_allocations_with_prob:
        found = False
        for actual_alloc, actual_prob in actual_allocations_with_prob:
            if actual_alloc == expected_alloc:
                found = True
                assert abs(actual_prob - expected_prob) < 1e-6, (
                    f"Allocation {actual_alloc} has wrong probability: got {actual_prob}, expected {expected_prob}"
                )
                break
        assert found, f"Missing expected allocation: {expected_alloc}"
