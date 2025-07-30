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


# Helper function to check the allocation distributions from the algorithm
def check_allocation_distribution(expected_allocs_with_prob, actual_allocs_with_prob, tolerance=1e-6):
    """
    Checks that every expected allocation is present in the actual output,
    and that their assigned probabilities are close within the given tolerance.

    Args:
        expected_allocs_with_prob: list of (allocation_dict, probability) pairs representing
            the expected allocations with their probabilities.
        actual_allocs_with_prob: list of (allocation_dict, probability) pairs
            from the algorithm output.
        tolerance: float, maximum allowed absolute difference between probabilities.

    Raises:
        AssertionError if any expected allocation is missing or probability mismatch.
    """
    # Normalize allocations for consistent comparison
    def normalize_allocation(alloc, agents):
        norm = {}
        for agent in agents:
            # Support both int and str keys for agents
            bundle = alloc.get(agent, alloc.get(str(agent), {}))
            if isinstance(bundle, dict):
                norm[int(agent)] = {str(k): 1 for k in bundle.keys()}
            elif isinstance(bundle, (list, set)):
                norm[int(agent)] = {str(item): 1 for item in bundle}
            else:
                norm[int(agent)] = {}
        return norm

    # Extract all agents appearing in expected and actual allocations to normalize properly
    agents = sorted(set(
        int(agent)
        for alloc_dict, _ in actual_allocs_with_prob + expected_allocs_with_prob
        for agent in alloc_dict.keys()
    ))

    # Check that each expected allocation appears in the actual allocations
    for expected_alloc, expected_prob in expected_allocs_with_prob:
        found = False
        expected_norm = normalize_allocation(expected_alloc, agents)
        for actual_alloc, actual_prob in actual_allocs_with_prob:
            actual_norm = normalize_allocation(actual_alloc, agents)
            if actual_norm == expected_norm:
                found = True
                # Assert probability close enough to expected
                assert abs(actual_prob - expected_prob) < tolerance, (
                    f"Allocation {actual_alloc} has wrong probability: got {actual_prob}, expected {expected_prob}"
                )
                break
        # Fail if expected allocation is missing
        assert found, f"Missing expected allocation: {expected_alloc}"
    return True



def test_empty_instance():
    try:
        inst = Instance(valuations={}, agent_capacities={}, item_capacities={})
    except (StopIteration, AssertionError):
        return  # Skip the test gracefully

    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    print("test_empty_instance:")
    print("Bundles:", alloc.bundles)
    assert alloc.bundles == {}


def test_invalid_no_items():
    """
    Test case where the agent exists but there are no items at all.
    This causes Instance construction to fail internally due to empty item set.
    """
    try:
        inst = Instance(valuations={"1": {}}, agent_capacities={"1": 1}, item_capacities={})
    except (StopIteration, AssertionError):
        return  # Skip the test: instance is not constructible with no items

    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    print("test_invalid_no_items:")
    print("Bundles:", alloc.bundles)
    assert alloc.bundles == {"1": {}}


def test_agents_with_no_compatible_items():
    # Agent 2 cannot receive any item because they have no valuation
    inst = Instance(
        valuations={"1": {"a": 1}, "2": {}},
        agent_capacities={"1": 1, "2": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)

    bundle = alloc.bundles["2"]
    print("test_agents_with_no_compatible_items:")
    print("Bundles:", alloc.bundles)
    print("Distribution:", alloc.distribution)
    assert not bundle, f"Expected empty bundle, got: {bundle}"

def test_duplicate_preferences():
    # Agent has the same item listed multiple times — this should have no effect
    inst = Instance(
        valuations={"1": {"a": 1, "a": 1}},  # duplicate keys ignored in Python dict
        agent_capacities={"1": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    print("test_duplicate_preferences:")
    print("Bundles:", alloc.bundles)
    print("Distribution:", alloc.distribution)
    assert sum(len(b) for b in alloc.bundles.values()) <= 1

def test_agent_with_zero_capacity():
    # Agent 1 has a positive valuation but zero capacity and shouldn't receive anything.
    # We verify that the item "a" was assigned to someone else (if applicable).

    inst = Instance(
        valuations={"1": {"a": 1}, "2": {"a": 1}},
        agent_capacities={"1": 0, "2": 1},
        item_capacities={"a": 1}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    print("test_agent_with_zero_capacity:")
    print("Bundles:", alloc.bundles)
    print("Distribution:")

    # Agent 1 should receive nothing due to zero capacity
    assert not alloc.bundles["1"], f"Expected empty bundle for agent 1, got: {alloc.bundles['1']}"

    # At least one allocation should exist with agent 2 receiving "a"
    found = False
    for bundle, prob in alloc.distribution:
        if bundle.get("2") and "a" in bundle["2"]:
            print(f"{bundle} with probability {prob}")
            found = True
            break
    assert found, "Expected item 'a' to be allocated to agent 2 in at least one outcome"


def test_item_with_zero_capacity():
    # Item appears in valuations but has zero available units
    inst = Instance(
        valuations={"1": {"a": 1}},
        agent_capacities={"1": 1},
        item_capacities={"a": 0}
    )
    alloc = AllocationBuilder(inst)
    leximin_primal(alloc)
    print("test_item_with_zero_capacity:")
    print("Bundles:", alloc.bundles)
    assert sum(len(b) for b in alloc.bundles.values()) == 0


def test_example_1_simple_allocations():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"a": 1, "b": 1}},  # F1, F2, F3
        agent_capacities={1: 1, 2: 1, 3: 1}, # d1 = d2 = d3 = 1
        item_capacities={"a": 1, "b": 1}  # Ca = Cb = 1
    )
    alloc = AllocationBuilder(instance) # Initialize allocator
    leximin_primal(alloc) # Execute algorithm

    actual_allocations_with_prob = alloc.distribution  # list of (dict, probability)

    # Print all actual allocations for debug visibility
    print("test_example_1_simple_allocations:")
    for i, (alloc_dict, prob) in enumerate(actual_allocations_with_prob, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

    # Expected allocations: all possible deterministic allocations that the randomized result may include.
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {}}, 1 / 3),      # Agents 1 and 2 receive facilities, agent 3 gets nothing
        ({1: {"a": 1}, 3: {"b": 1}, 2: {}}, 1 / 3),      # Agents 1 and 3 receive facilities, agent 2 gets nothing
        ({3: {"a": 1}, 2: {"b": 1}, 1: {}}, 1 / 3)       # Agents 2 and 3 receive facilities, agent 1 gets nothing
    ]

    # Assert using the helper function
    assert check_allocation_distribution(expected_allocations_with_prob, alloc.distribution)


def test_example_2_with_all_branches_allocations():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}, 4: {"a": 1, "b": 1}},  # F1, F2, F3, F4
        agent_capacities={1: 1, 2: 1, 3: 1, 4: 1},  # d1 = d2 = d3 = d4 = 1
        item_capacities={"a": 1, "b": 2}  # Ca = 1, Cb = 2
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    print("test_example_2_with_all_branches_allocations:")
    for i, (alloc_dict, prob) in enumerate(alloc.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

    # Expected allocations: all deterministic allocations that could appear in the randomized distribution.
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {}, 4: {}}, 0.5),
        ({3: {"b": 1}, 4: {"a": 1}, 1: {}, 2: {}}, 0.5),
    ]

    # Assert using the helper function
    assert check_allocation_distribution(expected_allocations_with_prob, alloc.distribution)




def test_example_3_perfect_allocation():
    instance = Instance(
        valuations={1: {"a": 1}, 2: {"b": 1}, 3: {"b": 1}},  # F1, F2, F3
        agent_capacities={1: 1, 2: 1, 3: 1},  # d1 = d2 = d3 = 1
        item_capacities={"a": 1, "b": 2}  # Ca = 1, Cb = 2
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    print("test_example_3_perfect_allocation:")
    for i, (alloc_dict, prob) in enumerate(alloc.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

    # Expected allocations: both b facilities are used and each agent receives one facility
    expected_allocations_with_prob = [
        ({1: {"a": 1}, 2: {"b": 1}, 3: {}}, 0.5),
        ({1: {"a": 1}, 2: {}, 3: {"b": 1}}, 0.5),
    ]

    # Assert using the helper function
    assert check_allocation_distribution(expected_allocations_with_prob, alloc.distribution)



def test_example_4_bad_input_allocation():
    instance = Instance(
        valuations={i: {"a": 1} for i in range(1, 11)},  # All agents want facility "a"
        agent_capacities={i: 1 for i in range(1, 11)},  # di = 1 for all i
        item_capacities={"a": 1}  # Only one unit of "a" available
    )
    alloc = AllocationBuilder(instance)  # Initialize allocator
    leximin_primal(alloc)  # Execute algorithm

    print("test_example_4_bad_input_allocation:")
    for i, (alloc_dict, prob) in enumerate(alloc.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in sorted(alloc_dict.items()):
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

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

    # Assert using the helper function
    assert check_allocation_distribution(expected_allocations_with_prob, alloc.distribution)


def test_example_inputs():
    # === Test Case 1 ===
    instance1 = Instance(
        valuations={
            1: {"a": 1, "b": 1},
            2: {"a": 1},
            3: {"a": 1, "b": 1}
        },
        agent_capacities={1: 3, 2: 1, 3: 1},
        item_capacities={"a": 3, "b": 2}
    )
    alloc1 = AllocationBuilder(instance1)
    leximin_primal(alloc1)

    print("\n--- Test Case 1 ---")
    for i, (alloc_dict, prob) in enumerate(alloc1.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

    # === Test Case 2 ===
    instance2 = Instance(
        valuations={
            1: {"a": 1, "b": 1},
            2: {"b": 1},
            3: {"a": 1}
        },
        agent_capacities={1: 2, 2: 3, 3: 3},
        item_capacities={"a": 1, "b": 2}
    )
    alloc2 = AllocationBuilder(instance2)
    leximin_primal(alloc2)

    print("\n--- Test Case 2 ---")
    for i, (alloc_dict, prob) in enumerate(alloc2.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()

    # === Test Case 3 ===
    instance3 = Instance(
        valuations={
            1: {"a": 1},
            2: {"a": 1}
        },
        agent_capacities={1: 3, 2: 3},
        item_capacities={"a": 3, "b": 3}
    )
    alloc3 = AllocationBuilder(instance3)
    leximin_primal(alloc3)

    print("\n--- Test Case 3 ---")
    for i, (alloc_dict, prob) in enumerate(alloc3.distribution, 1):
        print(f"--- Allocation {i} ---")
        print(f"Probability: {prob:.4f}")
        for agent, items in alloc_dict.items():
            if items:
                print(f"Agent {agent} receives: {list(items)}")
            else:
                print(f"Agent {agent} receives nothing.")
        print()
