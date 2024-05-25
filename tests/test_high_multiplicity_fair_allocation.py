"""
Tests for the high-multiplicity-fair-allocation algorithm.

Article Title: High-Multiplicity Fair Allocation Made More Practical
Article URL: https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p260.pdf

Algorithm Name: High Multiplicity Fair Allocation
Algorithm Description: This algorithm finds an allocation maximizing the sum of utilities
                         for given instance with envy-freeness and Pareto-optimality constraints if exists.

Programmers: Naor Ladani and Elor Israeli
Since : 2024-05
"""

import pytest

from fairpyx import Instance, AllocationBuilder
import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES=10

def test_feasibility():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=70, num_of_items=10, normalized_sum_of_values=1000,
            agent_capacity_bounds=[2,6], 
            item_capacity_bounds=[20,40], 
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        allocation = fairpyx.divide(fairpyx.algorithms.high_multiplicity_fair_allocation, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}")


def test_functionality_overall():
    item_capacities={"s": 4, "t": 1, "m": 3, "p": 11, "k": 1}
    max_item_capacity = sum(item_capacities.values())
    agent_capacities={"Alice": max_item_capacity, "Bob": max_item_capacity, "Carol": max_item_capacity}
    instance = Instance(
        agent_capacities=agent_capacities,
        item_capacities=item_capacities,
        valuations={"Alice": {"k": 2, "s": 1, "t": 0, "m": 2, "p": 2},
                    "Bob":   {"k": -1, "s": -3, "t": 5, "m": 4, "p": 5},
                    "Carol": {"k": 6, "s": 5, "t": 1, "m": 5, "p": 2}}
    )
    result = high_multiplicity_fair_allocation(instance)
    assert result.get_allocation() == {"Alice":  {"k": 0, "s": 0, "t": 0, "m": 6, "p": 0},
                                        "Bob":   {"k": 0, "s": 0, "t": 3, "m": 4, "p": 1},
                                        "Carol": {"k": 4, "s": 1, "t": 0, "m": 1, "p": 0}}


def test_validate_allocation_for_each_envy_free_iteration():
    """
    Test validation for each envy-free iteration.
    """
    constraints = []
    allocation = find_envy_free_allocation(constraints)

    # Check if no agent gets more than the existing multiplicity of item
    for agent, items in allocation.get_allocation().items():
        for item in items:
            assert allocation.count_item(item) <= item_capacities[item]

    # Check if all the items from all agents together aren't more than the existing multiplicity of item
    total_allocated_items = allocation.count_allocated_items()
    for item, capacity in item_capacities.items():
        assert total_allocated_items[item] <= capacity

    # Check if there is no allocation of negative item
    assert all(item_allocation >= 0 for item_allocation in total_allocated_items.values())

    # Check if items are allocated only from the list of items existing in the instance
    allocated_items = set(item for items in allocation.get_allocation().values() for item in items)
    assert all(item in item_capacities for item in allocated_items)


def test_find_envy_free_allocation():
    """
    Test find_envy_free_allocation function.
    """
    item_capacities={"s": 4, "t": 1, "m": 3, "p": 11, "k": 1}
    max_item_capacity = sum(item_capacities.values())
    agent_capacities={"Alice": max_item_capacity, "Bob": max_item_capacity, "Carol": max_item_capacity}
    instance = Instance(
        agent_capacities=agent_capacities,
        item_capacities=item_capacities,
        valuations={"Alice": {"k": 2, "s": 1, "t": 0, "m": 2, "p": 2},
                    "Bob":   {"k": -1, "s": -3, "t": 5, "m": 4, "p": 5},
                    "Carol": {"k": 6, "s": 5, "t": 1, "m": 5, "p": 2}}
    )
    allocation = find_envy_free_allocation([], instance)
    
    """
    To check this function that returns an envy-free allocation - we can check the allocation through the matrix of sum of utilities.
    Because we get the instance we can calculate the sum of utilities for each agent gives to each pack of items of all agents
    and check that no agent envies another agent.
    we would like to get a matrix like that: matrix[i][j] = 'the sum of utilities agent i give to the whole pack of items of agent j'
    Then we can check on each row that the maximum value in each row is in the [i][i] position.
    """

    # Initialize sum_utilities_matrix
    agents = list(agent_capacities.keys())
    sum_utilities_matrix = np.zeros((len(agents), len(agents)))
    
    # Calculate sum of utilities for each agent's allocation
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            sum_utilities_matrix[i, j] = sum(valuations[agent_i][item] * allocation[agent_j].get(item, 0) for item in item_capacities)
    
    # Check for envy-freeness
    envy_free = True
    for i in range(len(agents)):
        if np.any(sum_utilities_matrix[i, :] > sum_utilities_matrix[i, i]):
            envy_free = False
            break
    
    print("Sum Utilities Matrix:")
    print(sum_utilities_matrix)
    print("Is the allocation envy-free?", envy_free)
    assert envy_free, "The allocation is not envy-free!"

    """
    can be checked also by the following assert: (but not promising that it will work because its not necessary gives that specific allocation)
    assert allocation.get_allocation() == np.array([[0, 0, 0, 6, 0], [0, 0, 3, 4, 1], [4, 1, 0, 1, 0]])
    this matrix is like this:
    {"Alice":  {"k": 0, "s": 0, "t": 0, "m": 6, "p": 0},
      "Bob":   {"k": 0, "s": 0, "t": 3, "m": 4, "p": 1},
      "Carol": {"k": 4, "s": 1, "t": 0, "m": 1, "p": 0}}

    another option: (again - not promising that it will work because its not necessary gives that specific allocation)
    agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
    item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    assert find_envy_free_allocation([], instance).get_allocation() == np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]])
    """


if __name__ == "__main__":
     pytest.main(["-v",__file__])

