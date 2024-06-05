"""
"Practical algorithms and experimentally validated incentives
for equilibrium-based fair division (A-CEEI)"
tests for algorithm 3 - Tabu Search

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""

import random

import pytest

import fairpyx
from fairpyx import Instance, divide
from fairpyx.algorithms import tabu_search


# Each student will get all the courses
def test_case1():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=500, agent_capacity_bounds=(500, 500),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1,5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance)
    for agent in range(instance.num_of_agents):
        for item in range(instance.num_of_items):
            assert (item in allocation["allocation"][f"s{agent}"])


# Each student i will get course i
def test_case2():
    utilities = [{f"s{i}": [1 if j == i - 1 else 0 for j in range(100)]} for i in range(1, 101)]
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1,
                        agents=[range(1, 101)], items=[range(1, 101)])
    allocation = divide(tabu_search, instance=instance)
    for i in range(instance.num_of_agents):
        assert (i in allocation["allocation"][f"s{i}"])


# Each student i will get course i, because student i have the highest i budget.
def test_case3():
    utilities = [{"{}".format(i): list(range(100, 0, -1))} for i in range(1, 101)]
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1,
                        agents=[range(1, 101)], items=[range(1, 101)])
    b0 = list(range(100, 0, -1))
    allocation = divide(tabu_search, instance=instance, *b0)
    for i in range(instance.num_of_agents):
        assert (i in allocation["allocation"][f"s{i}"])


# Each student will get his 3 favorite courses
def test_case4():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=300, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1,5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance)
    for agent in range(instance.num_of_agents):
        agent_valuations = instance.valuations[agent]  # Get valuations for the agent
        allocated_items = allocation["allocation"][f"s{agent}"]

        # Sort items based on valuations and get the top 3
        top_items = sorted(enumerate(agent_valuations), key=lambda x: x[1], reverse=True)[:3]
        top_items_indices = [item[0] for item in top_items]

        # Check that every agent gets their top 3 items based on valuations
        for item_index in top_items_indices:
            assert item_index in allocated_items


# Checking if the values that the function returns are correct
def test_case5():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=300, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1,5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance)
    fairpyx.validate_allocation(instance, allocation, title="validate Algorithm 1")


if __name__ == "__main__":
    # pytest.main(["-v", __file__])
    import doctest
    doctest.testmod()
