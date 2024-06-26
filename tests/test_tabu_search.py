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
from fairpyx.algorithms.tabu_search import tabu_search

random_delta = {random.uniform(0.1, 1)}
random_beta = random.uniform(1, 100)


def random_initial_budgets(num):
    return {f"s{key}": random.uniform(1, 1 + random_beta) for key in range(1, num + 1)}


# Each student will get all the courses
def test_case1():
    instance = Instance.random_uniform(num_of_agents=200, num_of_items=6, agent_capacity_bounds=(6, 6),
                                       item_capacity_bounds=(400, 400), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(1, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=random_initial_budgets(instance.num_of_agents),
                        beta=random_beta, delta=random_delta)
    for agent in instance.agents:
        for item in instance.items:
            assert (item in allocation[agent])


# TODO: not working: 3
# Each student i will get course i
def test_case2():
    num_of_agents = 100
    utilities = {f"s{i}": {f"c{j}": 1 if j == i else 0 for j in range(1, num_of_agents)} for i in
                 range(1, num_of_agents)}
    initial_budgets = {f"s{key}": (random_beta + 1) for key in range(1, num_of_agents)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=initial_budgets,
                        beta=random_beta, delta=random_delta)
    for i in range(1, num_of_agents):
        assert (f"c{i}" in allocation[f"s{i}"])


# Each student i will get course i, because student i have the highest i budget.
def test_case3():
    utilities = {f"s{i}": {f"c{101 - j}": j for j in range(100, 0, -1)} for i in range(1, 101)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    initial_budgets = {f"s{key}": (101 - key) for key in range(1, 101)}
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=initial_budgets,
                        beta=random_beta, delta=random_delta)
    for i in range(1, 101):
        assert (f"c{i}" in allocation[f"s{i}"])


def test_case__3_mini():
    # for delta in np.linspace(0.1, 2, 20):
    #     logger.info(f"----------DELTA = {delta}---------------")
    utilities = {f"s{i}": {f"c{44 - j}": j for j in range(43, 0, -1)} for i in range(1, 44)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    initial_budgets = {f"s{key}": (44 - key) for key in range(1, 44)}
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=random_initial_budgets(instance.num_of_agents),
                        beta=random_beta, delta=random_delta)
    for i in range(1, 44):
        assert (f"c{i}" in allocation[f"s{i}"])


# Each student will get his 3 favorite courses
def test_case4():
    instance = Instance.random_uniform(num_of_agents=200, num_of_items=6, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=random_initial_budgets(instance.num_of_agents),
                        beta=random_beta, delta=random_delta)

    # Checking if each student receives the 3 courses with the highest valuation
    for student, allocated_courses in allocation.items():
        # Getting the agent's ranking of items and selecting the top 3
        top_3_courses = list(instance.agent_ranking(student))[:3]
        # Asserting if the allocated courses for the student are within their top 3 favorite courses
        assert all(course in top_3_courses for course in allocated_courses)


# Checking if the values that the function returns are correct
def test_case5():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=6, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=random_initial_budgets(instance.num_of_agents),
                        beta=random_beta, delta=random_delta)
    fairpyx.validate_allocation(instance, allocation, title="validate Algorithm 3")


if __name__ == "__main__":
    # pytest.main(["-v", __file__])
    import doctest

    doctest.testmod()
