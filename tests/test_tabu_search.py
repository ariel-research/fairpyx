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
from fairpyx.algorithms.ACEEI_algorithms.tabu_search import tabu_search

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



# Each student i will get course i, because student i have the highest i budget.
def test_case2():
    num_of_agents = 20
    utilities = {f"s{i}": {f"c{num_of_agents + 1 - j}": j for j in range(num_of_agents, 0, -1)} for i in
                 range(1, num_of_agents + 1)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    initial_budgets = {f"s{key}": (num_of_agents + 1 - key) for key in range(1, num_of_agents + 1)}
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=initial_budgets,
                        beta=0.1, delta={0.9})
    for i in range(1, num_of_agents + 1):
        assert (f"c{i}" in allocation[f"s{i}"])



# Each student will get his 3 favorite courses
def test_case3():
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
def test_case4():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=6, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(tabu_search, instance=instance,
                        initial_budgets=random_initial_budgets(instance.num_of_agents),
                        beta=random_beta, delta=random_delta)
    fairpyx.validate_allocation(instance, allocation, title="validate Algorithm 3")


if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.INFO)
    pytest.main(["-v", __file__])
