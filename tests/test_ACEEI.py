"""
"Practical algorithms and experimentally validated incentives
for equilibrium-based fair division (A-CEEI)"
tests for algorithm 1 - ACEEI

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""

import random
import pytest
import logging
import fairpyx
from fairpyx import Instance, divide
# from fairpyx.algorithms import ACEEI
# from fairpyx.algorithms.linear_program import optimize_model
from fairpyx.algorithms.ACEEI.ACEEI import EFTBStatus, logger, find_ACEEI_with_EFTB
import numpy as np


def random_initial_budgets(num):
    return {f"s{key}": random.randint(10, 20) for key in range(1, num + 1)}


random_value = random.uniform(0.1, 2)
random_t = random.choice(list(EFTBStatus))


# Each student will get all the courses
def test_case_1():
    instance = Instance.random_uniform(num_of_agents=200, num_of_items=6, agent_capacity_bounds=(6, 6),
                                       item_capacity_bounds=(400, 400), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(1, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets(instance.num_of_agents),
                        delta=random_value, epsilon=random_value, t=random_t)
    for agent in instance.agents:
        for item in instance.items:
            assert (item in allocation[agent])


# Each student i will get course i
def test_case_2():
    utilities = {f"s{i}": {f"c{j}": 1 if j == i else 0 for j in range(1, 101)} for i in range(1, 101)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets(100),
                        delta=random_value, epsilon=random_value, t=random_t)
    for i in range(1, 101):
        assert (f"c{i}" in allocation[f"s{i}"])


# Each student i will get course i, because student i have the highest i budget.
# def test_case_3():
#     utilities = {f"s{i}": {f"c{101 - j}": j for j in range(100, 0, -1)} for i in range(1, 101)}
#     instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
#     initial_budgets = {f"s{key}": (101 - key) for key in range(1, 101)}
#     allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
#                         delta=0.5, epsilon=0.5, t=EFTBStatus.NO_EF_TB)
#     for i in range(1, 101):
#         assert (f"c{i}" in allocation[f"s{i}"])


def test_case__3_mini():
    utilities = {f"s{i}": {f"c{44 - j}": j for j in range(43, 0, -1)} for i in range(1, 44)}
    instance = Instance(valuations=utilities, agent_capacities=1, item_capacities=1)
    initial_budgets = {f"s{key}": (44 - key) for key in range(1, 44)}
    allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
                        delta=0.5, epsilon=0.5, t=EFTBStatus.EF_TB)
    for i in range(1, 44):
        assert (f"c{i}" in allocation[f"s{i}"])


# Each student will get his 3 favorite courses
#
def test_case_4():
    instance = Instance.random_uniform(num_of_agents=200, num_of_items=6, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets(instance.num_of_agents),
                        delta=random_value, epsilon=random_value, t=random_t)

    # Checking if each student receives the 3 courses with the highest valuation
    for student, allocated_courses in allocation.items():
        # Getting the agent's ranking of items and selecting the top 3
        top_3_courses = list(instance.agent_ranking(student))[:3]
        # Asserting if the allocated courses for the student are within their top 3 favorite courses
        assert all(course in top_3_courses for course in allocated_courses)


# def test_case_4_mini():
#     instance = Instance.random_uniform(num_of_agents=10, num_of_items=30, agent_capacity_bounds=(3, 3),
#                                        item_capacity_bounds=(20, 20), item_base_value_bounds=(1, 5),
#                                        item_subjective_ratio_bounds=(0.5, 1.5),
#                                        normalized_sum_of_values=1000)
#     allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets,
#                         delta=random_value, epsilon=random_value, t=random_t)
#     # Checking if each student receives the 3 courses with the highest valuation
#     for student, allocated_courses in allocation.items():
#         # Getting the agent's ranking of items and selecting the top 3
#         top_3_courses = list(instance.agent_ranking(student))[:3]
#         # Asserting if the allocated courses for the student are within their top 3 favorite courses
#         assert all(course in top_3_courses for course in allocated_courses)


# Checking if the values that the function returns are correct
def test_case_5():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=6, agent_capacity_bounds=(3, 3),
                                       item_capacity_bounds=(200, 200), item_base_value_bounds=(1, 5),
                                       item_subjective_ratio_bounds=(0.5, 1.5),
                                       normalized_sum_of_values=1000)
    allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets(instance.num_of_agents),
                        delta=random_value, epsilon=random_value, t=random_t)
    fairpyx.validate_allocation(instance, allocation, title="validate Algorithm 1")


# def test_case_5_mini():
#     instance = Instance.random_uniform(num_of_agents=10, num_of_items=30, agent_capacity_bounds=(3, 3),
#                                        item_capacity_bounds=(20, 20), item_base_value_bounds=(1, 5),
#                                        item_subjective_ratio_bounds=(0.5, 1.5),
#                                        normalized_sum_of_values=1000)
#     allocation = divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=random_initial_budgets,
#                         delta=random_value, epsilon=random_value, t=random_t)
#     fairpyx.validate_allocation(instance, allocation, title="validate Algorithm 1")


if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.INFO)
    pytest.main(["-v", __file__])
