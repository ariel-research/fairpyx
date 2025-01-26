"""
Test that course-allocation algorithms return a feasible solution.

Programmer: OFEK, TAMAR, MORIYA ESTER
Since:  2024-03
"""

import pytest

import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES=2


def test_optimal_change_result():
    s1 = {"c1": 50, "c2": 49, "c3": 1}
    s2 = {"c1": 48, "c2": 46, "c3": 6}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2}
    )

    assert fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance) == {'s1': ['c2'], 's2': ['c1']}, "ERROR"

def test_with_two_students_one_course():
    s1 = {"c1": 10}
    s2 = {"c1": 11}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1},
        item_capacities={"c1": 1},
        valuations={"s1": s1, "s2": s2}
    )
    assert fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance) == {'s1': [], 's2': ['c1']}, "ERROR"

def test_optimal_improve_cardinal_and_ordinal_results():
    s1 = {"c1": 50, "c2": 30, "c3": 20}  # c1: 50, c2: 30, c3: 30
    s2 = {"c1": 40, "c2": 50, "c3": 10}  # c2: 50, c1: 40, c3: 10
    s3 = {"c1": 60, "c2": 11, "c3": 29}  # c1: 60, c3: 29, c2: 11
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 2, "c2": 3, "c3": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance) == {'s1': ['c1', 'c2'], 's2': ['c2', 'c3'],  's3': ['c1', 'c2']}, "ERROR"


def test_sub_round_within_sub_round():
    s1 = {"c1": 40, "c2": 10, "c3": 20, "c4": 30}       #c1:40, c4:30, c3:20, c2:10
    s2 = {"c1": 50, "c2": 10, "c3": 15, "c4": 25}       #c1:50, c4:25, c3:15, c2:10
    s3 = {"c1": 60, "c2": 30, "c3": 2, "c4": 8}         #c1:60, c2:30, c4:8, c3:2
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance) == {'s1': ['c3', 'c4'], 's2': ['c1', 'c2'], 's3': ['c2', 'c3']}, "ERROR"

def test_for_Erel():
    s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}
    s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}
    s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}
    s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}
    s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2},
        item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    )
    fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance)

def test_random():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=20, num_of_items=5, normalized_sum_of_values=1000,
            agent_capacity_bounds=[2,6],
            item_capacity_bounds=[20,40],
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        allocation = fairpyx.divide(fairpyx.algorithms.SP_O_function, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, SP_O_function")
if __name__ == "__main__":
    pytest.main(["-v",__file__])
