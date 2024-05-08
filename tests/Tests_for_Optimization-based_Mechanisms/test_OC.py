"""
Test that course-allocation algorithms return a feasible solution.

Programmer: OFEK, TAMAR, MORIYA ESTER
Since:  2024-03
"""

import pytest

import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES=10

def test_optimal_number_of_courses():
    s1 = {"c1": 25, "c2": 25, "c3": 25, "c4": 25}
    s2 = {"c1": 20, "c2": 20, "c3": 40, "c4": 20}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 3, "s2": 3},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2}
    )
    assert fairpyx.divide(fairpyx.algorithms.OC_function, instance=instance) == {'s1': ['c1', 'c2', 'c3'], 's2': ['c2', 'c3', 'c4']}, "ERROR"

def test_optimal_change_result():
    s1 = {"c1": 50, "c2": 49, "c3": 1}
    s2 = {"c1": 48, "c2": 46, "c3": 6}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2}
    )

    assert fairpyx.divide(fairpyx.algorithms.OC_function, instance=instance) == {'s1': ['c2'], 's2': ['c1']}, "ERROR"


def test_student_bids_the_same_for_different_courses():
    s1 = {"c1": 44, "c2": 39, "c3": 17}
    s2 = {"c1": 50, "c2": 45, "c3": 5}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2},
        item_capacities={"c1": 2, "c2": 1, "c3": 2},
        valuations={"s1": s1, "s2": s2}
    )

    assert fairpyx.divide(fairpyx.algorithms.OC_function, instance=instance) == {'s1': ['c1', 'c3'], 's2': ['c1', 'c2']}, "ERROR"

def test_random():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=70, num_of_items=10, normalized_sum_of_values=1000,
            agent_capacity_bounds=[2,6],
            item_capacity_bounds=[20,40],
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        allocation = fairpyx.divide(fairpyx.algorithms.OC_function, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, OC_function")

if __name__ == "__main__":
    pytest.main(["-v",__file__])
