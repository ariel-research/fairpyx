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
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance) == {'s1': ['c2'], 's2': ['c1']}, "ERROR"


def test_student_get_k_courses():  # in ttc a student didn't get k cources
    s1 = {"c1": 50, "c2": 10, "c3": 40}   # rank: {"c1": 3, "c2": 1, "c3": 2}    his rank:11    our:13
    s2 = {"c1": 45, "c2": 30, "c3": 25}   # rank: {"c1": 3, "c2": 2, "c3": 1}    his bids:184   our:210
    s3 = {"c1": 49, "c2": 15, "c3": 36}   # rank: {"c1": 3, "c2": 1, "c3": 2}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 2, "c2": 2, "c3": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance) == {'s1': ['c2', 'c3'], 's2': ['c1', 'c2'], 's3': ['c1', 'c3']}, "ERROR"


def test_optimal_improve_cardinal_and_ordinal_results():
    s1 = {"c1": 50, "c2": 30, "c3": 20}  # rank: {"c1": 3, "c2": 2, "c3": 1}        round1: rank=9, bids=160
    s2 = {"c1": 40, "c2": 50, "c3": 10}  # rank: {"c1": 2, "c2": 3, "c3": 1}        round2: rank=4, bids=60
    s3 = {"c1": 60, "c2": 10, "c3": 30}  # rank: {"c1": 3, "c2": 1, "c3": 2}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 2, "c2": 3, "c3": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance) == {'s1': ['c1', 'c2'], 's2': ['c2'],  's3': ['c1', 'c3']}, "ERROR"


def test_sub_round_within_sub_round():
    s1 = {"c1": 40, "c2": 10, "c3": 20, "c4": 30}   # rank: {"c1": 4, "c2": 1, "c3": 2, "c4: 3"}        round1: rank=10, bids=110
    s2 = {"c1": 50, "c2": 10, "c3": 15, "c4": 25}   # rank: {"c1": 4, "c2": 1, "c3": 2, "c4: 3"}        round2: rank=4, bids=35
    s3 = {"c1": 60, "c2": 30, "c3": 2, "c4": 8}     # rank: {"c1": 4, "c2": 3, "c3": 1, "c4: 2"}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance) == {'s1': ['c3', 'c4'], 's2': ['c1', 'c3'], 's3': ['c2']}, "ERROR"

def test_optimal_cardinal_utility():
    s1 = {"c1": 30, "c2": 35, "c3": 35}
    s2 = {"c1": 10, "c2": 80, "c3": 10}
    s3 = {"c1": 34, "c2": 32, "c3": 34}

    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 2, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2, "s3":s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance) == {'s1': ['c3'],
                                                                                  's2': ['c2'],
                                                                                  's3': ['c1']}, "ERROR"

def test_random():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=20, num_of_items=5, normalized_sum_of_values=100,
            agent_capacity_bounds=[2, 6],
            item_capacity_bounds=[20, 40],
            item_base_value_bounds=[1, 1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
        )
        allocation = fairpyx.divide(fairpyx.algorithms.TTC_O_function, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, TTC_O_function")

if __name__ == "__main__":
    pytest.main(["-v",__file__])
