"""
Test that course-allocation algorithms return a feasible solution.

Programmer: OFEK, TAMAR, MORIYA ESTER
Since:  2024-03
"""

import pytest

import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES = 10
def test_small_example():
    s1 = {"c1": 50, "c2": 40, "c3": 5, "c4": 5}
    s2 = {"c1": 20, "c2": 20, "c3": 30, "c4": 30}
    s3 = {"c1": 60, "c2": 30, "c3": 1, "c4": 9}

    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 1, "c2": 1, "c3": 2, "c4": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c2', 'c3'], 's2': ['c3', 'c4'], 's3': ['c1', 'c4']}, "ERROR"

def test_big_example():
    s1 = {"c1": 50, "c2": 20, "c3": 11, "c4": 10, "c5": 9}
    s2 = {"c1": 4, "c2": 5, "c3": 7, "c4": 26, "c5": 60}
    s3 = {"c1": 20, "c2": 30, "c3": 10, "c4": 21, "c5": 19}
    s4 = {"c1": 24, "c2": 8, "c3": 25, "c4": 17, "c5": 26}
    s5 = {"c1": 5, "c2": 2, "c3": 1, "c4": 3, "c5": 90}
    s6 = {"c1": 13, "c2": 15, "c3": 49, "c4": 11, "c5": 12}
    s7 = {"c1": 3, "c2": 4, "c3": 6, "c4": 70, "c5": 19}

    instance = fairpyx.Instance(
        agent_capacities={"s1": 4, "s2": 4, "s3": 4, "s4": 4, "s5": 4, "s6": 4, "s7": 4},
        item_capacities={"c1": 6, "c2": 6, "c3": 7, "c4": 3, "c5": 6},
        valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1', 'c2', 'c3'],
                                                                                  's2': ['c2', 'c3', 'c4', 'c5'],
                                                                                  's3': ['c1', 'c2', 'c4', 'c5'],
                                                                                  's4': ['c1', 'c2', 'c3', 'c5'],
                                                                                  's5': ['c1', 'c2', 'c3', 'c5'],
                                                                                  's6': ['c1', 'c2', 'c3', 'c5'],
                                                                                  's7': ['c1', 'c3', 'c4', 'c5']}, "ERROR"

def test_small_number_2():
    s1 = {"c1": 30, "c2": 35, "c3": 36}
    s2 = {"c1": 10, "c2": 80, "c3": 11}
    s3 = {"c1": 34, "c2": 32, "c3": 35}

    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 2, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c3'],
                                                                                  's2': ['c2'],
                                                                                  's3': ['c1']}, "ERROR"

def test_number_of_courses_is_not_optimal():
    s1 = {"c1": 28, "c2": 26, "c3": 19, "c4": 27}
    s2 = {"c1": 21, "c2": 20, "c3": 40, "c4": 19}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 3, "s2": 3},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2}
    )
    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1', 'c2', 'c4'], 's2': ['c2', 'c3']}, "ERROR"


def test_optimal_change_result():
    s1 = {"c1": 50, "c2": 49, "c3": 1}
    s2 = {"c1": 48, "c2": 46, "c3": 6}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1'], 's2': ['c2']}, "ERROR"


def test_sub_round_within_round():
    s1 = {"c1": 44, "c2": 39, "c3": 17}
    s2 = {"c1": 50, "c2": 45, "c3": 5}
    s3 = {"c1": 45, "c2": 40, "c3": 15}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c3'], 's2': ['c1'], 's3': ['c2']}, "ERROR"

def test_student_dont_get_k_courses():
    s1 = {"c1": 50, "c2": 10, "c3": 40}
    s2 = {"c1": 45, "c2": 30, "c3": 25}
    s3 = {"c1": 49, "c2": 15, "c3": 36}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 2, "c2": 2, "c3": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1', 'c3'], 's2': ['c2'], 's3': ['c1', 'c3']}, "ERROR"


def test_sub_round_within_sub_round():
    s1 = {"c1": 40, "c2": 10, "c3": 20, "c4": 30}
    s2 = {"c1": 50, "c2": 10, "c3": 15, "c4": 25}
    s3 = {"c1": 60, "c2": 30, "c3": 2, "c4": 8}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c3', 'c4'], 's2': ['c2', 'c3'], 's3': ['c1', 'c2']}, "ERROR"

def test_student_bids_the_same_for_different_courses():
    s1 = {"c1": 50, "c2": 50}
    s2 = {"c1": 40, "c2": 60}
    s3 = {"c1": 75, "c2": 25}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 1},
        item_capacities={"c1": 2, "c2": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1'], 's2': ['c2'], 's3': ['c1']}, "ERROR"

def test_from_the_article():
    s1 = {"c1": 400, "c2": 150, "c3": 230, "c4": 200, "c5": 20}
    s2 = {"c1": 245, "c2": 252, "c3": 256, "c4": 246, "c5": 1}
    s3 = {"c1": 243, "c2": 230, "c3": 240, "c4": 245, "c5": 42}
    s4 = {"c1": 251, "c2": 235, "c3": 242, "c4": 201, "c5": 71}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 3, "s2": 3, "s3": 3, "s4": 3},
        item_capacities={"c1": 2, "c2": 3, "c3": 3, "c4": 2, "c5": 2},
        item_conflicts={"c1": ['c4'], "c4": ['c1']},
        valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1', 'c2', 'c5'], 's2': ['c2', 'c3', 'c4'], 's3': ['c3', 'c4', 'c5'], 's4': ['c1', 'c2', 'c3']}, "ERROR"

def test_different_k_for_students():
    s1 = {"c1": 400, "c2": 200, "c3": 150, "c4": 130, "c5": 120}
    s2 = {"c1": 160, "c2": 350, "c3": 150, "c4": 140, "c5": 200}
    s3 = {"c1": 300, "c2": 250, "c3": 110, "c4": 180, "c5": 160}
    s4 = {"c1": 280, "c2": 250, "c3": 180, "c4": 130, "c5": 160}
    s5 = {"c1": 140, "c2": 180, "c3": 270, "c4": 250, "c5": 160}
    s6 = {"c1": 150, "c2": 250, "c3": 200, "c4": 260, "c5": 140}
    s7 = {"c1": 250, "c2": 180, "c3": 210, "c4": 200, "c5": 160}
    instance = fairpyx.Instance(
        agent_capacities={"s1": 1, "s2": 1, "s3": 2, "s4": 3, "s5": 3, "s6": 4, "s7": 1},
        item_capacities={"c1": 2, "c2": 5, "c3": 4, "c4": 3, "c5": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7}
    )

    assert fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance) == {'s1': ['c1'], 's2': ['c2'], 's3': ['c1', 'c2'], 's4': ['c2', 'c3', 'c5'], 's5': ['c2', 'c3', 'c4'], 's6': ['c2', 'c3', 'c4', 'c5'], 's7': ['c3']}, "ERROR"


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
        allocation = fairpyx.divide(fairpyx.algorithms.TTC_function, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, TTC_function")

if __name__ == "__main__":
    pytest.main(["-v",__file__])