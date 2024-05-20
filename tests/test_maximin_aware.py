"""
* edge cases: empty input, bad input
* big inputs
* big random inputs - is mma
"""

from fairpyx.algorithms.maximin_aware import alloc_by_matching, divide_and_choose_for_three

"""
Test the maximin-aware algorithms

Programmer: Sonya Rybakov
Since:  2024-05
"""

import pytest

import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES = 10


def test_feasibility_for_three():
    """tests both algorithms feasibility for 3 agents, since divide and choose is defined for 3 only"""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        numitems = np.random.randint(1, 100)
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (3, numitems)))
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.divide_and_choose_for_three, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, mma: divide and choose")
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, mma: allocation by matching for 3")


def test_feasibility_for_any():
    """tests allocation by matching feasibility for any amount of agents"""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        numitems = np.random.randint(1, 100)
        numagents = np.random.randint(2, 100)
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (numagents, numitems)))
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, mma: allocation by matching for any")


def test_random_divide_and_choose():
    """tests whether the result is MMA1"""
    pass


def test_random_alloc_by_matching():
    """tests whether the result is MMAX"""
    pass


def test_errors():
    # three agents only for div/choose
    # instance check-
    #   item constraints to only one
    #   agent constraints as the number of items
    pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])
