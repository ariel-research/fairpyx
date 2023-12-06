"""
Test the utilitarian-matching algorithm.

Programmer: Erel Segal-Halevi
Since:  2023-07
"""

import pytest

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
        allocation = fairpyx.divide(fairpyx.algorithms.utilitarian_matching, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}")


if __name__ == "__main__":
     pytest.main(["-v",__file__])

