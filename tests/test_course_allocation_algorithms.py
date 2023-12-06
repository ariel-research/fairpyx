"""
Test that course-allocation algorithms return a feasible solution on random instances.

Programmer: Erel Segal-Halevi
Since:  2023-07
"""

import pytest

import fairpyx
import fairpyx.algorithms as crs
import numpy as np

      
def test_feasibility():
    algorithms = [
        crs.utilitarian_matching, 
        crs.iterated_maximum_matching, 
        crs.serial_dictatorship,                  # Very bad performance
        crs.round_robin, 
        crs.bidirectional_round_robin,
        crs.almost_egalitarian_allocation,
        ]
    for i in range(10):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=70, num_of_items=10, normalized_sum_of_values=1000,
            agent_capacity_bounds=[2,6], 
            item_capacity_bounds=[20,40], 
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        for algorithm in algorithms:
            allocation = fairpyx.divide(algorithm, instance=instance)
            fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, algorithm {algorithm.__name__}")


if __name__ == "__main__":
     pytest.main(["-v",__file__])

