"""
Test the iterated-matching algorithm.

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
            agent_capacity_bounds=[2,20], 
            item_capacity_bounds=[20,40], 
            item_weight_bounds=[2,4],
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_unadjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, unadjusted")
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, adjusted")


def test_item_weights():
    instance = fairpyx.Instance(
        valuations={
            "alon":   {"c1": 300, "c2": 200, "c3": 100, "c4": 150, "c5": 150, "c6": 100},
            "ruti":   {"c1": 100, "c2": 200, "c3": 300, "c4": 150, "c5": 100, "c6": 150},
            "sigalit":{"c1": 200, "c2": 150, "c3": 250, "c4": 200, "c5": 100, "c6": 100},
            "uri":    {"c1": 150, "c2": 100, "c3": 200, "c4": 300, "c5": 150, "c6": 100},
            "ron":    {"c1": 250, "c2": 100, "c3": 150, "c4": 200, "c5": 200, "c6": 100},
        },
        agent_capacities={
            "alon": 10,     # needs 10 credit points
            "ruti": 8,      # needs 8 credit points
            "sigalit": 16,  # needs 16 credit points
            "uri": 6,       # needs 6 credit points
            "ron": 4        # needs 4 credit points
        },
        item_capacities={
            "c1": 2,
            "c2": 3,
            "c3": 1,
            "c4": 2,
            "c5": 4,
            "c6": 2
        },
        item_weights={
            "c1": 2,  # 2 credit points
            "c2": 3,  # 3 credit points
            "c3": 4,  # 4 credit points
            "c4": 2,  # 2 credit points
            "c5": 3,  # 3 credit points
            "c6": 4   # 4 credit points
        }
    )

    from fairpyx.explanations import StringsExplanationLogger
    import logging
    string_explanation_logger = StringsExplanationLogger(agents=[name for name in instance.agents], language='he')
    allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance, explanation_logger = string_explanation_logger)
    fairpyx.validate_allocation(instance, allocation, title=f"adjusted")
    with open('explanation.txt','w') as f:
        print(string_explanation_logger.map_agent_to_explanation()['sigalit'],file=f)

if __name__ == "__main__":
     pytest.main(["-vs",__file__])

