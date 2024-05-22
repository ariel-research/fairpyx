"""
Article Title: High-Multiplicity Fair Allocation Made More Practical
Article URL: https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p260.pdf

Algorithm Name: High Multiplicity Fair Allocation
Algorithm Description: This algorithm finds an allocation maximizing the sum of utilities
                         for given instance with envy-freeness and Pareto-optimality constraints if exists.

Programmers: Naor Ladani and Elor Israeli
Since : 2024-05
"""
import cvxpy as cp
import mip as mp


from fairpyx.utils.graph_utils import many_to_many_matching_using_network_flow
from fairpyx import Instance, AllocationBuilder

import logging
logger = logging.getLogger(__name__)


def high_multiplicity_fair_allocation(alloc: AllocationBuilder):
    """
    Finds an allocation maximizing the sum of utilities for given instance with envy-freeness and Pareto-optimality constraints.

    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.


    >>> instance = Instance(
    ...     agent_capacities={"Ami": 2, "Tami": 2, "Rami": 2},
    ...     item_capacities={"Fork": 2, "Knife": 2, "Pen": 2},
    ...     valuations={
    ...         "Ami": {"Fork": 2, "Knife": 0, "Pen": 0},
    ...         "Rami": {"Fork": 0, "Knife": 1, "Pen": 1},
    ...         "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}
    ...     }
    ... )
    >>> initial_allocation = AllocationBuilder({
    ...     "Ami": [],
    ...     "Rami": [],
    ...     "Tami": []
    ... })
    >>> result = high_multiplicity_fair_allocation(initial_allocation, instance)
    >>> result.get_allocation() == {
    ...     "Ami": ["Fork", "Fork"],
    ...     "Rami": ["Pen", "Pen"],
    ...     "Tami": ["Knife", "Knife"]
    ... }
    True

    """

    ## Step 1: Find a envy-free allocation
    ## Step 2: Check if there is a Pareto-dominate allocation
    ## Step 3: If not, modify the ILP to disallow the current allocation
    ## Step 4: Repeat steps 1-3 until a Pareto-optimal allocation is found or no allocation exists


    pass


def find_envy_free_allocation(alloc: AllocationBuilder, constraints: list) -> AllocationBuilder:
    """
    Find an envy-free allocation of items to agents.
    
    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.
    - constraints (list): List of constraints for the ILP.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.

    >>> initial_allocation = AllocationBuilder({ "Ami": ["Pen", "Fork"], "Rami": ["Fork", "Pen"], "Tami": ["Knife", "Knife"] })
    >>> constraints = []  # Provide relevant constraints here
    >>> envy_free_allocation = find_envy_free_allocation(initial_allocation, constraints)
    >>> envy_free_allocation  = AllocationBuilder({ "Ami": ["Fork", "Fork"], "Rami": ["Knife", "Pen"], "Tami": ["Knife", "Pen"] })
    True
    """


def find_pareto_optimal_allocation(alloc: AllocationBuilder) -> AllocationBuilder:
    """
    Find a Pareto-optimal allocation of items to agents.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.


    >>> alloc_X = AllocationBuilder({"Ami": ["Pen", "Fork"], "Rami": ["Fork", "Pen"], "Tami": ["Knife", "Knife"]})
    >>> pareto_optimal_allocation = find_pareto_optimal_allocation(alloc_X)
    >>> pareto_optimal_allocation.get_allocation() == {"Ami": ["Fork", "Fork"], "Rami": ["Pen", "Pen"], "Tami": ["Knife", "Knife"]}
    True
    >>> alloc_X = AllocationBuilder({"Ami": ["Fork", "Fork"], "Rami": ["Pen", "Pen"], "Tami": ["Knife", "Knife"]})
    >>> pareto_optimal_allocation = find_pareto_optimal_allocation(alloc_X)
    >>> pareto_optimal_allocation.get_allocation() == None
    True
    """
    pass


def create_more_constraints_ILP(alloc_X: AllocationBuilder, alloc_Y: AllocationBuilder):
    """
    Create more constraints for the ILP to disallow the current allocation.

    Parameters:
    - alloc_X (AllocationBuilder): The current allocation of items to agents.
    - alloc_Y (AllocationBuilder): The desired allocation of items to agents.

    Returns:
    - constraints (list): List of additional constraints to disallow alloc_X and allow alloc_Y.

    >>> alloc_X = AllocationBuilder({"Ami": ["Pen", "Fork"], "Rami": ["Fork", "Pen"], "Tami": ["Knife", "Knife"]})
    >>> alloc_Y = AllocationBuilder({"Ami": ["Fork", "Fork"], "Rami": ["Pen", "Pen"], "Tami": ["Knife", "Knife"]})
    >>> constraints = create_more_constraints_ILP(alloc_X, alloc_Y)
    >>> len(constraints) == 19 # 3 agents * 3 agents * 2 items in each equation + the 9th constraint in the algorithm
    True
    """
    pass


#### MAIN

if __name__ == "__main__":
    import doctest, sys
    print("\n",doctest.testmod(), "\n")

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    from fairpyx.adaptors import divide_random_instance
    divide_random_instance(algorithm=high_multiplicity_fair_allocation, 
                           num_of_agents=10, num_of_items=4, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
                           item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
                           random_seed=1)