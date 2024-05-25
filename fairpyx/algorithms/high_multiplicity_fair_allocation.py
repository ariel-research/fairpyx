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
import numpy as np


from fairpyx.utils.graph_utils import many_to_many_matching_using_network_flow
from fairpyx import Instance, AllocationBuilder

import logging
logger = logging.getLogger(__name__)


def high_multiplicity_fair_allocation(instance: Instance) -> dict:
    """
    Finds an allocation maximizing the sum of utilities for given instance with envy-freeness and Pareto-optimality constraints.

    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.


    >>> from fairpyx.adaptors import divide
    >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> divide(high_multiplicity_fair_allocation, instance=instance)
    { "Ami": ["Fork", "Fork"], "Rami": ["Pen", "Pen"], "Tami": ["Knife", "Knife"] }

    """

    ## Step 1: Find a envy-free allocation
    ## Step 2: Check if there is a Pareto-dominate allocation
    ## Step 3: If not, modify the ILP to disallow the current allocation
    ## Step 4: Repeat steps 1-3 until a Pareto-optimal allocation is found or no allocation exists


    pass

def find_envy_free_allocation(constraints: list, instance: Instance) -> np.ndarray:
    """
    Find an envy-free allocation of items to agents. 
    
    Parameters:
    - constraints (list): List of constraints for the ILP.
    - instance (Instance): The instance of the problem.

    Returns:
    - allocation_matrix (np.ndarray): The allocation of items to agents as a matrix.
                                        maxtrix[i][j] = x -> times agent i gets item j.
    
    >>> from fairpyx.adaptors import divide
    >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> divide(find_envy_free_allocation, instance=instance)
    np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]])         ##  -> { "Ami": ["Fork", "Fork"], "Rami": ["Knife", "Pen"], "Tami": ["Knife", "Pen"] }
    """
    pass


def find_pareto_dominates_allocation(alloc_matrix: np.ndarray, instance: Instance) -> np.ndarray:
    """
    Find a Pareto-dominates allocation of items to agents.

    Returns:
    - allocation_matrix (np.ndarray): The allocation of items to agents as a matrix.
                                        maxtrix[i][j] = x -> times agent i gets item j.


    >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"] }
    >>> pareto_optimal_allocation = find_pareto_optimal_allocation(alloc_X, instance)
    np.ndarray([[2, 0, 0], [0, 0, 2], [0, 2, 0]]) # -> {"Ami": ["Fork", "Fork"], "Rami": ["Pen", "Pen"], "Tami": ["Knife", "Knife"]} || None
    """
    pass


def create_more_constraints_ILP(alloc_X: np.ndarray, alloc_Y: np.ndarray) -> list:
    """
    Create more constraints for the ILP to disallow the current allocation.

    Parameters:
    - alloc_X (np.ndarray): The current allocation of items to agents.
    - alloc_Y (np.ndarray): The new allocation of items to agents.

    Returns:
    - constraints (list): List of additional constraints to disallow alloc_X and allow alloc_Y.

    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> alloc_Y = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]) # -> {"Ami": ["Fork", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Pen", "Pen"]}
    >>> constraints = create_more_constraints_ILP(alloc_X, alloc_Y)
    len(constraints) == 13      # 3 agents * 2 * 2 items in each equation + the 9th constraint in the algorithm
    """
    pass


#### MAIN

if __name__ == "__main__":
    import doctest, sys
    print("\n",doctest.testmod(), "\n")

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    alloc = find_envy_free_allocation
    # from fairpyx.adaptors import divide_random_instance
    # divide_random_instance(algorithm=high_multiplicity_fair_allocation, 
    #                        num_of_agents=10, num_of_items=4, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)