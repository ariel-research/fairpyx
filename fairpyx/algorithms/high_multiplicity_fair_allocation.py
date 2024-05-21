"""
High Multiplicity Fair Allocation Algorithm - Finds an allocation maximizing the sum of utilities for given instance 
                                                with envy-freeness and Pareto-optimality constraints.
"""

from gurobipy import *

from fairpyx.utils.graph_utils import many_to_many_matching_using_network_flow
from fairpyx import Instance, AllocationBuilder

import logging
logger = logging.getLogger(__name__)

# Suppress Gurobi output
setParam('OutputFlag', 0)


def high_multiplicity_fair_allocation(alloc: AllocationBuilder):
    """
    Finds an allocation maximizing the sum of utilities for given instance with envy-freeness and Pareto-optimality constraints.

    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.
    """

    pass


def initial_ILP(alloc: AllocationBuilder):
    """
    Set up the initial Integer Linear Program (ILP) formulation to allocate items to agents and attempt to find a feasible solution.

    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.

    Returns:
    - alloc (AllocationBuilder): The allocation of items to agents.
    """

    pass

def check_validity(alloc: AllocationBuilder):
    """
    Check the validity of an allocation by solving an ILP to ensure that it is Pareto-optimal.

    Parameters:
    - alloc (AllocationBuilder): The allocation of items to agents.

    Returns:
    - valid (bool): True if the allocation is valid (Pareto-optimal), False otherwise.
    """

    pass


def modify_ILP(model, alloc: AllocationBuilder):
    """
    Modify the initial Integer Linear Program (ILP) formulation to disallow the current allocation.

    Parameters:
    - model (gurobipy.Model): The Gurobi model representing the ILP.
    - alloc (AllocationBuilder): The current allocation of items to agents.

    Returns:
    - model (gurobipy.Model): The modified Gurobi model.
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