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
import numpy as np
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


      >>> from fairpyx.adaptors import divide
      >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
      >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
      >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
      >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
      >>> divide(high_multiplicity_fair_allocation, instance=instance)
      {'Ami': ['Fork', 'Fork'], 'Rami': ['Pen', 'Pen'], 'Tami': ['Knife', 'Knife']}
      """

    ## Step 1: Find a envy-free allocation
    ## Step 2: Check if there is a Pareto-dominate allocation
    ## Step 3: If not, modify the ILP to disallow the current allocation
    ## Step 4: Repeat steps 1-3 until a Pareto-optimal allocation is found or no allocation exists
    logger.info("Starting high multiplicity fair allocation.")
    alloc.set_allow_multiple_items(True)
    agents, items, Z_constraints, constraints_ilp = [], [], [], []
    for i in alloc.remaining_items():
        items.append(i)
    for agent in alloc.remaining_agents():
        agents.append(agent)

    allocation = cp.Variable((len(alloc.remaining_agents()), len(alloc.remaining_items())), integer=True)

    logger.debug(f"Initial constraints: {constraints_ilp}")

    iteration_count = 0  # Initialize the iteration counter
    alloc_X = find_envy_free_allocation(alloc, allocation, constraints_ilp)

    while alloc_X is not None:
        iteration_count += 1  # Increment counter on each iteration
        logging.info(f"Attempting envy-free allocation, iteration {iteration_count}")

        alloc_Y = find_pareto_dominates_allocation(alloc, alloc_X)
        if alloc_Y is None:
            logger.info("No Pareto dominating allocation found, finalizing allocation.")
            logger.info(f"The allocatin matrix: {alloc_X}")

            i = 0
            for agent in alloc_X:
                for item in range(0, len(items)):
                    if agent[item] > 0:
                        for item_num in range(0, agent[item]):
                            logger.info(f"remaining items {alloc.remaining_item_capacities}")
                            logger.info(f"{agents[i]} get: {items[item]}")

                            alloc.give(agents[i], items[item])
                    agent[item] -= 1

                i += 1

            logger.info("Allocation complete.")
            return
        else:
            logger.info(f"Found Pareto dominating allocation: {alloc_Y}, updating constraints.")
            allocation.value = None
            new_cont = create_more_constraints_ILP(alloc, alloc_X, alloc_Y, allocation)
            for cont in new_cont:
                constraints_ilp.append(cont)

            alloc_X = find_envy_free_allocation(alloc, allocation, constraints_ilp)

    logger.info(f"No envy-free allocation found after {iteration_count} iterations, ending process.")
    return None


def find_envy_free_allocation(alloc: AllocationBuilder, allocation, constraints_ilp):
    """
       Find an envy-free allocation of items to agents.

       Parameters:
       - constraints (list): List of constraints for the ILP.
       - instance (Instance): The instance of the problem.

       Returns:
       - allocation_matrix : The allocation of items to agents as a matrix.
                                           maxtrix[i][j] = x -> times agent i gets item j.
    >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> allocation = cp.Variable((len(alloc.remaining_agents()), len(alloc.remaining_items())), integer=True)
    >>> alloc_X = find_envy_free_allocation(alloc, allocation, [])
    >>> print(alloc_X)
    [[2 0 0]
     [0 0 2]
     [0 2 0]]
"""

    logger.debug("Searching for envy-free allocation.")
    agents_names, items_names, all_constraints_ilp = [], [], []

    remaining_items = alloc.remaining_items()
    remaining_agents = alloc.remaining_agents()
    num_agents = len(remaining_agents)
    num_items = len(remaining_items)

    agent_valuations = np.zeros((num_agents, num_items))
    agent_capacities = np.zeros(num_agents)
    items_capacities = np.zeros(num_items)

    for i, agent in enumerate(remaining_agents):
        agents_names.append(agent)
        agent_capacities[i] = alloc.remaining_agent_capacities[agent]
        for j, item in enumerate(remaining_items):
            agent_valuations[i, j] = alloc.effective_value(agent, item)
            if i == 0:
                items_names.append(item)
                items_capacities[j] = alloc.remaining_item_capacities[item]

    # Define the objective function (maximize total value)
    objective = cp.Maximize(0)

    item_capacity_constraints = [cp.sum(allocation[:, j]) == items_capacities[j] for j in range(num_items)]
    [item_capacity_constraints.append(cp.sum(allocation[:, j]) >= 0) for j in range(num_items)]

    agent_capacity_constraints = []
    # Ensure no agent receives more items than their capacity
    for i in range(num_agents):
        for j in range(num_items):
            agent_capacity_constraints.append(allocation[i, j] >= 0)

    [agent_capacity_constraints.append(cp.sum(allocation[i, :]) <= agent_capacities[i]) for i in
     range(num_agents)]

    # Define envy-free constraints
    envy_free_constraints = []
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                envy_free_constraints.append(cp.sum(cp.multiply(agent_valuations[i, :], allocation[j, :])) <= cp.sum(
                    cp.multiply(agent_valuations[i, :], allocation[i, :])))

    # Define ILP constraints
    for cont in constraints_ilp:
        all_constraints_ilp.append(cont)

    # Define the problem
    prob = cp.Problem(objective, item_capacity_constraints + agent_capacity_constraints +
                      envy_free_constraints + all_constraints_ilp)

    # Solve the problem
    try:
        prob.solve()
        logger.info("Optimization problem solved successfully.")
        allocations = np.round(allocation.value).astype(int)
        logger.debug(f"Allocation results: {allocations}")
        return allocations
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")
        return None


def find_pareto_dominates_allocation(alloc: AllocationBuilder, alloc_matrix):
    """
    Find a Pareto-dominates allocation of items to agents.

    Returns:
    - allocation_matrix (np.ndarray): The allocation of items to agents as a matrix.
                                        maxtrix[i][j] = x -> times agent i gets item j.


    >>> agent_capacities = {"Ami": 6, "Tami": 6, "Rami": 6}
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominates_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[2 0 0]
     [0 1 2]
     [0 1 0]]

    >>> item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    >>> agent_capacities = {"Ami": 9, "Tami": 9, "Rami": 9}
    >>> valuations = {"Ami": {"Fork": 3, "Knife": 5, "Pen": 8}, "Rami": {"Fork": 5, "Knife": 7, "Pen": 5},"Tami": {"Fork": 4, "Knife": 1, "Pen": 11}}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[3, 0, 0], [0, 0, 3], [0, 3, 0]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominates_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[3 0 2]
     [0 3 0]
     [0 0 1]]

    >>> agent_capacities = {"Ami": 6, "Tami": 6, "Rami": 6, "Yumi": 6}
    >>> item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Yumi": {"Fork": 4, "Knife": 5, "Pen": 6} }
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1], [1, 1, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominates_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[1 0 0]
     [1 3 0]
     [0 0 1]
     [1 0 2]]
    """

    logger.debug("Searching for Pareto dominates allocation.")

    agents_names, items_names, item_capacities, \
        agent_capacities, agent_valuations = get_agents_items_and_capacities(alloc)

    logger.debug(f"Item names and capacities: {alloc.remaining_item_capacities}")

    num_agents = len(alloc.remaining_agents())
    num_items = len(alloc.remaining_items())

    logger.debug(f"Agents names and capacities: {alloc.remaining_agent_capacities}")
    logger.debug(f"Agent valuations: {agent_valuations}")
    logger.debug(f"Initial allocation matrix:\n{alloc_matrix}")

    # Convert agent_valuations to numpy array
    agent_valuations = np.array(agent_valuations)

    # Define decision variables
    allocation_variables = cp.Variable((num_agents, num_items), integer=True)

    # Define capacity constraints
    item_capacity_constraints = [cp.sum(allocation_variables[:, j]) == item_capacities[j] for j in range(num_items)]
    agent_capacity_constraints = [allocation_variables[i, j] >= 0 for i in range(num_agents) for j in range(num_items)]

    # Ensure no agent receives more items than their capacity
    agent_capacity_constraints += [cp.sum(allocation_variables[i, :]) <= agent_capacities[i] for i in range(num_agents)]

    # Define Pareto dominance constraints
    current_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], alloc_matrix[i, :])) for i in
                               range(num_agents)]
    new_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], allocation_variables[i, :])) for i in
                           range(num_agents)]
    pareto_dominates_constraints = [new_value_per_agent[i] >= current_value_per_agent[i] for i in range(num_agents)]

    # Ensure sum_val_y > sum_val_x
    pareto_dominates_constraints.append(cp.sum(new_value_per_agent) >= 1 + cp.sum(current_value_per_agent))

    # Create the optimization problem
    problem = cp.Problem(cp.Maximize(0),
                         pareto_dominates_constraints + item_capacity_constraints + agent_capacity_constraints)

    # Solve the problem
    try:
        problem.solve()
        if allocation_variables.value is None:
            return None
        allocations = np.round(allocation_variables.value).astype(int)
        logger.info("Pareto dominates allocation successfully found.")
        logger.debug(f"Pareto dominates allocation results:\n{allocations}")
        return allocations
    except Exception as e:
        logger.error("Failed to find Pareto dominates allocation: ", exc_info=True)
        return None


def create_more_constraints_ILP(alloc: AllocationBuilder, alloc_X: np.ndarray, alloc_Y: np.ndarray, allocation_variables: cp.Variable):
    """
    Creates additional ILP constraints based on current and previous allocations to ensure
    the new allocation differs from the previous one and satisfies specific conditions.

    Parameters:
    - alloc (AllocationBuilder): The allocation object containing the remaining agents and items.
    - alloc_X (np.ndarray): The previous allocation matrix.
    - alloc_Y (np.ndarray): The current allocation matrix.
    - allocation_variables (cp.Variable): The ILP variable representing the allocation.

    Returns:
    - list: A list of additional constraints.
    """
    logger.debug("Creating more ILP constraints based on current and previous allocations.")

    # Define variables
    agents, items, items_capacities = get_agents_items_and_capacities(alloc, True)  # True to return only this tuple

    # Create binary variables for each agent-item combination
    Z = {(agent, item): cp.Variable(boolean=True) for agent in agents for item in items}
    Z_bar = {(agent, item): cp.Variable(boolean=True) for agent in agents for item in items}

    # Add constraints for each agent-item combination
    constraints = []
    for agent in agents:
        for item in items:
            # Constraint 1 - inequality (7) in the paper.
            constraint1 = (allocation_variables[agent][item] + (alloc_Y[agent][item] - alloc_X[agent][item]) <=
                           -1 + (2 * items_capacities[item]) * (1 - Z[agent, item]))
            constraints.append(constraint1)
            logger.debug(f"Constraint 1 for agent {agent}, item {item}: {constraint1}")

            # Constraint 2 - inequality (8) in the paper.
            constraint2 = (allocation_variables[agent][item] + (alloc_Y[agent][item] - alloc_X[agent][item]) >=
                           Z_bar[agent, item] * (items_capacities[item] + 1))
            constraints.append(constraint2)
            logger.debug(f"Constraint 2 for agent {agent}, item {item}: {constraint2}")

    # Add constraint for each agent that at least one item must change: inequality (9) in the paper.
    constraints.append(
        cp.sum([Z[agent, item] for agent in agents for item in items]) +
        cp.sum([Z_bar[agent, item] for agent in agents for item in items]) >= 1)
    logger.info("Additional ILP constraints created successfully.")

    return constraints


def get_agents_items_and_capacities(alloc: AllocationBuilder, bool=False):
    """
    Helper function to get the remaining agents, items, and their capacities.

    Parameters:
    alloc: The AllocationBuilder object with methods to get remaining agents and items.

    Returns:
    tuple: A tuple containing the list of agents, the list of items, and the list of item capacities.
    """
    # Define variables
    agents_names = alloc.remaining_agents()
    items_names = alloc.remaining_items()
    item_capacities = [alloc.remaining_item_capacities[item] for item in items_names]
    if bool:
        # Return only the tuple of agents, items, and item capacities
        return agents_names, items_names, item_capacities

    # else
    agent_capacities = [alloc.remaining_agent_capacities[agent] for agent in agents_names]
    agent_valuations = [[alloc.effective_value(agent, item) for item in items_names] for agent in agents_names]
    # Return all variables
    return agents_names, items_names, item_capacities, agent_capacities, agent_valuations


#### MAIN ####

if __name__ == "__main__":
    import doctest, sys

    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n", doctest.testmod(), "\n")

    logger.setLevel(logging.DEBUG)

    agent_capacities = {"Ami": 6, "Tami": 6, "Rami": 6, "Yumi": 6}
    item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    valuations = {"Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1},
                  "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Yumi": {"Fork": 4, "Knife": 5, "Pen": 6}}
    instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    alloc = AllocationBuilder(instance)
    alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1], [1, 1, 1]])
    # pareto_optimal_allocation = find_pareto_dominates_allocation(alloc, alloc_X)
    # print(pareto_optimal_allocation)

    agent_capacities = {"Ami": 6, "Tami": 6}
    item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    valuations = {"Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}}
    instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    alloc = AllocationBuilder(instance)
    alloc_X = np.array([[3, 2, 1], [0, 1, 2]])
    pareto_optimal_allocation = find_pareto_dominates_allocation(alloc, alloc_X)
    print(pareto_optimal_allocation)
