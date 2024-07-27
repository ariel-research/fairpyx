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
from fairpyx import Instance, AllocationBuilder, divide
import logging

logger = logging.getLogger(__name__)


def high_multiplicity_fair_allocation(alloc: AllocationBuilder):
    """
      Finds an allocation maximizing the sum of utilities for given instance with envy-freeness and Pareto-optimality constraints.

      Parameters:
      - alloc (AllocationBuilder): The allocation of items to agents.

      Returns:
      - alloc (AllocationBuilder): The allocation of items to agents.


      >>> agent_capacities = {"Ami": 2, "Tami": 2, "Rami": 2}
      >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
      >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
      >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
      >>> divide(high_multiplicity_fair_allocation, instance=instance)
      {'Ami': ['Fork', 'Fork'], 'Rami': ['Pen', 'Pen'], 'Tami': ['Knife', 'Knife']}

    >>> agent_capacities = {"Ami": 9, "Tami": 9}
    >>> item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    >>> valuations = {"Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> divide(high_multiplicity_fair_allocation, instance=instance)
    {'Ami': ['Fork', 'Fork', 'Fork'], 'Tami': ['Knife', 'Knife', 'Knife', 'Pen', 'Pen', 'Pen']}
      """

    ## Step 1: Find a envy-free allocation
    ## Step 2: Check if there is a Pareto-dominate allocation
    ## Step 3: If not, modify the ILP to disallow the current allocation
    ## Step 4: Repeat steps 1-3 until a Pareto-optimal allocation is found or no allocation exists

    logger.info("Starting high multiplicity fair allocation.")
    alloc.set_allow_multiple_copies(True)
    agents, items, constraints_ilp = [], [], []
    for i in alloc.remaining_items():
        items.append(i)
    for agent in alloc.remaining_agents():
        agents.append(agent)

    allocation_variables = cp.Variable((len(alloc.remaining_agents()), len(alloc.remaining_items())), integer=True)

    iteration_count = 0  # Initialize the iteration counter
    alloc_X = find_envy_free_allocation(alloc, allocation_variables, constraints_ilp)

    while alloc_X is not None:
        iteration_count += 1  # Increment counter on each iteration
        logging.info(f"Attempting envy-free allocation, iteration {iteration_count}")
        logger.info(f"Attempting envy-free allocation, iteration {iteration_count}")

        logger.debug(f'The envy matrix of the allocation {calculate_envy_matrix(alloc, alloc_X)} ')
        alloc_Y = find_pareto_dominating_allocation(alloc, alloc_X)
        if alloc_Y is None:
            logger.info("No Pareto dominating allocation found, finalizing allocation:\n%s", alloc_X)

            i = 0
            for agent in alloc_X:
                for item in range(0, len(items)):
                    if agent[item] > 0:
                        for item_num in range(0, agent[item]):
                            # logger.info(f"remaining items {alloc.remaining_item_capacities}")
                            # logger.info(f"{agents[i]} get: {items[item]}")

                            alloc.give(agents[i], items[item], logger)
                    agent[item] -= 1

                i += 1

            logger.info(f"allocation found after {iteration_count} iterations")
            return
        else:
            logger.debug(
                f'The envy matrix of the Pareto dominating allocation {calculate_envy_matrix(alloc, alloc_Y)} ')
            logger.debug(f'The values of the first allocation {calculate_values(alloc, alloc_X)} ')
            logger.debug(f'The values of the Pareto dominating allocation {calculate_values(alloc, alloc_Y)}')

            constraints_ilp.extend(create_more_constraints_ILP(alloc, alloc_X, alloc_Y, allocation_variables))
            alloc_X = find_envy_free_allocation(alloc, allocation_variables, constraints_ilp)

    logger.info(f"No envy-free allocation found after {iteration_count} iterations, ending process.")
    return None


def find_envy_free_allocation(alloc: AllocationBuilder, allocation_variables, constraints_ilp):
    """
       Find an envy-free allocation of items to agents.

       Parameters:
       - constraints (list): List of constraints for the ILP.
       - instance (Instance): The instance of the problem.

       Returns:
       - allocation_matrix : The allocation of items to agents as a matrix.
                                           maxtrix[i][j] = x -> times agent i gets item j.
    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> allocation_vars = cp.Variable((len(alloc.remaining_agents()), len(alloc.remaining_items())), integer=True)
    >>> alloc_X = find_envy_free_allocation(alloc, allocation_vars, [])
    >>> print(alloc_X)
    [[2 0 0]
     [0 0 2]
     [0 2 0]]
    """

    logger.debug("Searching for envy-free allocation.")

    num_agents, num_items, items_capacities, \
        agent_capacities, agent_valuations = get_agents_items_and_capacities(alloc)

    # Define the objective function (maximize total value)
    objective = cp.Maximize(0)

    item_capacity_constraints = [cp.sum(allocation_variables[:, j]) == items_capacities[j] for j in range(num_items)]

    agent_capacity_constraints = []
    # Ensure no agent receives more items than their capacity
    for i in range(num_agents):
        for j in range(num_items):
            agent_capacity_constraints.append(allocation_variables[i, j] >= 0)

    agent_capacity_constraints += [cp.sum(allocation_variables[i, :]) <= agent_capacities[i] for i in
                                   range(num_agents)]
    # Define envy-free constraints
    envy_free_constraints = []
    for i in range(num_agents):
        i_profit = cp.sum(cp.multiply(agent_valuations[i, :], allocation_variables[i, :]))
        for j in range(num_agents):
            if i != j:
                j_profit = cp.sum(cp.multiply(agent_valuations[i, :], allocation_variables[j, :]))
                envy_free_constraints.append(j_profit <= i_profit)

    # Define the problem
    prob = cp.Problem(objective, item_capacity_constraints + agent_capacity_constraints +
                      envy_free_constraints + constraints_ilp)
    # logger.debug(f'Problem constraints:')
    # for constraint in prob.constraints:
    #     logger.debug(f'{constraint} ')
    # Solve the problem
    try:
        prob.solve()
        if allocation_variables.value is None:
            return None
        allocation = np.round(allocation_variables.value).astype(int)
        logger.debug(f"Allocation results:\n{allocation}")
        return allocation
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")
        return None


def find_pareto_dominating_allocation(alloc: AllocationBuilder, alloc_matrix):
    """
    Find a Pareto-dominates allocation of items to agents.

    Returns:
    - allocation_matrix (np.ndarray): The allocation of items to agents as a matrix.
                                        maxtrix[i][j] = x -> times agent i gets item j.

    >>> item_capacities = {"Fork": 2, "Knife": 2, "Pen": 2}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1} }
    >>> instance = Instance(item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominating_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[2 0 0]
     [0 1 2]
     [0 1 0]]

    >>> item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    >>> valuations = {"Ami": {"Fork": 3, "Knife": 5, "Pen": 8}, "Rami": {"Fork": 5, "Knife": 7, "Pen": 5},"Tami": {"Fork": 4, "Knife": 1, "Pen": 11}}
    >>> instance = Instance(item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[3, 0, 0], [0, 0, 3], [0, 3, 0]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominating_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[2 0 1]
     [0 1 2]
     [1 2 0]]


    >>> item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    >>> valuations = { "Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Rami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}, "Yumi": {"Fork": 4, "Knife": 5, "Pen": 6} }
    >>> instance = Instance(item_capacities=item_capacities, valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1], [1, 1, 1]]) # -> {"Ami": ["Pen", "Fork"], "Tami": ["Knife", "Knife"], "Rami": ["Fork", "Pen"]}
    >>> pareto_optimal_allocation = find_pareto_dominating_allocation(alloc, alloc_X)
    >>> print(pareto_optimal_allocation)
    [[1 0 0]
     [0 3 0]
     [1 0 1]
     [1 0 2]]

    """

    logger.debug("Searching for a Pareto-dominating allocation")

    num_agents, num_items, item_capacities, \
        agent_capacities, agent_valuations = get_agents_items_and_capacities(alloc)

    # logger.debug(f"Item names and capacities: {alloc.remaining_item_capacities}")
    # logger.debug(f"Agents names and capacities: {alloc.remaining_agent_capacities}")
    # logger.debug(f"Agent valuations: {agent_valuations}")

    # Define decision variables
    allocation_var = cp.Variable((num_agents, num_items), integer=True)

    # Define capacity constraints
    item_capacity_constraints = [cp.sum(allocation_var[:, j]) == item_capacities[j] for j in range(num_items)]
    agent_capacity_constraints = [allocation_var[i, j] >= 0 for i in range(num_agents) for j in range(num_items)]
    agent_capacity_constraints += [cp.sum(allocation_var[i, :]) <= agent_capacities[i] for i in range(num_agents)]
    # Define Pareto dominance constraints
    current_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], alloc_matrix[i, :])) for i in
                               range(num_agents)]
    new_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], allocation_var[i, :])) for i in
                           range(num_agents)]
    pareto_dominating_constraints = [new_value_per_agent[i] >= current_value_per_agent[i] for i in range(num_agents)]

    # Ensure sum_val_y > sum_val_x
    pareto_dominating_constraints.append(cp.sum(new_value_per_agent) >= 1 + cp.sum(current_value_per_agent))

    # Create the optimization problem
    problem = cp.Problem(cp.Maximize(0),
                         pareto_dominating_constraints + item_capacity_constraints + agent_capacity_constraints)

    # Solve the problem
    try:
        problem.solve()
        if allocation_var.value is None:
            return None
        allocations = np.round(allocation_var.value).astype(int)
        logger.debug(f"Pareto dominating allocation found:\n{allocations}")
        return allocations
    except Exception as e:
        logger.error("Failed to find Pareto dominating allocation: ", exc_info=True)
        return None


def create_more_constraints_ILP(alloc: AllocationBuilder, alloc_X: np.ndarray, alloc_Y: np.ndarray,
                                allocation_variables: cp.Variable):
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
    num_agents = len(agents)
    num_items = len(items)

    # Create binary variables for each agent-item combination
    Z = cp.Variable((num_agents, num_items), boolean=True)
    Z_bar = cp.Variable((num_agents, num_items), boolean=True)

    # Add constraints for each agent-item combination
    constraints = []

    delta = alloc_Y - alloc_X
    logger.debug(f'Delta:\n%s', delta)
    for i in range(num_agents):
        for j in range(num_items):
            # Constraint 7 - inequality (7) in the paper.
            constraint7 = allocation_variables[i][j] + delta[i][j] <= -1 + (2 * items_capacities[j] + 1) * (1 - Z[i][j])
            constraints.append(constraint7)

            # Constraint 8 - inequality (8) in the paper.
            constraint8 = allocation_variables[i][j] + delta[i][j] >= -items_capacities[j] + Z_bar[i][j] * (
                    2 * items_capacities[j] + 1)
            constraints.append(constraint8)
    # Add constraint for each agent that at least one item must change: inequality (9) in the paper.

    constraint9 = (cp.sum([Z[i][j] for i in range(num_agents) for j in range(num_items)]) +
                   cp.sum([Z_bar[i][j] for i in range(num_agents) for j in range(num_items)])) >= 1
    constraints.append(constraint9)

    return constraints


def calculate_values(alloc: AllocationBuilder, alloc_X: np.ndarray):
    """
        Calculates a final value for each agent

    Parameters:
    - alloc (AllocationBuilder): The allocation object containing the remaining agents and items.
    - alloc_X (np.ndarray): The allocation matrix.


    Returns:
    - Dict : A dict of final values per agent.
    """
    logger.info("Calculates values")

    # Define variables
    agents, items, items_capacities = get_agents_items_and_capacities(alloc, True)  # True to return only this tuple
    agent_valuations = [[alloc.effective_value(agent, item) for item in items] for agent in agents]

    values = {agent: 0 for agent in agents}
    for i in range(len(agents)):
        for j in range(len(items)):
            values[agents[i]] += alloc_X[i][j] * agent_valuations[i][j]
    return values


def calculate_envy_matrix(alloc: AllocationBuilder, alloc_X: np.ndarray):
    """
        Calculates the envy matrix

    Parameters:
    - alloc (AllocationBuilder): The allocation object containing the remaining agents and items.
    - alloc_X (np.ndarray): The allocation matrix.


    Returns:
    - Dict : A dict of envy matrix.
    """
    logger.info("Calculates nvy matrix")

    # Define variables
    agents, items, items_capacities = get_agents_items_and_capacities(alloc, True)  # True to return only this tuple
    agent_valuations = [[alloc.effective_value(agent, item) for item in items] for agent in agents]

    baskets_values = {agent: [] for agent in agents}

    for agent_index in range(len(agents)):
        for i in range(len(agents)):
            baskets_values[agents[agent_index]].append(0)
            for j in range(len(items)):
                baskets_values[agents[agent_index]][i] += alloc_X[i][j] * agent_valuations[agent_index][j]

    return baskets_values


def get_agents_items_and_capacities(alloc: AllocationBuilder, bool=False):
    """
    Helper function to get the remaining agents, items, and their capacities.

    Parameters:
    alloc: The AllocationBuilder object with methods to get remaining agents and items.

    Returns:
    tuple: A tuple containing the list of agents, the list of items, and the list of item capacities.
    """
    # Define variables
    agents_names = [agent for agent in alloc.remaining_agents()]
    items_names = [item for item in alloc.remaining_items()]
    item_capacities = [alloc.remaining_item_capacities[item] for item in alloc.remaining_items()]
    if bool:
        # Return only the tuple of agents, items, and item capacities
        return agents_names, items_names, item_capacities

    # else
    agent_capacities = [alloc.remaining_agent_capacities[agent] for agent in agents_names]
    agent_valuations = [[alloc.effective_value(agent, item) for item in items_names] for agent in agents_names]
    agent_valuations = np.array(agent_valuations)

    # Return all variables
    return len(agents_names), len(items_names), item_capacities, agent_capacities, agent_valuations


#### MAIN ####

def instance_4_3():
    item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    valuations = {
        "Ami": {"Fork": 2, "Knife": 0, "Pen": 0},
        "Rami": {"Fork": 0, "Knife": 1, "Pen": 1},
        "Tami": {"Fork": 0, "Knife": 1, "Pen": 1},
        "Yumi": {"Fork": 4, "Knife": 5, "Pen": 6}}
    return Instance(item_capacities=item_capacities, valuations=valuations)


def instance_4_6():
    item_capacities = {'c1': 5, 'c2': 6, 'c3': 2, 'c4': 3, 'c5': 5, 'c6': 2}
    valuations = {
        's1': {'c1': 152, 'c2': 86, 'c3': 262, 'c4': 68, 'c5': 263, 'c6': 169},
        's2': {'c1': 124, 'c2': 70, 'c3': 98, 'c4': 244, 'c5': 329, 'c6': 135},
        's3': {'c1': 170, 'c2': 235, 'c3': 295, 'c4': 91, 'c5': 91, 'c6': 118},
        's4': {'c1': 158, 'c2': 56, 'c3': 134, 'c4': 255, 'c5': 192, 'c6': 205}}
    return Instance(item_capacities=item_capacities, valuations=valuations)


def instance2_4_6():
    item_capacities = {'c1': 2, 'c2': 2, 'c3': 2, 'c4': 2, 'c5': 2, 'c6': 2}
    valuations = {
        's1': {'c1': 100, 'c2': 60, 'c3': 60, 'c4': 60, 'c5': 70, 'c6': 60},
        's2': {'c1': 60, 'c2': 100, 'c3': 60, 'c4': 60, 'c5': 70, 'c6': 60},
        's3': {'c1': 60, 'c2': 60, 'c3': 100, 'c4': 60, 'c5': 60, 'c6': 70},
        's4': {'c1': 60, 'c2': 60, 'c3': 60, 'c4': 100, 'c5': 60, 'c6': 70}}
    return Instance(item_capacities=item_capacities, valuations=valuations)


if __name__ == "__main__":
    import doctest

    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n", doctest.testmod(), "\n")

    logger.setLevel(logging.DEBUG)

    # instance = Instance.random_uniform(
    #     num_of_agents=4, num_of_items=6,
    #     agent_capacity_bounds=(6,6), item_capacity_bounds=(2,6),
    #     item_base_value_bounds=(50,150),
    #     item_subjective_ratio_bounds=(0.5,1.5),
    #     normalized_sum_of_values=1000,
    #     random_seed=1)
    # print(instance)
    divide(high_multiplicity_fair_allocation, instance_4_3())

    # alloc = AllocationBuilder(instance)
    # alloc_X = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1], [1, 1, 1]])
    # pareto_optimal_allocation = find_pareto_dominating_allocation(alloc, alloc_X)
    # print(pareto_optimal_allocation)

    # agent_capacities = {"Ami": 6, "Tami": 6}
    # item_capacities = {"Fork": 3, "Knife": 3, "Pen": 3}
    # valuations = {"Ami": {"Fork": 2, "Knife": 0, "Pen": 0}, "Tami": {"Fork": 0, "Knife": 1, "Pen": 1}}
    # instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    # alloc = AllocationBuilder(instance)
    # alloc_X = np.array([[3, 2, 1], [0, 1, 2]])
    # pareto_optimal_allocation = find_pareto_dominating_allocation(alloc, alloc_X)
    # print(pareto_optimal_allocation)
