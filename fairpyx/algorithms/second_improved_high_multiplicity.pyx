# fairpyx/algorithms/second_improved_high_multiplicity.pyx

import numpy as np
import cvxpy as cp
from fairpyx import Instance, AllocationBuilder, divide
import logging

cimport numpy as np
cimport cython

# to run this code, you need to run the setup.py -> open cmd and type:
# python setup.py build_ext --inplace

@cython.boundscheck(False)
@cython.wraparound(False)
def second_improved_high_multiplicity_fair_allocation(object alloc):
    cdef list agents, items, constraints_ilp
    cdef int iteration_count
    cdef np.ndarray alloc_X, alloc_Y

    alloc.set_allow_multiple_items(True)
    agents = list(alloc.remaining_agents())
    items = list(alloc.remaining_items())
    constraints_ilp = []

    allocation_variables = cp.Variable((len(agents), len(items)), integer=True)
    iteration_count = 0
    alloc_X = second_improved_find_envy_free_allocation(alloc, allocation_variables, constraints_ilp)

    while alloc_X is not None:
        iteration_count += 1
        logging.info(f"Attempting envy-free allocation, iteration {iteration_count}")
        alloc_Y = second_improved_find_pareto_dominating_allocation(alloc, alloc_X)
        if alloc_Y is None:
            for i, agent in enumerate(alloc_X):
                for item in range(len(items)):
                    if agent[item] > 0:
                        for item_num in range(agent[item]):
                            alloc.give(agents[i], items[item])
                    agent[item] -= 1
            return
        else:
            constraints_ilp.extend(
                second_improved_create_more_constraints_ILP(alloc, alloc_X, alloc_Y, allocation_variables))
            alloc_X = second_improved_find_envy_free_allocation(alloc, allocation_variables, constraints_ilp)
    return

def second_improved_find_envy_free_allocation(object alloc, object allocation_variables, list constraints_ilp):
    cdef list constraints
    cdef int num_agents, num_items
    cdef np.ndarray item_capacities, agent_capacities, agent_valuations
    constraints = []
    num_agents, num_items, item_capacities, agent_capacities, agent_valuations = get_agents_items_and_capacities(alloc)
    objective = cp.Maximize(0)
    item_capacity_constraints = [cp.sum(allocation_variables[:, j]) == item_capacities[j] for j in range(num_items)]
    agent_capacity_constraints = [allocation_variables[i, j] >= 0 for i in range(num_agents) for j in range(num_items)]
    agent_capacity_constraints += [cp.sum(allocation_variables[i, :]) <= agent_capacities[i] for i in range(num_agents)]
    envy_free_constraints = []
    for i in range(num_agents):
        i_profit = cp.sum(cp.multiply(agent_valuations[i, :], allocation_variables[i, :]))
        for j in range(num_agents):
            if i != j:
                j_profit = cp.sum(cp.multiply(agent_valuations[i, :], allocation_variables[j, :]))
                envy_free_constraints.append(j_profit <= i_profit)

    prob = cp.Problem(objective,
                      item_capacity_constraints + agent_capacity_constraints + envy_free_constraints + constraints_ilp)
    try:
        prob.solve()
        if allocation_variables.value is None:
            return None
        allocation = np.round(allocation_variables.value).astype(np.int32)
        return allocation
    except:
        return None

def second_improved_find_pareto_dominating_allocation(object alloc, np.ndarray alloc_matrix):
    cdef int num_agents, num_items
    cdef np.ndarray item_capacities, agent_capacities, agent_valuations
    num_agents, num_items, item_capacities, agent_capacities, agent_valuations = get_agents_items_and_capacities(alloc)
    allocation_var = cp.Variable((num_agents, num_items), integer=True)
    item_capacity_constraints = [cp.sum(allocation_var[:, j]) == item_capacities[j] for j in range(num_items)]
    agent_capacity_constraints = [allocation_var[i, j] >= 0 for i in range(num_agents) for j in range(num_items)]
    agent_capacity_constraints += [cp.sum(allocation_var[i, :]) <= agent_capacities[i] for i in range(num_agents)]
    current_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], alloc_matrix[i, :])) for i in
                               range(num_agents)]
    new_value_per_agent = [cp.sum(cp.multiply(agent_valuations[i, :], allocation_var[i, :])) for i in range(num_agents)]
    pareto_dominating_constraints = [new_value_per_agent[i] >= current_value_per_agent[i] for i in range(num_agents)]
    pareto_dominating_constraints.append(cp.sum(new_value_per_agent) >= 1 + cp.sum(current_value_per_agent))

    problem = cp.Problem(cp.Maximize(0),
                         pareto_dominating_constraints + item_capacity_constraints + agent_capacity_constraints)
    try:
        problem.solve()
        if allocation_var.value is None:
            return None
        allocations = np.round(allocation_var.value).astype(np.int32)
        return allocations
    except:
        return None

def second_improved_create_more_constraints_ILP(object alloc, np.ndarray alloc_X, np.ndarray alloc_Y,
                                                object allocation_variables):
    cdef list constraints
    cdef int num_agents, num_items
    cdef np.ndarray delta

    agents, items, item_capacities = get_agents_items_and_capacities(alloc, True)
    num_agents, num_items = len(agents), len(items)

    Z = cp.Variable((num_agents, num_items), boolean=True)
    Z_bar = cp.Variable((num_agents, num_items), boolean=True)
    constraints = []
    delta = alloc_Y - alloc_X

    for i in range(num_agents):
        for j in range(num_items):
            constraint7 = allocation_variables[i, j] + delta[i, j] <= -1 + (2 * item_capacities[j]) * (1 - Z[i, j])
            constraints.append(constraint7)
            constraint8 = allocation_variables[i, j] + delta[i, j] >= -item_capacities[j] + Z_bar[i, j] * (
                        2 * item_capacities[j] + 1)
            constraints.append(constraint8)

    constraint9 = (cp.sum([Z[i, j] for i in range(num_agents) for j in range(num_items)]) +
                   cp.sum([Z_bar[i, j] for i in range(num_agents) for j in range(num_items)])) >= 1
    constraints.append(constraint9)
    return constraints

def second_improved_calculate_values(object alloc, np.ndarray alloc_X):
    cdef dict values
    cdef list agents, items
    cdef np.ndarray agent_valuations

    agents, items, item_capacities = get_agents_items_and_capacities(alloc, True)
    agent_valuations = np.array([[alloc.effective_value(agent, item) for item in items] for agent in agents])
    values = {agent: 0 for agent in agents}

    for i in range(len(agents)):
        for j in range(len(items)):
            values[agents[i]] += alloc_X[i, j] * agent_valuations[i, j]
    return values

def second_improved_calculate_envy_matrix(object alloc, np.ndarray alloc_X):
    cdef dict baskets_values
    cdef list agents, items
    cdef np.ndarray agent_valuations

    agents, items, item_capacities = get_agents_items_and_capacities(alloc, True)
    agent_valuations = np.array([[alloc.effective_value(agent, item) for item in items] for agent in agents])
    baskets_values = {agent: [] for agent in agents}

    for agent_index in range(len(agents)):
        for i in range(len(agents)):
            baskets_values[agents[agent_index]].append(0)
            for j in range(len(items)):
                baskets_values[agents[agent_index]][i] += alloc_X[i, j] * agent_valuations[agent_index][j]
    return baskets_values

def get_agents_items_and_capacities(object alloc, bint simple=False):
    cdef list agents_names, items_names
    cdef np.ndarray item_capacities, agent_capacities, agent_valuations

    agents_names = list(alloc.remaining_agents())
    items_names = list(alloc.remaining_items())
    item_capacities = np.array([alloc.remaining_item_capacities[item] for item in items_names], dtype=np.int32)

    if simple:
        return agents_names, items_names, item_capacities
    agent_capacities = np.array([alloc.remaining_agent_capacities[agent] for agent in agents_names], dtype=np.int32)
    agent_valuations = np.array(
        [[alloc.effective_value(agent, item) for item in items_names] for agent in agents_names], dtype=np.int32)
    return len(agents_names), len(items_names), item_capacities, agent_capacities, agent_valuations