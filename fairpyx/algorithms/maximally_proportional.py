"""
An implementation of the algorithm in:
"An algorithm for the proportional division of indivisible items"
by Brams, Steven J. and Kilgour, D. Marc and Klamler, Christian (2014),
http://https://mpra.ub.uni-muenchen.de/56587/

Programmer: Elroi Carmel
Date: 2025-05
"""

from fairpyx import Instance, AllocationBuilder, divide
from typing import Any
from collections.abc import Iterable, Callable
from random import choice
import logging, cvxpy as cp, numpy as np
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)


def maximally_proportional_allocation(alloc: AllocationBuilder):
    """
    Finds an allocation maximizing the possible number of players with proportional bundles,
    guaranteeing the best attainable minimum rank among them.

    >>> instance = Instance(valuations={"Alice":[40, 35, 25], "Bob":[35, 40, 25], "Tom":[40, 25, 35]})
    >>> divide(maximally_proportional_allocation, instance)
    {'Alice': [0], 'Bob': [1], 'Tom': [2]}
    >>> valuations = [[7, 4, 33, 21, 24],
    ...              [47, 15, 43, 21, 25],
    ...              [10, 24, 20, 31, 5],
    ...              [32, 33, 3, 2, 23],
    ...              [30, 4, 1, 11, 31]]
    >>> instance = Instance(valuations=valuations)
    >>> divide(maximally_proportional_allocation, instance)
    {0: [2], 1: [0], 2: [3], 3: [1], 4: [4]}
    >>> instance = Instance(valuations=[[8,30,48,10,15],[12,7,29,15,5],[35,42,5,22,44]])
    >>> divide(maximally_proportional_allocation, instance)
    {0: [2], 1: [0, 3], 2: [1, 4]}

    """
    agents, items = alloc.remaining_agents(), alloc.remaining_items()
    nagents, nitems = alloc.instance.num_of_agents, alloc.instance.num_of_items
    min_bundles = {agent: get_minimal_bundles(alloc, agent, False) for agent in agents}
    max_rank = max(map(len, min_bundles.values()))
    max_agents_received = 0
    rank = 0
    col_to_bundle = {}
    ids = {
        "agents": dict(zip(agents, range(nagents))),
        "items": dict(zip(items, range(nagents, nagents + nitems))),
    }

    keep_descending = True
    indptr, indices = [0], []
    while keep_descending:
        if rank >= max_rank or max_agents_received == len(agents):
            keep_descending = False
        else:
            for agent, mb in min_bundles.items():
                if rank in range(len(mb)):
                    indices.append(ids["agents"][agent])
                    for item in mb[rank]:
                        indices.append(ids["items"][item])
                    col_to_bundle[len(indptr) - 1] = (agent, rank)
                    indptr.append(len(indices))

            # build probelm
            sparse = csc_matrix(
                (np.ones(len(indices)), indices, indptr),
                shape=(len(agents) + len(items), len(indptr) - 1),
                copy=False,
            )
            x = cp.Variable(len(indptr) - 1, boolean=True)
            objective = cp.Maximize(cp.sum(x))
            constrains = [sparse @ x <= 1]
            prob = cp.Problem(objective=objective, constraints=constrains)
            agents_recived = prob.solve()
            if agents_recived > max_agents_received:  # type: ignore
                maxmin_solution = x  # save the solution
                max_agents_received = agents_recived
            rank += 1

    # solution is a dict of[agent -> rank of minimal bundle he got]
    maxmin_sparse_matrix = sparse[:, : maxmin_solution.size]
    pool = [extract_solution_from_variable(maxmin_solution, col_to_bundle)]
    search_more_solutions = True
    constrains = [
        maxmin_sparse_matrix @ maxmin_solution <= 1,
        cp.sum(maxmin_solution) == max_agents_received,
    ]
    objective = cp.Maximize(0)

    while search_more_solutions:
        # main work will be in adding constraints
        constrains.append(cp.sum([maxmin_solution[i] for i in maxmin_solution.value.nonzero()[0]]) <= max_agents_received - 1)  # type: ignore
        prob = cp.Problem(objective, constrains)
        prob.solve()
        if prob.status != "optimal":
            search_more_solutions = False
        else:
            # add solution to pool
            pool.append(extract_solution_from_variable(maxmin_solution, col_to_bundle))

    prt_opt = pareto_frontier(pool)

    for agent, rank in choice(prt_opt).items():
        alloc.give_bundle(agent, min_bundles[agent][rank])


def extract_solution_from_variable(x: cp.Variable, col_map: dict):
    chosen_columns = x.value.nonzero()[0]  # type: ignore
    return {col_map[c][0]: col_map[c][1] for c in chosen_columns}


def pareto_dominates(rank_alloc_a: dict, rank_alloc_b: dict) -> bool:
    """
    Given 2 allocations of minimal bundles A and B, A pareto dominates B
    iff:
    1. For each agent, his minimal bundle rank in A <= than in B
    2. There exists at least one agent who's minimal bundle rank in A
       is < than in B.

    Args:
        rank_alloc_a (dict): A mapper from agent to the rank of his received minimal bundle
        rank_alloc_b (dict): A mapper from agent to the rank of his received minimal bundle

    Returns:
        bool: True if a pareto dominates b

    Usage:
    >>> pareto_dominates({"a": 1, "b": 4}, {"a": 2, "b": 4})
    True
    >>> pareto_dominates({"a": 2, "b": 4}, {"a": 2, "b": 4})
    False
    >>> pareto_dominates({"a": 2, "b": 4}, {"a": 1, "b": 7})
    False
    >>> pareto_dominates({"a": 0, "b": 2}, {"a": 1, "c": 4})
    False
    """
    if rank_alloc_a.keys() != rank_alloc_b.keys():
        return False
    better = False
    for agent in rank_alloc_a:
        ra, rb = rank_alloc_a[agent], rank_alloc_b[agent]
        if ra > rb:  # a harms one agent
            return False
        if ra < rb:  # a is better for at least one agent
            better = True
    return better


def pareto_frontier(rank_allocs: list[dict]) -> list:
    """
    Filters the pareto dominated alloactions

    Args:
        rank_allocs (list[dict]): List of rank allocations

    Returns:
        list: Only the pareto dominating allocations

    Usage:
    >>> allocs = [{"Alice": 1, "Bob": 4}, {"Alice": 2, "Bob": 4}]
    >>> pareto_frontier(allocs)
    [{'Alice': 1, 'Bob': 4}]
    """
    frontier = []
    for rnk in rank_allocs:
        dominated = False
        to_drop = []

        for f in frontier:
            if pareto_dominates(f, rnk):
                dominated = True  # rnk is useless
                break
            if pareto_dominates(rnk, f):  # f turned out as not pareto optimal
                to_drop.append(f)

        if not dominated:
            frontier.append(rnk)
            for f in to_drop:
                frontier.remove(f)

    return frontier


def get_minimal_bundles(
    alloc: AllocationBuilder, agent: Any, sort_bundle: bool = True
) -> list:
    """
    Finds all of the agent's minimal bundles and returns them in descending
    order by their total value.

    Args:
        alloc (AllocationBuilder): Used to retrive the items and the agent's utility
        agent (Any): One of the agents

    Returns:
        List: agent's minimal bundle

    Examples:
    >>> val = [[35, 30, 10, 25],[25, 21, 34, 20],[16, 32, 12, 38], [91, 66, 12, 89]]
    >>> instance = Instance(valuations=val)
    >>> alloc = AllocationBuilder(instance)
    >>> get_minimal_bundles(alloc, 0)
    [[0], [1], [3]]
    >>> get_minimal_bundles(alloc, 1)
    [[1, 3], [2], [0]]
    >>> get_minimal_bundles(alloc, 2)
    [[3], [1], [0, 2]]
    >>> get_minimal_bundles(alloc, 3)
    [[0], [3], [1]]
    >>> val = [[15, 33, 16, 6, 26], [34, 38, 4, 16, 8]]
    >>> alloc = AllocationBuilder(Instance(valuations=val))
    >>> get_minimal_bundles(alloc, 0)
    [[1, 4], [0, 2, 4], [1, 2], [0, 1], [2, 3, 4]]
    >>> get_minimal_bundles(alloc, 1)
    [[0, 1], [1, 3], [1, 2, 4], [0, 3]]
    """
    res = []
    logger.info(" collecting all of agent %s minimal bundles ".center(50, "#"), agent)
    items_sorted = sorted(
        alloc.remaining_items(),
        key=lambda x: alloc.effective_value(agent, x),
        reverse=True,
    )
    proportional_share = alloc.agent_bundle_value(agent, alloc.remaining_items()) / len(
        alloc.remaining_agents()
    )
    logger.info("proportional value of agent %s is %s", agent, proportional_share)
    subgroup = []

    def backtrack(i, bundle_value):
        logger.debug("assesing subgroup %s", subgroup)
        if bundle_value >= proportional_share:
            res.append(sorted(subgroup) if sort_bundle else list(subgroup))
            logger.debug(
                "subgroup is minimal bundle! total value: %s. added to result",
                bundle_value,
            )
        elif i >= len(items_sorted):
            logger.debug("no left items to grow the group")
            return
        else:
            logger.debug("subgroup total value is too low. add some item")
            item = items_sorted[i]
            subgroup.append(item)
            backtrack(i + 1, bundle_value + alloc.effective_value(agent, item))
            subgroup.pop()
            backtrack(i + 1, bundle_value)

    backtrack(0, 0)
    res.sort(key=lambda bundle: alloc.agent_bundle_value(agent, bundle), reverse=True)
    return res


def is_minimal_bundle(
    bundle: Iterable, valuation_func: Callable, prop_share: float
) -> bool:
    """
    A bundle is minimal for an agent if and only if it satisfies the
    following conditions:
    1. The agent's valuation of the bundle is >= (sum_items_value)/(num of agents) i.e. proportional.
    2. Removing any item from the bundle makes it strictly less than proportional.

    Args:
        bundle (Iterable): items in the bundle
        valuation_func (Callable): agent's utility function
        prop_share (float): agnet's proportional share of utility

    Returns:
        bool: True if the bundle is minimal else False

    Examples:
    >>> is_minimal_bundle([0], [40, 35, 25], 100/3)
    True
    >>> is_minimal_bundle([0, 1], [40, 35, 25], 100/3)
    False
    >>> is_minimal_bundle([0, 1], [35, 30, 10, 25], 25)
    False
    >>> is_minimal_bundle([0, 1, 3], [35, 30, 10, 25], 25)
    False
    """
    if isinstance(valuation_func, list):
        valuation_func = valuation_func.__getitem__

    items_value = sum(valuation_func(item) for item in bundle)
    if items_value < prop_share:
        return False
    for item in bundle:
        if items_value - valuation_func(item) >= prop_share:
            return False
    return True


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
