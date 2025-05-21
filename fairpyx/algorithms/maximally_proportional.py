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
from itertools import chain, pairwise
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
    idx = {
        "agents": dict(zip(agents, range(nagents))),
        "items": dict(zip(items, range(nagents, nagents + nitems))),
    }
    # 1. Descend players' ranking until complete allocation is avialaible or reached min rank
    # 2. Remember what is the lowest rank
    # 3. Have lists as building blocks for the sparse matrix
    # 4. Build new optimization probelm at each level
    # 5. Add agent as an index in the matrix rows for constraints simplicity

    keep_descending = True
    indptr, indices = [0], []
    while keep_descending:
        if rank >= max_rank or max_agents_received == len(agents):
            keep_descending = False
        else:
            for agent, mb in min_bundles.items():
                if rank in range(len(mb)):
                    indices.append(idx["agents"][agent])
                    for item in mb[rank]:
                        indices.append(idx["items"][item])
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

    # TODO: Add pareto-optimal choosing

    # Construct the allocation from the sparse matrix and the maxmin_solution variable using the maps
    chosen_columns = [i for i in range(maxmin_solution.size) if np.allclose(maxmin_solution.value[i], 1)]  # type: ignore
    for c in chosen_columns:
        agent, rank = col_to_bundle[c]
        alloc.give_bundle(agent, min_bundles[agent][rank])


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
    True
    >>> is_minimal_bundle([0, 1, 3], [35, 30, 10, 25], 25)
    False
    """
    items_value = sum(valuation_func(item) for item in bundle)
    if items_value < prop_share:
        return False
    for item in bundle:
        if items_value - valuation_func(item) >= prop_share:
            return False
    return True


if __name__ == "__main__":
    import doctest

    # # doctest.testmod(verbose=False)
    # logger.addHandler(logging.StreamHandler())
    # # logger.setLevel(logging.INFO)
    # doctest.run_docstring_examples(
    #     maximally_proportional_allocation, globs=globals(), verbose=True
    # )
