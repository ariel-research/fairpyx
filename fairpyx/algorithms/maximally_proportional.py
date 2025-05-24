"""
An implementation of the algorithm in:
"An algorithm for the proportional division of indivisible items"
by Brams, Steven J. and Kilgour, D. Marc and Klamler, Christian (2014),
http://https://mpra.ub.uni-muenchen.de/56587/

Programmer: Elroi Carmel
Date: 2025-05
"""

from fairpyx import Instance, AllocationBuilder, divide
from typing import Any, Optional
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
import logging, cvxpy as cp, numpy as np

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

    agents = alloc.remaining_agents()
    minimal_bundles_by_agent = {
        agent: get_minimal_bundles(alloc, agent, sort_bundle=False) for agent in agents
    }
    logger.info("collected minimal bundles by order of priorities for all agents")

    # Each minimal bundle will be represented by a boolean Variable instance that will be used
    # by cvxpy module to maximize the amount of bundles given at each iteration

    # maps each agent to it's list of minimal bundles Variables
    # used also for constraint so that each agent won't get more than one bundle
    bundles_vars_by_agent = defaultdict(list)

    # maps each item to the list of bundles that contain it
    # used also for constraint so that each item is given exactly one time
    bundles_vars_by_item = defaultdict(list)

    max_rank = max(map(len, minimal_bundles_by_agent.values()))
    logger.info(
        "start allocating with 'fall back' strategey. stop when reach rank %s or found complete allocation",
        max_rank,
    )

    rank = agents_received = 0

    # while any agent can compromise and at least one agent didn't receive a bundle
    while rank < max_rank and agents_received < len(agents):

        logger.debug("max rank of minimal bundles allowed: %d", rank)
        # descend by one level while updating the relevant constraints
        descend(minimal_bundles_by_agent, bundles_vars_by_agent, bundles_vars_by_item)

        agents_received = solve_with_cvxpy(bundles_vars_by_agent, bundles_vars_by_item)
        logger.debug("number af agents who received bundles: %d", agents_received)

        solution = extract_solution(bundles_vars_by_agent)

        logger.debug("current maxmin allocation (value := bundle index): %s", solution)
        rank += 1

    rank -= 1
    logger.info(
        "stopped descending in rank %s. number of agents who received bundles: %d",
        rank,
        agents_received,
    )

    logger.info("finding the pareto optimal solution...")

    solve_with_cvxpy(
        bundles_vars_by_agent,
        bundles_vars_by_item,
        pareto_optimal_target=agents_received,
    )

    ranking_solution = extract_solution(bundles_vars_by_agent)
    logger.info("Pareto optimal solution found: %s", ranking_solution)
    # finally, give the bundles for the agents
    for agent, rank in ranking_solution.items():
        alloc.give_bundle(agent, minimal_bundles_by_agent[agent][rank])


def solve_with_cvxpy(
    bundles_vars_by_agent: dict,
    bundles_vars_by_item: dict,
    pareto_optimal_target: Optional[int] = None,
) -> int:
    """
    Maximizing the amount of agents receiving minimal bundles

    with the following constraints:

    1. Each player can receive at most one bundle
    2. Two bundles with at least one shared item cannot both be given

    Once the maximum amount of agents that can be given a bundle is found,

    the paramater *pareto_optimal_target* can be used to find the pareto optimal solution


    Args:
        bundle_indicators_by_agent (dict[Hashable, list[cp.Variable]]): map each agent to it's list of minimal bundles Variables
        bundles_constraints (dict[Hashable, Iterable[cp.Variable]]): map each item to the list of bundles that contain it
        pareto_optimal_target (Optional[int]): amount of agents that need to get a bundle in the pareto optimal solution

    Returns:
       int: amount of agents who received bundles in this solution
    """
    agents_constraint = list(map(lambda x: sum(x) <= 1, bundles_vars_by_agent.values()))
    items_constraints = list(map(lambda x: sum(x) <= 1, bundles_vars_by_item.values()))

    constraints = agents_constraint + items_constraints

    # collect of all the bundles Variabls to one list
    all_bundles_vars = list(chain.from_iterable(bundles_vars_by_agent.values()))

    if pareto_optimal_target is None:
        # no pareto optimization, simply maximize amount of agents who receive bundles
        objective = cp.Maximize(cp.sum(all_bundles_vars))
    else:
        # fix the amount of agents receiving a bundle
        constraints.append(cp.sum(all_bundles_vars) == pareto_optimal_target)
        # weight of bundle = how much the agents prefers it
        weighted_vars = []
        for min_bun in bundles_vars_by_agent.values():
            for i, bundle_var in enumerate(min_bun):
                # 0 -> most prefered bundle
                weighted_vars.append(-i * bundle_var)
        objective = cp.Maximize(cp.sum(weighted_vars))

    prob = cp.Problem(objective=objective, constraints=constraints)
    agents_received = prob.solve()
    return round(agents_received)


def descend(min_bundles, bundle_vars_by_agent, bundles_vars_by_item) -> None:
    """
    checks for each agent if he has a minimal bundle at the current iteration's level \\
    and if so creates a cvxpy Variable as an indicator for the Maximization problem.
    """
    for agent, minimal_bundles in min_bundles.items():
        curr_num_of_bundles = len(bundle_vars_by_agent[agent])
        #  if agent can compromise by one level:
        if curr_num_of_bundles < len(minimal_bundles):
            # Inidicator for this specific bundle
            bundle_var = cp.Variable(boolean=True)
            bundle_vars_by_agent[agent].append(bundle_var)
            for item in minimal_bundles[curr_num_of_bundles]:  # used for constraints
                bundles_vars_by_item[item].append(bundle_var)


def extract_solution(bundles_vars_by_agent: dict[Hashable, list[cp.Variable]]):
    """
    To be used *after* the solve_with_cvxpy() was called on the bundle_indicators_by_agent


    Returns:
        dict: mapper from agent to his allocated minimal bundle's rank
    """
    res = {}
    for agent, bundle_indicators in bundles_vars_by_agent.items():
        for i, indicator in enumerate(bundle_indicators):
            if np.isclose(indicator.value, 1):
                res[agent] = i
    return res


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
    logger.info(' collecting all of agent "%s" minimal bundles '.center(50, "#"), agent)
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


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
