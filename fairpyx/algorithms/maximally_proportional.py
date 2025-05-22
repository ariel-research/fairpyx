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
from pprint import pformat
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

    minimal_bundles_by_agent = {
        agent: get_minimal_bundles(alloc, agent, sort_bundle=False)
        for agent in alloc.remaining_agents()
    }
    logger.info("collected minimal bundles by order of priorities for all agents")

    max_rank = max(map(len, minimal_bundles_by_agent.values()))
    max_agents_received = 0
    rank = 0

    # Each minimal bundle will be represented by a boolean Variable instance that will be used
    # by cvxpy module to maximize the amount of bundles given at each iteration

    # map each agent to it's list of minimal bundles Variables
    bundle_indicators_by_agent = defaultdict(list)
    # map each item to the list of bundles that contain it
    bundles_conflicts_by_item = defaultdict(list)

    logger.info(
        "start allocating with 'fall back' strategey. stop when reach rank %s or found complete allocation",
        max_rank,
    )

    nagents = len(alloc.remaining_agents())
    keep_descending = True
    while keep_descending:

        if rank >= max_rank or max_agents_received == nagents:
            keep_descending = False
        else:
            logger.debug("max rank of minimal bundles allowed: %d", rank)
            # Update constraints of agents and colliding bundles by current level
            update_constraints_and_variables(
                minimal_bundles_by_agent,
                rank,
                bundle_indicators_by_agent,
                bundles_conflicts_by_item,
            )

            agents_received = solve_with_cvxpy(
                bundle_indicators_by_agent, bundles_conflicts_by_item
            )
            logger.debug("number af agents who received bundles: %d", agents_received)

            solution = extract_solution(bundle_indicators_by_agent)

            logger.debug(
                "current maxmin allocation (value := bundle index): %s", solution
            )
            if agents_received > max_agents_received:  # type: ignore
                max_agents_received = agents_received
                maxmin_solution = solution
            rank += 1

    rank -= 1
    logger.info(
        "stopped descending in rank %s. number of agents who received bundles: %d",
        rank,
        max_agents_received,
    )

    logger.info("checking if %s is pareto optimal...", maxmin_solution)
    res = get_pareto_optimal(
        bundle_indicators_by_agent, maxmin_solution, bundles_conflicts_by_item
    )
    if res != maxmin_solution:
        logger.info("found a pareto optimal!: %s", res)
    else:
        logger.info("this is pareto optimal")

    # finally give the bundles for the agents
    for agent, rank in res.items():
        alloc.give_bundle(agent, minimal_bundles_by_agent[agent][rank])


def solve_with_cvxpy(
    bundle_indicators_by_agent,
    bundles_constraints,
    prev_solutions_constraints: Optional[list] = None,
) -> int:
    """
    Maximizing the amount of agents receiving minimal bundles,
    while avoiding giving a player more than one bundle and giving bundles
    having the same item, using the cvxpy formulation.
    Accept an optional constraints s.t. the output is not identical to one of the previous solution.

    Args:
        bundle_indicators_by_agent (dict[Hashable, list[cp.Variable]]): map each agent to it's list of minimal bundles Variables
        bundles_constraints (dict[Hashable, Iterable[cp.Variable]]): map each item to the list of bundles that contain it
        prev_solutions_constraints (Optional[list], optional): List of constraints represnting the previous solutions. Defaults to None.

    Returns:
       int: amount of agents who received bundles in this solution
    """
    constraints = [
        sum(constraint) <= 1
        for constraint in chain(
            bundle_indicators_by_agent.values(), bundles_constraints.values()
        )
    ]
    if prev_solutions_constraints is not None:
        constraints += prev_solutions_constraints
    objective = cp.Maximize(sum(chain(*bundle_indicators_by_agent.values())))
    prob = cp.Problem(objective=objective, constraints=constraints)  # type: ignore
    agents_received = prob.solve()
    return round(agents_received)  # type: ignore


def update_constraints_and_variables(
    min_bundles, rank, bundle_indicators_by_agent, items_constraints
):
    """
    checks for each agent if he has a minimal bundle at the current iteration's level
    and if so creates a cvxpy Variable as an indicator for the Maximization problem.
    """
    for agent, minimal_bundles in min_bundles.items():
        if rank in range(len(minimal_bundles)):
            bundle_var = cp.Variable(boolean=True)
            bundle_indicators_by_agent[agent].append(bundle_var)
            for item in minimal_bundles[rank]:
                items_constraints[item].append(bundle_var)


def extract_solution(bundle_indicators_by_agent: dict[Hashable, list[cp.Variable]]):
    """
    To be used *after* the solve_with_cvxpy() was called on the bundle_indicators_by_agent


    Returns:
        dict: mapper from agent to his allocated minimal bundle's rank
    """
    res = {}
    for agent, bundle_indicators in bundle_indicators_by_agent.items():
        for i, indicator in enumerate(bundle_indicators):
            if np.isclose(indicator.value, 1):  # type: ignore
                res[agent] = i
    return res


def get_pareto_optimal(
    bundle_indicators_by_agent: dict,
    baseline_solution: dict,
    items_constraints,
) -> dict:
    """
    An allocation is pareto optimal if there's no other allocation which pareto dominates it.
    Allocation A pareto dominates allocation B iff it doesn't harm anyone while benfiets at
    least one agent.
    Given a start baseline solution the algorithm will try to find an allocation which pareto
    dominates it and will keep searching until there's no room for improvement.

    Args:
        bundle_indicators_by_agent (dict): used for the cvxpy solve
        baseline_solution (dict): map from agent to rank of minimal bundle
        items_constraints (_type_): used for the cvxpy solve

    Returns:
        dict: pareto optimal solution
    """
    prev_solutions_constraints = []
    search = True
    optimal_agents_received = len(baseline_solution)
    logger.debug("try to dominate: %s", baseline_solution)
    while search:

        # choose for each player a bundle that's at least good as the one he got in previous solution
        temp = {}
        for agent, rank in baseline_solution.items():
            temp[agent] = bundle_indicators_by_agent[agent][: rank + 1]
        bundle_indicators_by_agent = temp

        # add a constraint to NOT repeat previous solution
        prev_solutions_constraints.append(
            sum(map(lambda x: x[-1], bundle_indicators_by_agent.values()))
            <= optimal_agents_received - 1
        )

        agents_received = solve_with_cvxpy(
            bundle_indicators_by_agent, items_constraints, prev_solutions_constraints
        )

        if agents_received < optimal_agents_received:
            search = False
        else:
            curr_sol = extract_solution(bundle_indicators_by_agent)
            logger.debug("found dominating allocation: %s", curr_sol)
            logger.debug(
                "agents with better assignments: %s",
                list(
                    agent
                    for agent in curr_sol
                    if curr_sol[agent] < baseline_solution[agent]
                ),
            )
            baseline_solution = curr_sol

    return baseline_solution


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
