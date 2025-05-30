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
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from itertools import chain
from math import isclose
import logging, cvxpy as cp

logging.basicConfig(
    format="{asctime} - {levelname} - {message}", style="{", level=logging.DEBUG
)
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

    max_rank = max(map(len, minimal_bundles_by_agent.values())) - 1

    logger.info(
        "start allocating with 'fall back' strategey. stop when reach rank %s or found complete allocation",
        max_rank,
    )

    rank = max_agents_received = 0

    # while any agent can compromise and at least one agent didn't receive a bundle
    while rank <= max_rank and max_agents_received < len(agents):

        logger.debug("max rank of minimal bundles allowed: %d", rank)

        # descend by one level while updating the relevant constraints
        update_bundles_variables(
            minimal_bundles_by_agent,
            bundles_vars_by_agent,
            bundles_vars_by_item,
            rank,
        )

        curr_alloc = find_max_alloc(bundles_vars_by_agent, bundles_vars_by_item)
        num_agents_received = len(curr_alloc)
        logger.debug("number af agents who received bundles: %d", num_agents_received)
        logger.debug("found allocation (value := bundle rank): %s", curr_alloc)

        if num_agents_received > max_agents_received:
            maxmin_alloc = curr_alloc
            max_agents_received = num_agents_received
            logger.debug("------New Maxmin allocation found------")

        rank += 1

    logger.info(
        "stopped descending in rank %s. number of agents who received bundles: %d",
        rank - 1,
        max_agents_received,
    )

    logger.info("finding the pareto optimal solution...")
    pareto_optimal_alloc = find_pareto_optimal_alloc(
        bundles_vars_by_agent, bundles_vars_by_item, maxmin_alloc
    )
    logger.info("Pareto optimal solution found: %s", pareto_optimal_alloc)
    # finally, give the bundles for the agents
    for agent, rank in pareto_optimal_alloc.items():
        alloc.give_bundle(agent, minimal_bundles_by_agent[agent][rank])

    logger.info("Final allocation: %s", alloc.sorted())


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


def update_bundles_variables(
    min_bundles: Mapping[Hashable, list[Iterable]],
    bundle_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
    rank: int,
) -> None:
    """
    checks for each agent if he has a minimal bundle at the current iteration's level \\
    and if so creates a cvxpy Variable as an indicator for the Maximization problem.
    """
    for agent, minimal_bundles in min_bundles.items():
        #  if agent can has a bundle in this rank
        if rank < len(minimal_bundles):
            # Inidicator for this specific bundle
            bundle_var = cp.Variable(boolean=True)
            bundle_vars_by_agent[agent].append(bundle_var)
            for item in minimal_bundles[rank]:
                bundles_vars_by_item[item].append(bundle_var)


def find_max_alloc(
    bundles_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
) -> Mapping[Hashable, int]:
    """
    Finds allocation which maximizes the amount of agents who receive bundle

    with the following constraints:

    1. Each player can receive at most one bundle
    2. Two bundles with at least one shared item cannot both be given


    Args:
        bundle_indicators_by_agent (BundleVarsMapper): map each agent to it's list of minimal bundles Variables
        bundles_constraints (BundleVarsMapper): map each item to the list of bundles that contain it

    Returns:
       Mapping[Hashable, int]: Mapper from agent to the rank of the bundle given to him
    """
    agents_constraint = list(map(lambda x: sum(x) <= 1, bundles_vars_by_agent.values()))
    items_constraints = list(map(lambda x: sum(x) <= 1, bundles_vars_by_item.values()))

    constraints = agents_constraint + items_constraints

    # collect of all the bundles Variabls to one list
    all_bundles_vars = list(chain.from_iterable(bundles_vars_by_agent.values()))

    objective = cp.Maximize(cp.sum(all_bundles_vars))

    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    # Extract soliution from cvxpy Variables
    alloc_by_rank = {}
    for agent, mb_vars in bundles_vars_by_agent.items():
        for rank, bundle_var in enumerate(mb_vars):
            if isclose(bundle_var.value, 1):
                alloc_by_rank[agent] = rank
    return alloc_by_rank


def find_pareto_optimal_alloc(
    bundles_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
    maxmin_alloc: Mapping[Hashable, int],
) -> Mapping[Hashable, int]:

    # 1. to keep the solution maxminimall, discard bundle's variables
    # with rank worse than the rank in the maxmin allocation

    logger.info("Search allocation that pareto dominates allocation: %s", maxmin_alloc)
    updated_vars_by_agent = {}
    for agent, rank in maxmin_alloc.items():
        updated_vars_by_agent[agent] = bundles_vars_by_agent[agent][: rank + 1]

    agents_constraint = list(map(lambda x: sum(x) <= 1, updated_vars_by_agent.values()))
    items_constraints = list(map(lambda x: sum(x) <= 1, bundles_vars_by_item.values()))

    # collect of all the bundles Variabls to one list
    all_bundles_vars = list(chain.from_iterable(updated_vars_by_agent.values()))
    # num of agents that must receive a bundle
    num_agents_received_constraint = [sum(all_bundles_vars) == len(maxmin_alloc)]

    constraints = agents_constraint + items_constraints + num_agents_received_constraint

    # find allocation that pareto dominates the maxmin allocation given
    # by assigning weight to each bundle variable from 0 (most prefferd bundle)
    # to -(agents-num-of-min-bundles). If such allocation exists cvxpy Maximization
    # problem should find it

    weighted_vars = []
    for bundle_vars in updated_vars_by_agent.values():
        for rank, bv in enumerate(bundle_vars):
            weighted_vars.append(-rank * bv)

    objective = cp.Maximize(sum(weighted_vars))
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()

    # Extract solution from cvxpy Variables
    ranking_alloc = {}
    for agent, mb_vars in updated_vars_by_agent.items():
        for rank, bundle_var in enumerate(mb_vars):
            if isclose(bundle_var.value, 1):
                ranking_alloc[agent] = rank
    return ranking_alloc


if __name__ == "__main__":
    import doctest

    # Delete if you want to see the backtrack() func DEBUG messages
    logger.addFilter(lambda record: record.funcName != "backtrack")

    # print(doctest.testmod())
    nagents, nitems = 7, 10
    title = f"Large Input. Agents - {nagents}. Items - {nitems}"
    instance = Instance.random_uniform(
        num_of_agents=nagents,
        num_of_items=nitems,
        item_capacity_bounds=(1, 1),
        agent_capacity_bounds=(nitems, nitems),
        item_base_value_bounds=(20, 120),
        item_subjective_ratio_bounds=(0.5, 2.4),
        normalized_sum_of_values=1000,
        random_seed=45335,
    )
    divide(maximally_proportional_allocation, instance)
