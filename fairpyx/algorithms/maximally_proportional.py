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
import logging

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
    {0: [2], 1: [0, 3],  2: [1, 4]}

    """
    pass


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
    [[0, 1], [0, 3], [1, 3]]
    >>> get_minimal_bundles(alloc, 1)
    [[0, 1, 3], [0, 2], [1, 2], [2, 3]]
    >>> get_minimal_bundles(alloc, 2)
    [[1, 3], [0, 1, 2], [0, 3], [2, 3]]
    >>> get_minimal_bundles(alloc, 3)
    [[0, 3], [0, 1], [1, 3]]
    """
    pass


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
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
