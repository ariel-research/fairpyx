"""
An implementation of the algorithms in:

"Efficient Nearly-Fair Division with Capacity Constraints", by A. Hila Shoshan,  Noam Hazon,  Erel Segal-Halevi
(2023), https://arxiv.org/abs/2205.07779

Programmer: Matan Ziv.
Date: 2025-04-27.
"""

# The end-users of the algorithm feed the input into an "Instance" variable, which tracks the original input (agents, items and their capacities).
# But the algorithm implementation uses an "AllocationBuilder" variable, which tracks both the ongoing allocation and the remaining input (the remaining capacities of agents and items).
# The function `divide` is an adaptor - it converts an Instance to an AllocationBuilder with an empty allocation.
from fairpyx import Instance, AllocationBuilder, divide


# The end-users of the algorithm feed the input into an "Instance" variable, which tracks the original input (agents, items and their capacities).
# The input of the algorithm is:
# a dict of a dict witch tells the valuations of the items for each player
# a dict of a sets, dict of category each category is a set contains the items in the category
# a dict of categories and the capacities

# the returned value of the algorithem is


# The `logging` facility is used both for debugging and for illustrating the steps of the algorithm.
# It can be used to automatically generate running examples or explanations.
import logging

logger = logging.getLogger(__name__)

# ---------------------------
# Example instance with categories and capacities:
# ---------------------------
example_instance = Instance(
    valuations={
        "Agent1": {"o1": 0, "o2": -1, "o3": -4, "o4": -5, "o5": 0, "o6": 2},
        "Agent2": {"o1": 0, "o2": -1, "o3": -2, "o4": -1, "o5": -1, "o6": 0},
    },
    # agent_capacities={"Agent1": 3, "Agent2": 3},
    item_categories={
        "o1": "cat1", "o2": "cat1", "o3": "cat1", "o4": "cat1",  # 4 items in category 1
        "o5": "cat2", "o6": "cat2"                               # 2 items in category 2
    },
    item_capacities={item: 1 for item in item_categories.keys()},
    # item_capacities={"o1": 1, "o2": 1, "o3": 1, "o4": 1, "o5": 1, "o6": 1},
    category_capacities={"cat1": 2, "cat2": 1},  # each agent can get at most 2 from cat1, 1 from cat2
)


def fair_capacity_algorithm(alloc: AllocationBuilder) -> None:
    """
    Implements the EF[1,1] + PO allocation algorithm from:
    "Efficient Nearly-Fair Division with Capacity Constraints" (AAMAS 2023).

    Parameters:
        alloc (AllocationBuilder): allocation builder with capacity-tracked state

    Returns:
        None (modifies alloc in place)

    Examples and tests:
    >>> divide(fair_capacity_algorithm, example_instance)
    {'Agent1': ['o1', 'o2', 'o5'], 'Agent2': ['o3', 'o4', 'o6']}

    >>> instance2 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1},
    ...         "B": {"o1": -1},
    ...     },
    ...     item_capacities={"o1": 1},
    ...     item_categories={"o1": "g"},
    ...     category_capacities={"g": 1},
    ... )
    >>> divide(fair_capacity_algorithm, instance2)
    {'A': ['o1'], 'B': []}


    >>> instance3 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1, "o2": -1},
    ...         "B": {"o1": 1, "o2": -1},
    ...     },
    ...     item_capacities={"o1": 1, "o2": 1},
    ...     item_categories={"o1": "g", "o2": "g"},
    ...     category_capacities={"g": 1},
    ... )
    >>> divide(fair_capacity_algorithm, instance2)
    {'A': ['o1'], 'B': ['o2']}

    >>> instance4 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1, "o2": -1, "o3": 1, "o4": 1.5},
    ...         "B": {"o1": 1, "o2": -1, "o3": 1, "o4": 1},
    ...     },
    ...     item_capacities={"o1": 1, "o2": 1, "o3": 1, "o4": 1},
    ...     item_categories={"o1": "g", "o2": "g", "o3": "c", "o4": "c"},
    ...     category_capacities={"g": 1, "c": 2},
    ... )
    >>> divide(fair_capacity_algorithm, instance2)
    {'A': ['o1', 'o3'], 'B': ['o2', 'o4']}

    >>> instance5 = Instance(
    ...     valuations={
    ...         "A": {"o1": 0, "o2": -1, "o3": -4, "o4": -5, "o5": 0, "o6": 2},
    ...         "B": {"o1": 0, "o2": -1, "o3": -2, "o4": -1, "o5": -1, "o6": 0},
    ...     },
    ...     item_capacities={"o1": 1, "o2": 1, "o3": 1,"o4": 1, "o5": 1, "o6": 1},
    ...     item_categories={"o1": "1", "o2": "1", "o3": "1", "o4": "1", "o5": "2", "o6": "2"},
    ...     category_capacities={"1": 2, "2": 1},
    ... )
    >>> divide(fair_capacity_algorithm, instance2)
    {'A': ['o1', 'o2', 'o5'], 'B': ['o3', 'o4', 'o6']}

    """
    pass
    return 0  # Empty implementation


if __name__ == "__main__":
    import doctest
    import sys
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    print(doctest.testmod())

