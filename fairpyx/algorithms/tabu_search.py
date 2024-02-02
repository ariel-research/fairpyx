"""
Implement "Tabu search" course allocation,

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
from fairpyx import Instance


def tabu_search(instance: Instance, initial_budgets: list, beta: float):
    """
   "Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
    by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
    ALGORITHM 3: Tabu search

   :param instance: a fair-course-allocation instance
   :param initial_budgets: Students' initial budgets, b_0∈[1,1+β]^n
   :param beta: creates the range of initial_budgets

   :return final courses prices, final distribution

    >>> from fairpyx.adaptors import divide
    >>> from fairpyx.utils.test_utils import stringify
    >>> from fairpyx import Instance

    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets={5, 4, 3}
    >>> beta = 4
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x', 'z'], tzumi:['x', 'z'] }"

    >>> instance = Instance(
    ... valuations={"ami":{"x":5, "y":4, "z":3, "w":2}, "tami":{"x":5, "y":2, "z":4, "w":3}},
    ... agent_capacities=3,
    ... item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets={8, 6}
    >>> beta = 9
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['x','y','z'], tami:['x', 'z', 'w']}"

    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":3, "z":3}, "tami":{"x":3, "y":3, "z":3}, "tzumi":{"x":4, "y":4, "z":4}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> initial_budgets={4, 5, 2}
    >>> beta = 5
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x', 'w'], tzumi:['y', 'z'] }"

    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":3})
    >>> initial_budgets={6, 4}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['x','y'], tami:['y', 'z']}"

    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":1, "z":1})
    >>> initial_budgets={5, 3}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x']}"
    """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
