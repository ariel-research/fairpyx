"""
Implement "Tabu search" course allocation,

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
import random

from fairpyx import Instance


def excess_demand(instance, initial_budgets: dict, prices, allocation):
    z = [0] * instance.num_of_items  # in size of the number of courses
    for course in range(instance.num_of_items):
        sum = 0
        for student in range(instance.num_of_agents):
            sum += allocation[student][course]
        z[course] = sum - instance.item_capacity[course]
    return z


def clipped_excess_demand(instance, initial_budgets, prices, allocation):
    z = excess_demand(instance, initial_budgets, prices, allocation)
    clipped_z = [max(0, z[i]) if prices[i] == 0 else z[i] for i in range(len(z))]
    return clipped_z


def student_best_bundle(prices: dict, instance: Instance, initial_budgets: dict):
    """
    Return a dict that says for each student what is the bundle with the maximum utility that a student can take

    :param prices: dictionary with courses prices
    :param instance: fair-course-allocation instance
    :param initial_budgets: students' initial budgets

    :return: a dictionary that maps each student to its best bundle.
    """
    pass


def find_all_equivalent_prices(instance: Instance, history: list, prices: dict):
    """
    Update history - list of all equivalent prices of 𝒑

    :param instance: fair-course-allocation
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices

    :return: None
    """
    pass


def find_all_neighbors(instance: Instance, neighbors: list, history:list,  prices: dict, delta: float):
    # todo: ask erel about delta
    """
    Update neighbors N (𝒑) - list of Gradient neighbors and Individual price adjustment neighbors.

    :param instance: fair-course-allocation
    :param neighbors: list of Gradient neighbors and Individual price adjustment neighbors.
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param delta:
    """
    pass


def find_min_error_prices(instance:Instance, prices:dict, neighbors:list, initial_budgets:dict):
    """
    Return the update prices that minimize the market clearing error.

    :param instance: fair-course-allocation
    :param prices: dictionary with course prices
    :param neighbors: list of Gradient neighbors and Individual price adjustment neighbors.
    :param initial_budgets: students' initial budgets

    :return: update prices
    """
    pass


def tabu_search(instance: Instance, initial_budgets: dict, beta: float):
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
    >>> initial_budgets={"ami":5, "tami":4, "tzumi":3}
    >>> beta = 4
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x', 'z'], tzumi:['x', 'z'] }"

    >>> instance = Instance(
    ... valuations={"ami":{"x":5, "y":4, "z":3, "w":2}, "tami":{"x":5, "y":2, "z":4, "w":3}},
    ... agent_capacities=3,
    ... item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets={"ami":8, "tami":6}
    >>> beta = 9
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['x','y','z'], tami:['x', 'z', 'w']}"

    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":3, "z":3}, "tami":{"x":3, "y":3, "z":3}, "tzumi":{"x":4, "y":4, "z":4}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> initial_budgets={"ami":4, "tami":5, "tzumi":2}
    >>> beta = 5
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x', 'w'], tzumi:['y', 'z'] }"

    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":3})
    >>> initial_budgets={"ami":6, "tami":4}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['x','y'], tami:['y', 'z']}"

    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":1, "z":1})
    >>> initial_budgets={"ami":5, "tami":3}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta))
    "{ami:['y','z'], tami:['x']}"
    """
    # 1) Let 𝒑 ← uniform(1, 1 + 𝛽)^𝑚, H ← ∅.
    prices = {course: random.uniform(1, 1 + beta) for course in instance.items}
    history = []

    # 2)  If ∥𝒛(𝒖,𝒄, 𝒑, 𝒃0)∥2 = 0, terminate with 𝒑∗ = 𝒑.
    excess_demand_val = 1
    while excess_demand_val:
        neighbors = []  # resets on every iteration
        allocation = student_best_bundle(prices, instance, initial_budgets)  # TODO - implement
        excess_demand_val = clipped_excess_demand(instance, initial_budgets, prices, allocation)

        # 3) Otherwise,
        # • include all equivalent prices of 𝒑 into the history: H ← H + {𝒑′ : 𝒑′ ∼𝑝 𝒑},
        find_all_equivalent_prices(instance, history, prices)  # TODO - implement
        delta = 1  # TODO- ask erel how to get delta
        find_all_neighbors(instance, neighbors, history, prices, delta) # TODO - implement

        # • update 𝒑 ← arg min𝒑′∈N (𝒑)−H ∥𝒛(𝒖,𝒄, 𝒑', 𝒃0)∥2, and then
        find_min_error_prices(instance, prices, neighbors, initial_budgets)

    return allocation  # TODO: print p*


if __name__ == "__main__":
    import doctest

    doctest.testmod()
