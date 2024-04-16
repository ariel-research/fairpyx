"""
Implement "Tabu search" course allocation,

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
import logging
import random
from itertools import combinations

import numpy as np

from fairpyx import Instance

logger = logging.getLogger(__name__)


# TODO: add doco
def excess_demand(instance: Instance, allocation: dict):
    z = {}  # Initialize z as a dictionary
    for course in instance.items:
        sum_allocation = 0
        for student in range(instance.num_of_agents):
            sum_allocation += allocation[student][course]
        z[course] = sum_allocation - instance.item_capacity[course]
    return z


# TODO: add doco
def clipped_excess_demand(instance: Instance, initial_budgets: dict, prices: dict):
    z = excess_demand(instance, initial_budgets)
    clipped_z = {course: max(0, z[course]) if prices[course] == 0 else z[course] for course in z}
    return clipped_z


def student_best_bundle(prices: dict, instance: Instance, initial_budgets: dict):
    """
    Return a dict that says for each student what is the bundle with the maximum utility that a student can take

    :param prices: dictionary with courses prices
    :param instance: fair-course-allocation instance
    :param initial_budgets: students' initial budgets

    :return: a dictionary that maps each student to its best bundle.

     Example run 1 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":4, "z":2}, "Bob":{"x":4, "y":3, "z":2}, "Eve":{"x":2, "y":4, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets = {"Alice": 5, "Bob": 4, "Eve": 3}
    >>> prices = {"x": 1, "y": 2, "z": 1}
    >>> student_best_bundle(prices, instance, initial_budgets)
    {'Alice': ('x', 'y'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}

     Example run 2 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":3, "w":2}, "Bob":{"x":5, "y":2, "z":4, "w":3}},
    ...     agent_capacities=3,
    ...     item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets = {"Alice": 8, "Bob": 6}
    >>> prices = {"x": 1, "y": 2, "z": 3, "w":4}
    >>> student_best_bundle(prices, instance, initial_budgets)
    {'Alice': ('x', 'y', 'z'), 'Bob': ('x', 'y', 'z')}


    Example run 3 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":3, "z":3, "w":3}, "Bob":{"x":3, "y":3, "z":3, "w":3}, "Eve":{"x":4, "y":4, "z":4, "w":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> initial_budgets = {"Alice": 4, "Bob": 5, "Eve": 2}
    >>> prices = {"x": 1, "y": 1, "z": 1, "w":1}
    >>> student_best_bundle(prices, instance, initial_budgets)
    {'Alice': ('x', 'y'), 'Bob': ('x', 'y'), 'Eve': ('x', 'y')}

    """
    best_bundle = {student: None for student in instance.agents}
    logger.info("START combinations")

    for student in instance.agents:

        # Creating a list of combinations of courses up to the size of the student's capacity
        combinations_courses_list = []
        capacity = instance.agent_capacity(student)
        for r in range(1, capacity + 1):
            combinations_courses_list.extend(combinations(instance.items, r))
        logger.info(f"FINISH combinations for {student}")

        # Define a lambda function that calculates the valuation of a combination
        valuation_function = lambda combination: instance.agent_bundle_value(student, combination)

        # Sort the combinations_set based on their valuations in descending order
        combinations_courses_sorted = sorted(combinations_courses_list, key=valuation_function, reverse=True)

        for combination in combinations_courses_sorted:
            price_combination = sum(prices[course] for course in combination)
            if price_combination <= initial_budgets[student]:
                best_bundle[student] = combination
                break

    return best_bundle


def find_all_equivalent_prices(instance: Instance, history: list, prices: dict):
    """
    Update history - list of all equivalent prices of 𝒑

    :param instance: fair-course-allocation
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices

    :return: None
    """

    pass

def find_gradient_neighbors(neighbors: list, history: list, prices: dict, delta: float, excess_demand_vector: dict):
    #TODO ask erel about delta
    """
    Add the gradient neighbors to the neighbors list

    :param neighbors: list of Gradient neighbors and Individual price adjustment neighbors.
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param delta:
    :param excess_demand_vector: excess demand of the courses
    :return: None

    Example run 1 iteration 1
    >>> neighbors = []
    >>> history = []
    >>> prices = {"x": 1, "y": 2, "z": 1}
    >>> delta = 1
    >>> excess_demand_vector = {"x":0,"y":2,"z":-2}
    >>> find_gradient_neighbors(neighbors,history,prices,delta,excess_demand_vector)
    {'x': 1, 'y': 4, 'z': 0}


     Example run 1 iteration 2
    >>> neighbors = []
    >>> history = [
    ...    {'x': 1, 'y': 2, 'z': 1}, {'x': 0, 'y': 0, 'z': 0}, {'x': 1, 'y': 0, 'z': 0},
    ...    {'x': 0, 'y': 1, 'z': 0}, {'x': 0, 'y': 0, 'z': 1}, {'x': 1, 'y': 1, 'z': 0},
    ...    {'x': 1, 'y': 0, 'z': 1}, {'x': 0, 'y': 1, 'z': 1}, {'x': 1, 'y': 1, 'z': 1},
    ...    {'x': 0, 'y': 1, 'z': 2}, {'x': 0, 'y': 2, 'z': 1}, {'x': 1, 'y': 0, 'z': 2},
    ...    {'x': 1, 'y': 2, 'z': 0}, {'x': 2, 'y': 0, 'z': 1}, {'x': 2, 'y': 1, 'z': 0},
    ...    {'x': 2, 'y': 2, 'z': 0}, {'x': 2, 'y': 2, 'z': 1}, {'x': 1, 'y': 1, 'z': 2},
    ...    {'x': 2, 'y': 1, 'z': 1}, {'x': 3, 'y': 0, 'z': 0}, {'x': 3, 'y': 1, 'z': 0},
    ...    {'x': 3, 'y': 0, 'z': 1}, {'x': 3, 'y': 1, 'z': 1}, {'x': 0, 'y': 3, 'z': 0},
    ...    {'x': 1, 'y': 3, 'z': 0}, {'x': 0, 'y': 0, 'z': 3}, {'x': 1, 'y': 0, 'z': 3},
    ...   {'x': 2, 'y': 0, 'z': 3}, {'x': 3, 'y': 0, 'z': 3}, {'x': 4, 'y': 0, 'z': 3},
    ...   {'x': 1, 'y': 4, 'z': 0}, {'x': 2, 'y': 3, 'z': 1}, {'x': 0, 'y': 5, 'z': 0}]
    >>> prices = {"x": 1, "y": 4, "z": 0}
    >>> delta = 1
    >>> excess_demand_vector = {"x":1,"y":0,"z":0}
    >>> find_gradient_neighbors(neighbors,history,prices,delta,excess_demand_vector)
    {'x': 2, 'y': 4, 'z': 0}
    """
    # N gradient(𝒑, Δ) = {𝒑 + 𝛿 · 𝒛(𝒖,𝒄, 𝒑, 𝒃) : 𝛿 ∈ Δ} .
    updated_prices = {}
    for item, price in prices.items():
        updated_prices[item] = max(0, price + delta * excess_demand_vector.get(item, 0))

    if updated_prices not in history:
        neighbors.append(updated_prices)

    return updated_prices
def find_individual_price_adjustment_neighbors(instance: Instance, neighbors: list, history: list, prices: dict,
                                               excess_demand_vector: dict):
    pass


def find_all_neighbors(instance: Instance, neighbors: list, history: list, prices: dict, delta: float,
                       excess_demand_vector: dict):
    # todo: ask erel about delta
    """
    Update neighbors N (𝒑) - list of Gradient neighbors and Individual price adjustment neighbors.

    :param instance: fair-course-allocation
    :param neighbors: list of Gradient neighbors and Individual price adjustment neighbors.
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param delta:
    """

    find_gradient_neighbors(neighbors, history, prices, delta, excess_demand_vector)
    find_individual_price_adjustment_neighbors(instance, neighbors, history, prices, excess_demand_vector)  # TODO: implement


def find_min_error_prices(instance: Instance, prices: dict, neighbors: list, initial_budgets: dict):
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
    norma2 = 1
    while norma2:
        neighbors = []  # resets on every iteration
        allocation = student_best_bundle(prices, instance, initial_budgets)
        excess_demand_vector = clipped_excess_demand(instance, initial_budgets, prices)
        values = np.array(list(excess_demand_vector.values()))
        norma2 = np.linalg.norm(values)
        # 3) Otherwise,
        # • include all equivalent prices of 𝒑 into the history: H ← H + {𝒑′ : 𝒑′ ∼𝑝 𝒑},
        find_all_equivalent_prices(instance, history, prices)  # TODO - implement
        delta = 1  # TODO- ask erel how to get delta
        find_all_neighbors(instance, neighbors, history, prices, delta, excess_demand_vector)  # TODO - implement

        # • update 𝒑 ← arg min𝒑′∈N (𝒑)−H ∥𝒛(𝒖,𝒄, 𝒑', 𝒃0)∥2, and then
        find_min_error_prices(instance, prices, neighbors, initial_budgets)

    return allocation  # TODO: print p*


if __name__ == "__main__":
    import doctest

    doctest.testmod()