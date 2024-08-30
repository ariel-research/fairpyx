"""
"Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
    by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
    link to the article: https://arxiv.org/pdf/2305.11406
    ALGORITHM 3: Tabu search

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
import logging
import random
from itertools import combinations, product

import numpy as np

from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms.ACEEI_algorithms.calculate_combinations import get_combinations_courses_sorted
from functools import lru_cache

# Setup logger and colored logs
logger = logging.getLogger(__name__)


# ---------------------The main function---------------------

def tabu_search(alloc: AllocationBuilder, **kwargs):
    """
    ALGORITHM 3: Tabu search

   :param alloc: a fair-course-allocation instance
   :param initial_budgets: Students' initial budgets, b_0∈[1,1+β]^n
   :param beta: creates the range of initial_budgets

   :return final courses prices, final distribution

    >>> from fairpyx.adaptors import divide
    >>> from fairpyx.utils.test_utils import stringify
    >>> from fairpyx import Instance

    >>> random.seed(9865)
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets={"ami":5, "tami":4, "tzumi":3}
    >>> beta = 4
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.1, 0.8}))
    "{ami:['y', 'z'], tami:['x', 'z'], tzumi:['x', 'z']}"

    Example run 2
    >>> random.seed(4675)
    >>> instance = Instance(
    ... valuations={"ami":{"x":5, "y":4, "z":3, "w":2}, "tami":{"x":5, "y":2, "z":4, "w":3}},
    ... agent_capacities=3,
    ... item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets={"ami":8, "tami":6}
    >>> beta = 9
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.2}))
    "{ami:['w', 'x', 'y'], tami:['w', 'y', 'z']}"

    >>> random.seed(1805)
    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":3})
    >>> initial_budgets={"ami":6, "tami":4}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.72}))
    "{ami:['x', 'y'], tami:['y', 'z']}"

    >>> random.seed(100)
    >>> instance = Instance(
    ... valuations={"ami":{"x":4, "y":3, "z":2}, "tami":{"x":5, "y":1, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":1, "z":1})
    >>> initial_budgets={"ami":5, "tami":3}
    >>> beta = 6
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.91}))
    "{ami:['x', 'y'], tami:['z']}"

    Example run 3
    >>> random.seed(4341)
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":3, "z":3}, "tami":{"x":3, "y":3, "z":3}, "tzumi":{"x":4, "y":4, "z":4}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":2, "z":2})
    >>> initial_budgets={"ami":4, "tami":5, "tzumi":2}
    >>> beta = 5
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.34}))
    "{ami:['x', 'y'], tami:['y', 'z'], tzumi:['z']}"

    >>> random.seed(0)
    >>> instance = Instance(
    ... valuations={'s1': {'c1': 275, 'c2': 79, 'c3': 59, 'c4': 63, 'c5': 54, 'c6': 226, 'c7': 133, 'c8': 110},
    ...             's2': {'c1': 105, 'c2': 17, 'c3': 222, 'c4': 202, 'c5': 227, 'c6': 89, 'c7': 30, 'c8': 107},
    ...             's3': {'c1': 265, 'c2': 120, 'c3': 37, 'c4': 230, 'c5': 160, 'c6': 44, 'c7': 30, 'c8': 113},
    ...             's4': {'c1': 194, 'c2': 132, 'c3': 224, 'c4': 77, 'c5': 29, 'c6': 230, 'c7': 62, 'c8': 52},
    ...             's5': {'c1': 174, 'c2': 89, 'c3': 229, 'c4': 249, 'c5': 24, 'c6': 83, 'c7': 99, 'c8': 52}},
    ... agent_capacities=5,
    ... item_capacities={'c1': 3.0, 'c2': 3.0, 'c3': 3.0, 'c4': 3.0, 'c5': 3.0, 'c6': 3.0, 'c7': 3.0, 'c8': 3.0})
    >>> initial_budgets = {'s1': 1.0005695511898616, 's2': 1.0009070710569965, 's3': 1.000699704772071,
    ...                    's4': 1.000078616581918, 's5': 1.0008131880118405}
    >>> beta = 0.001
    >>> stringify(divide(tabu_search, instance=instance, initial_budgets=initial_budgets,beta=beta, delta={0.34}))
    "{s1:['c1', 'c2', 'c6', 'c7', 'c8'], s2:['c1', 'c3', 'c4', 'c5', 'c8'], s3:['c1', 'c2', 'c4', 'c5', 'c8'], s4:['c2', 'c3', 'c6', 'c7'], s5:['c3', 'c4', 'c7']}"
    """
    initial_budgets = kwargs.get('initial_budgets')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    logger.info("Tabu search: initial budgets = %s, beta = %s, delta = %s", initial_budgets, beta, delta)

    prices = {course: random.uniform(1, 1 + beta) for course in alloc.instance.items}
    logger.info("1) Let 𝒑 ← uniform(1, 1 + 𝛽)^𝑚, H ← ∅: p = %s", prices)
    history = []
    combinations_courses_sorted = get_combinations_courses_sorted(alloc.instance)

    logger.info("2) If ∥𝒛(𝒖,𝒄, 𝒑, 𝒃0)∥2 = 0, terminate with 𝒑∗ = 𝒑.")

    best_allocation, best_prices, best_norma = None, None, np.inf
    while True:
        max_utilities_allocations = student_best_bundles(prices.copy(), alloc.instance, initial_budgets,
                                                         combinations_courses_sorted)
        allocation, excess_demand_vector, norma = min_excess_demand_for_allocation(alloc.instance, prices,
                                                                                   max_utilities_allocations)
        logger.info("\nprices=%s, excess demand=%s, best bundle=%s, norma=%s", prices, excess_demand_vector, allocation,
                    norma)
        # best_allocation, best_prices, best_norma = allocation, prices, norma
        if np.allclose(norma, 0):
            logger.info("2) ∥𝒛(𝒖,𝒄, 𝒑, 𝒃0)∥2 = 0: terminate with 𝒑∗ = 𝒑.")
            break

        logger.info("3) Include all equivalent prices of 𝒑 into the history: H ← H + {𝒑′ : 𝒑′ ∼𝑝 𝒑}")
        equivalent_prices = find_all_equivalent_prices(alloc.instance, initial_budgets, allocation)
        history.append(equivalent_prices)
        neighbors = find_all_neighbors(alloc.instance, history, prices, delta, excess_demand_vector,
                                       initial_budgets,
                                       allocation, combinations_courses_sorted)
        constraints = create_constraints_from_neighbors(neighbors)
        history.extend(constraints)
        logger.info("Found %d neighbors", len(neighbors))
        if len(neighbors) == 0:
            logger.info("--- No new neighbors to price-vector - no optimal solution")
            break

        logger.info("   update 𝒑 ← arg min𝒑′∈N (𝒑)−H ∥𝒛(𝒖,𝒄, 𝒑', 𝒃0)∥2")
        allocation, excess_demand_vector, norma, prices = find_min_error_prices(alloc.instance, neighbors,
                                                                                initial_budgets,
                                                                                combinations_courses_sorted)
        if norma < best_norma:
            logger.info("   Found a better norma")
            best_allocation, best_prices, best_norma = allocation, prices, norma

    logger.info(f"\nfinal prices p* = {best_prices}")
    logger.info(f"allocation is: {best_allocation}")
    for student, bundle in best_allocation.items():
        for item in bundle:
            if item in alloc.remaining_item_capacities:
                alloc.give(student, item, logger)

    return best_allocation


# ---------------------helper functions:---------------------
def create_constraints_from_neighbors(neighbors):
    """
    Create lambda constraints from neighbors.

    :param neighbors: (list of dicts): List of dictionaries where each dictionary represents a neighbor with courses
                    and their values.

    :return: lambda_groups (list of list of lambda functions): List of groups of lambda functions, each group
            representing constraints based on a neighbor.



    >>> neighbors = [{'x': 1, 'y': 4, 'z': 0}]
    >>> ans = create_constraints_from_neighbors(neighbors)
    >>> p = {'x': 1, 'y': 4, 'z': 0}
    >>> all([f(p) for f in ans[0]])
    True

    >>> neighbors = [{'x': 1, 'y': 4, 'z': 0}, {'x': 2, 'y': 3, 'z': 5}]
    >>> ans = create_constraints_from_neighbors(neighbors)
    >>> p = {'x': 1, 'y': 3, 'z': 0}
    >>> any(all([f(p) for f in s]) for s in ans)
    False

    >>> p = {'x': 2, 'y': 3, 'z': 5}
    >>> any(all([f(p) for f in s]) for s in ans)
    True

    """
    lambda_groups = []

    for neighbor in neighbors:
        lambda_group = []

        for course, value in neighbor.items():
            lambda_func = lambda p, key=course, val=value: p[key] == val
            lambda_group.append(lambda_func)

        lambda_groups.append(lambda_group)

    return lambda_groups


def min_excess_demand_for_allocation(instance: Instance, prices: dict, max_utilities_allocations: list[dict]):
    """
    Goes through all allocations with the highest utilities of the students, and returns the allocation with the
    lowest norm

    :param instance: fair-course-allocation instance
    :param prices: dictionary with courses prices
    :param max_utilities_allocations: A list of dictionaries that tells for each student all the package options he
                                      can take with the maximum utility

    :return: Allocation, and its excess demand vector, which gives the lowest norm

    Example run 2 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":3, "w":2}, "Bob":{"x":5, "y":2, "z":4, "w":3}},
    ...     agent_capacities=3,
    ...     item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> prices = {"x": 1, "y": 2, "z": 3, "w":4}
    >>> max_utilities_allocations = [{'Alice': ('x', 'y', 'z'), 'Bob': ('x', 'y', 'z')}]
    >>> min_excess_demand_for_allocation(instance, prices, max_utilities_allocations)
    ({'Alice': ('x', 'y', 'z'), 'Bob': ('x', 'y', 'z')}, {'x': 1, 'y': 0, 'z': 1, 'w': -2}, 2.449489742783178)


    Example run 3 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":3, "z":3, "w":3}, "Bob":{"x":3, "y":3, "z":3, "w":3}, "Eve":{"x":4, "y":4, "z":4, "w":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> prices = {'x': 2.6124658024539347, 'y': 0, 'z': 1.1604071365185367, 'w': 5.930224022321449}
    >>> max_utilities_allocations = [{'Alice': ('x', 'y'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'y'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'y'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}]
    >>> min_excess_demand_for_allocation(instance, prices, max_utilities_allocations)
    ({'Alice': ('x', 'y'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'x': 1, 'y': 0, 'z': 0, 'w': -1}, 1.4142135623730951)
    """
    min_norma = float("inf")
    min_excess_demand = {}
    best_alloc = {}
    for alloc in max_utilities_allocations:
        excess_demand_vector = clipped_excess_demand(instance, prices, alloc)
        values = np.array(list(excess_demand_vector.values()))
        curr_norma = np.linalg.norm(values)

        if curr_norma < min_norma:
            min_norma = curr_norma
            min_excess_demand = excess_demand_vector
            best_alloc = alloc

    return best_alloc, min_excess_demand, min_norma


def excess_demand(instance: Instance, allocation: dict):
    """
    Calculate for every course its excess demand
    𝑧𝑗 (𝒖,𝒄, 𝒑, 𝒃) = ∑︁ 𝑎𝑖𝑗 (𝒖, 𝒑, 𝒃) − 𝑐𝑗
                  𝑖=1 to n

    :param instance: fair-course-allocation instance
    :param allocation: a dictionary that maps each student to his bundle

    :return: a dictionary that maps each course to its excess demand

    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> allocation = {"ami":('x','y'), "tami":('x','y')}
    >>> excess_demand(instance, allocation)
    {'x': 0, 'y': 1, 'z': -3}
    """
    z = {}  # Initialize z as a dictionary
    for course in instance.items:
        sum_allocation = 0
        for student, bundle in allocation.items():
            if course in bundle:
                sum_allocation += 1
        z[course] = sum_allocation - instance.item_capacity(course)
    return z


def clipped_excess_demand(instance: Instance, prices: dict, allocation: dict):
    """
       Calculate for every course its clipped excess demand
       𝑧˜𝑗 (𝒖,𝒄, 𝒑, 𝒃) =  𝑧𝑗 (𝒖,𝒄, 𝒑, 𝒃) if 𝑝𝑗 > 0,
                         max{0, 𝑧𝑗 (𝒖,𝒄, 𝒑, 𝒃)} if 𝑝𝑗 = 0


       :param instance: fair-course-allocation instance
       :param allocation: a dictionary that maps each student to his bundle

       :return: a dictionary that maps each course to its clipped excess demand

       >>> instance = Instance(
       ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}},
       ... agent_capacities=2,
       ... item_capacities={"x":3, "y":1, "z":3})
       >>> allocation = {"ami":('x','y'), "tami":('x','y')}
       >>> prices = {"x":0, "y":2, "z":0}
       >>> clipped_excess_demand(instance ,prices, allocation)
       {'x': 0, 'y': 1, 'z': 0}
    """
    z = excess_demand(instance, allocation)
    clipped_z = {course: max(0, z[course]) if prices[course] == 0 else z[course] for course in z}
    # logger.debug(f"excess demand: {clipped_z}")
    return clipped_z


def student_best_bundles(prices: dict, instance: Instance, initial_budgets: dict, combinations_courses_sorted: dict):
    """
    Return a list of dictionaries that tells for each student all the bundle options he can take with the maximum benefit.

    :param prices: dictionary with courses prices
    :param instance: fair-course-allocation instance
    :param initial_budgets: students' initial budgets
    :param combinations_courses_sorted: sorted list of course combinations for each student

    :return: a list of dictionaries that maps each student to its best bundle.

     Example run 1 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":4, "z":2}, "Bob":{"x":4, "y":3, "z":2}, "Eve":{"x":2, "y":4, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets = {"Alice": 5, "Bob": 4, "Eve": 3}
    >>> prices = {"x": 1, "y": 2, "z": 1}
    >>> combinations_courses_sorted = {'Alice': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'Bob': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Eve': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}
    >>> student_best_bundles(prices, instance, initial_budgets, combinations_courses_sorted)
    [{'Alice': ('x', 'y'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}]

     Example run 2 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":3, "w":2}, "Bob":{"x":5, "y":2, "z":4, "w":3}},
    ...     agent_capacities=3,
    ...     item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets = {"Alice": 8, "Bob": 6}
    >>> prices = {"x": 1, "y": 2, "z": 3, "w":4}
    >>> combinations_courses_sorted = {'Alice': [('x', 'y', 'z'), ('x', 'y', 'w'), ('x', 'z', 'w'), ('y', 'z', 'w'), ('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('x',), ('z', 'w'), ('y',), ('z',), ('w',)], 'Bob': [('x', 'z', 'w'), ('x', 'y', 'z'), ('x', 'y', 'w'), ('y', 'z', 'w'), ('x', 'z'), ('x', 'w'), ('x', 'y'), ('z', 'w'), ('y', 'z'), ('x',), ('y', 'w'), ('z',), ('w',), ('y',)]}
    >>> student_best_bundles(prices, instance, initial_budgets, combinations_courses_sorted)
    [{'Alice': ('x', 'y', 'z'), 'Bob': ('x', 'y', 'z')}]


    Example run 3 iteration 1
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":3, "z":3, "w":3}, "Bob":{"x":3, "y":3, "z":3, "w":3}, "Eve":{"x":4, "y":4, "z":4, "w":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> initial_budgets = {"Alice": 4, "Bob": 5, "Eve": 2}
    >>> prices = {'x': 2.6124658024539347, 'y': 0, 'z': 1.1604071365185367, 'w': 5.930224022321449}
    >>> combinations_courses_sorted = {'Alice': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'Bob': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'Eve': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)]}
    >>> student_best_bundles(prices, instance, initial_budgets, combinations_courses_sorted)
    [{'Alice': ('x', 'y'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'y'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'y'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('x', 'z'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('x', 'y'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('x', 'z'), 'Eve': ('y', 'z')}, {'Alice': ('y', 'z'), 'Bob': ('y', 'z'), 'Eve': ('y', 'z')}]
    """

    @lru_cache(maxsize=None)
    def bundle_valuation(student, combination):
        return instance.agent_bundle_value(student, combination)

    all_combinations = {student: [] for student in instance.agents}
    for student in instance.agents:
        max_valuation = -1
        for combination in combinations_courses_sorted[student]:
            price_combination = sum(prices[course] for course in combination)
            if price_combination <= initial_budgets[student]:
                current_valuation = bundle_valuation(student, combination)
                if current_valuation >= max_valuation:
                    if current_valuation > max_valuation:
                        all_combinations[student] = []
                    max_valuation = current_valuation
                    all_combinations[student].append(combination)

        if not all_combinations[student]:
            all_combinations[student].append(())

    all_combinations_list = list(product(*all_combinations.values()))

    valid_allocations = []
    for allocation in all_combinations_list:
        valid_allocation = {}
        for student, bundle in zip(instance.agents, allocation):
            if sum(prices[item] for item in bundle) <= initial_budgets[student]:
                valid_allocation[student] = bundle
        if len(valid_allocation) == len(instance.agents):
            valid_allocations.append(valid_allocation)

    return valid_allocations


def find_all_equivalent_prices(instance: Instance, initial_budgets: dict, allocation: dict):
    """
    find all equivalent prices list of all equivalent prices of 𝒑 (the history).

    :param instance: fair-course-allocation
    :param initial_budgets: students' initial budgets
    :param allocation: a dictionary that maps each student to his bundle

    Example run 1
    >>> instance = Instance(valuations={"A":{"x":3, "y":4, "z":2},
    ...    "B":{"x":4, "y":3, "z":2}, "C":{"x":2, "y":4, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets = {"A": 5, "B":4, "C":3}
    >>> allocation = {"A": {'x', 'y'}, "B":{'x', 'y'}, "C":{'y', 'z'}}
    >>> equivalent_prices = find_all_equivalent_prices(instance, initial_budgets, allocation)
    >>> p = {"x":1, "y":2, "z":1}
    >>> all([f(p) for f in equivalent_prices])
    True
    >>> p = {"x":5, "y":5, "z":5}
    >>> all([f(p) for f in equivalent_prices])
    False

    # [(['x', 'y'], '<=', 5), (['x', 'y'], '<=', 4), (['y', 'z'], '<=', 3)]

    Example run 1
    >>> instance = Instance(valuations={"A":{"x":3, "y":4, "z":2},
    ...    "B":{"x":4, "y":3, "z":2}, "C":{"x":2, "y":4, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":2, "y":1, "z":3})
    >>> initial_budgets = {"A": 5, "B":4, "C":3}
    >>> allocation = {"A": {'x', 'y'}, "B":{'x', 'z'}, "C":{'x', 'z'}}
    >>> equivalent_prices = find_all_equivalent_prices(instance, initial_budgets, allocation)
    >>> p = {"x":1, "y":5, "z":1}
    >>> equivalent_prices[0](p)
    False
    >>> all([equivalent_prices[0](p)])
    False
    >>> all([f(p) for f in equivalent_prices])
    False


    >>> p = {"x":0, "y":0, "z":0}
    >>> all([f(p) for f in equivalent_prices])
    False

    >>> p = {"x":1, "y":5, "z":1}
    >>> all([f(p) for f in equivalent_prices])
    False

    Example run 2 iteration 1
    >>> instance = Instance(valuations={"A":{"x":5, "y":4, "z":3, "w":2},"B":{"x":5, "y":2, "z":4, "w":3}},
    ...     agent_capacities=3,
    ...     item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> initial_budgets = {"A": 8, "B":6}
    >>> allocation = {"A": {'x', 'y','z'}, "B":{'x','y' ,'z'}}
    >>> equivalent_prices = find_all_equivalent_prices(instance, initial_budgets, allocation)
    >>> p = {"x":2, "y":2, "z":4, "w":2}
    >>> all([f(p) for f in equivalent_prices])
    False

    >>> p =  {"x":2, "y":4, "z":3,"w":0}
    >>> all([f(p) for f in equivalent_prices])
    False

    # p(x) + p(y) +p (z) <=8, p(x) + p(y) +p (z) <=6
    # p(x) + p(z) + p (w) > 6


    Example run 3
    >>> instance = Instance(
    ... valuations={"ami": {"x": 3, "y": 3, "z": 3, "w":3}, "tami": {"x": 3, "y": 3, "z": 3, "w":3},
    ... "tzumi": {"x": 4, "y": 4, "z": 4, "w":4}},
    ... agent_capacities=2,
    ... item_capacities={"x": 1, "y": 2, "z": 2, "w":1})
    >>> initial_budgets = {"ami": 4, "tami": 5, "tzumi": 2}
    >>> allocation = {'ami': ('x', 'z'), 'tami': ('x', 'z'), 'tzumi': 'z'}
    >>> equivalent_prices = find_all_equivalent_prices(instance, initial_budgets, allocation)
    >>> p = {'x': 2.6124658024539347, 'y': 0, 'z': 1.1604071365185367, 'w': 5.930224022321449}
    >>> all([f(p) for f in equivalent_prices])
    False

    """
    equivalent_prices = []
    # The constraints that the bundles they get in allocation meet their budgets
    for student in instance.agents:
        func = lambda p, agent=student, keys=allocation[student]: (
                sum(p[key] for key in keys) <= initial_budgets[agent])
        equivalent_prices.append(func)

    # Constraints that will ensure that this is the allocation that will be accepted
    for student in instance.agents:
        # Creating a list of combinations of courses up to the size of the student's capacity
        combinations_courses_list = []
        capacity = instance.agent_capacity(student)
        for r in range(1, capacity + 1):
            combinations_courses_list.extend(combinations(instance.items, r))

        original_utility = instance.agent_bundle_value(student, allocation[student])
        current_alloc = False

        for combination in combinations_courses_list:
            current_utility = instance.agent_bundle_value(student, combination)
            sorted_combination = sorted(combination)  # Sort the combination
            sorted_alloc_student = sorted(allocation[student])

            if sorted_combination == sorted_alloc_student:
                current_alloc = True
                continue

            if current_alloc and len(sorted_combination) == len(sorted_alloc_student):
                continue

            if current_utility >= original_utility:
                func = lambda p, agent=student, keys=allocation[student]: (
                        sum(p[key] for key in keys) > initial_budgets[agent])
                equivalent_prices.append(func)

    return list(equivalent_prices)


def find_gradient_neighbors(prices: dict, delta: set, excess_demand_vector: dict, history: list[list]):
    """
    Add the gradient neighbors to the neighbors list
    N_gradient(𝒑, Δ) = {𝒑 + 𝛿 · 𝒛(𝒖,𝒄, 𝒑, 𝒃) : 𝛿 ∈ Δ}

    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param delta: The step size
    :param excess_demand_vector: excess demand of the courses
    :return: None

    Example run 1 iteration 1
    >>> prices = {"x": 1, "y": 2, "z": 1}
    >>> delta = {1}
    >>> excess_demand_vector = {"x":0,"y":2,"z":-2}
    >>> history = []
    >>> find_gradient_neighbors(prices,delta,excess_demand_vector, history)
    [{'x': 1, 'y': 4, 'z': 0}]

    Example run 1 iteration 2
    >>> prices = {"x": 1, "y": 4, "z": 0}
    >>> delta = {1}
    >>> excess_demand_vector = {"x":1,"y":0,"z":0}
    >>> history = []
    >>> find_gradient_neighbors(prices,delta,excess_demand_vector, history)
    [{'x': 2, 'y': 4, 'z': 0}]

     Example run 1 iteration 2
    >>> prices = {'x': 1, 'y': 4, 'z': 0}
    >>> delta = {1}
    >>> excess_demand_vector = {'x':1,'y':0,'z':0}
    >>> history = [[lambda p: p['x']+p['y']<=6, lambda p: p['x']+p['y']<=8, lambda p: p['y']+p['z']<=5]]
    >>> find_gradient_neighbors(prices,delta,excess_demand_vector, history)
    []

    >>> prices = {"x": 1, "y": 4, "z": 0}
    >>> delta = {0.5, 1}
    >>> excess_demand_vector = {"x":1,"y":0,"z":2}
    >>> history = []
    >>> find_gradient_neighbors(prices,delta,excess_demand_vector, history)
    [{'x': 1.5, 'y': 4.0, 'z': 1.0}, {'x': 2, 'y': 4, 'z': 2}]
    """
    new_neighbors = []
    updated_prices = {}
    for d in delta:
        for course, price in prices.items():
            updated_prices[course] = max(0, price + d * excess_demand_vector.get(course, 0))

        if not any(all(f(updated_prices) for f in sublist) for sublist in history):
            new_neighbors.append(updated_prices.copy())  # Using copy() to append a new dictionary to the list

    return new_neighbors


def differ_in_one_value(original_allocation: dict, new_allocation: dict, course: str) -> bool:
    """
    Check if two dictionaries differ with each other in exactly one value.

    :param original_allocation: First dictionary
    :param new_allocation: Second dictionary
    :return: True if the dictionaries differ in exactly one value, False otherwise

    >>> allocation1 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> allocation2 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','t')}
    >>> course ="z"
    >>> differ_in_one_value(allocation1, allocation2, course)
    True

    >>> allocation1 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> allocation2 = {"ami":('x','y'),"tami":('h','z'),"tzumi":('x','t')}
    >>> course = "x"
    >>> differ_in_one_value(allocation1, allocation2, course)
    False

    >>> allocation1 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> allocation2 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> course = "z"
    >>> differ_in_one_value(allocation1, allocation2, course)
    False

    >>> allocation1 = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> allocation2 = {"ami":('y','z'),"tami":('x','z'),"tzumi":('x','z')}
    >>> course = "x"
    >>> differ_in_one_value(allocation1, allocation2 , course)
    True
    """
    # Count the number of differing values
    diff_count = 0
    diff_course = None
    for key in original_allocation:
        if key in new_allocation and original_allocation[key] != new_allocation[key]:
            diff_course = key
            diff_count += 1
            # If more than one value differs, return False immediately
            if diff_count > 1:
                return False
    # Return True if exactly one value differs
    return diff_count == 1 and course in original_allocation[diff_course] and course not in new_allocation[diff_course]


def find_individual_price_adjustment_neighbors(instance: Instance, history: list[list], prices: dict,
                                               excess_demand_vector: dict, initial_budgets: dict, allocation: dict,
                                               combinations_courses_sorted: dict = {}):
    """
    Add the individual price adjustment neighbors N(p) to the neighbors list

    :param instance: fair-course-allocation
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param excess_demand_vector: excess demand of the courses
    :param initial_budgets: students' initial budgets
    :param allocation: a dictionary that maps each student to his bundle
    :return: None

    Example run 1 iteration 1
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> history = [[lambda p: p['x']+p['y']<=5, lambda p: p['x']+p['y']<=4, lambda p: p['y']+p['z']<=3]]
    >>> prices = {"x": 1, "y": 2, "z": 1}
    >>> excess_demand_vector = {"x":0,"y":2,"z":-2}
    >>> initial_budgets = {"ami":5,"tami":4,"tzumi":3}
    >>> allocation = {"ami":('x','y'),"tami":('x','y'),"tzumi":('y','z')}
    >>> combinations_courses_sorted = {'ami': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'tami': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'tzumi': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}
    >>> find_individual_price_adjustment_neighbors(instance, history, prices, excess_demand_vector, initial_budgets, allocation, combinations_courses_sorted)
    [{'x': 1, 'y': 2.7071067811865475, 'z': 1}]


     Example run 1 iteration 2
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> history = [[lambda p: p['x']+p['y']<=5, lambda p: p['x']+p['y']<=4, lambda p: p['y']+p['z']<=3,
    ...           lambda p: p['x']+p['z']<=4, lambda p: p['x']+p['z']<=3, lambda p: p['y']+p['z']>=3, lambda p: p['x']+p['y']>=4]]
    >>> prices = {"x": 1, "y": 4, "z": 0}
    >>> excess_demand_vector = {"x":1,"y":0,"z":0}
    >>> initial_budgets = {"ami":5,"tami":4,"tzumi":3}
    >>> allocation = {"ami":('x','y'),"tami":('x','z'),"tzumi":('x','z')}
    >>> combinations_courses_sorted = {'ami': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'tami': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'tzumi': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}
    >>> find_individual_price_adjustment_neighbors(instance, history, prices, excess_demand_vector, initial_budgets, allocation, combinations_courses_sorted)
    [{'x': 1.7071067811865475, 'y': 4, 'z': 0}, {'x': 2.414213562373095, 'y': 4, 'z': 0}]


    Example run 3 iteration 1
    >>> instance = Instance(
    ... valuations={"ami": {"x": 3, "y": 3, "z": 3, "w":3}, "tami": {"x": 3, "y": 3, "z": 3, "w":3},
    ... "tzumi": {"x": 4, "y": 4, "z": 4, "w":4}},
    ... agent_capacities=2,
    ... item_capacities={"x": 1, "y": 2, "z": 2, "w":1})
    >>> history = [[lambda p: p['x']+p['z']<=4, lambda p: p['x']+p['z']<=5, lambda p: p['z']<=2,
    ...             lambda p: p['x']+p['y']>4, lambda p: p['x']+p['y']>5, lambda p: p['x']+p['y']>2,
    ...             lambda p: p['x']+p['z']>2, lambda p: p['x']+p['w']>2, lambda p: p['y']+p['z']>2,
    ...             lambda p: p['y']+p['w']>2, lambda p: p['z']+p['w']>2, lambda p: p['x']>2,
    ...             lambda p: p['y']>2]]
    >>> prices = {'x': 2.6124658024539347, 'y': 4.138416343413373, 'z': 1.1604071365185367, 'w': 5.930224022321449}
    >>> excess_demand_vector = {'x': 1, 'y': -2, 'z': 1, 'w': -1}
    >>> initial_budgets = {"ami": 4, "tami": 5, "tzumi": 2}
    >>> allocation = {'ami': ('x', 'z'), 'tami': ('x', 'z'), 'tzumi': 'z'}
    >>> combinations_courses_sorted = {'ami': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'tami': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'tzumi': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)]}
    >>> find_individual_price_adjustment_neighbors(instance, history, prices, excess_demand_vector, initial_budgets, allocation, combinations_courses_sorted)
    [{'x': 2.6124658024539347, 'y': 0, 'z': 1.1604071365185367, 'w': 5.930224022321449}, {'x': 2.6124658024539347, 'y': 4.138416343413373, 'z': 1.1604071365185367, 'w': 0}]
    """
    new_neighbors = []
    for course, excess_demand in excess_demand_vector.items():
        if len(new_neighbors) >= 35:
            break
        if excess_demand == 0:
            continue
        updated_prices = prices.copy()
        if excess_demand > 0:
            for _ in range(10):
                updated_prices[course] += np.sqrt(0.5)
                if any(all(f(updated_prices) for f in sublist) for sublist in history):
                    continue
                # get the new demand of the course
                new_allocations = student_best_bundles(updated_prices.copy(), instance, initial_budgets,
                                                       combinations_courses_sorted)

                for new_allocation in new_allocations:
                    if differ_in_one_value(allocation, new_allocation, course):
                        new_neighbors.append(updated_prices.copy())

        elif excess_demand < 0:
            updated_prices[course] = 0
            if not any(all(f(updated_prices) for f in sublist) for sublist in history):
                new_neighbors.append(updated_prices)

    return new_neighbors


def find_all_neighbors(instance: Instance, history: list, prices: dict, delta: set,
                       excess_demand_vector: dict, initial_budgets: dict, allocation: dict,
                       combinations_courses_sorted: dict = {}):
    """
    Update neighbors N (𝒑) - list of Gradient neighbors and Individual price adjustment neighbors.

    :param instance: fair-course-allocation
    :param history: all equivalent prices of 𝒑
    :param prices: dictionary with courses prices
    :param delta: The step size
    """

    gradient_neighbors = find_gradient_neighbors(prices, delta, excess_demand_vector, history)
    individual_price_adjustment_neighbors = find_individual_price_adjustment_neighbors(instance, history,
                                                                                       prices,
                                                                                       excess_demand_vector,
                                                                                       initial_budgets, allocation,
                                                                                       combinations_courses_sorted)
    logger.debug(f"neighbors: \ngradient_neighbors = {gradient_neighbors}")
    logger.debug(f"individual_price = {individual_price_adjustment_neighbors}")

    return gradient_neighbors + individual_price_adjustment_neighbors


def find_min_error_prices(instance: Instance, neighbors: list, initial_budgets: dict,
                          combinations_courses_sorted: dict = {}):
    """
    Return the update prices that minimize the market clearing error.

    :param instance: fair-course-allocation
    :param prices: dictionary with course prices
    :param neighbors: list of Gradient neighbors and Individual price adjustment neighbors.
    :param initial_budgets: students' initial budgets

    :return: allocation, excess_demand, norm and update price with the minimum norm

    Example run 1 iteration 1
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> neighbors = [{"x":1, "y":4, "z":0}, {"x":1, "y":3, "z":1}]
    >>> initial_budgets={"ami":5, "tami":4, "tzumi":3}
    >>> combinations_courses_sorted = {'ami': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'tami': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'tzumi': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}
    >>> find_min_error_prices(instance, neighbors, initial_budgets, combinations_courses_sorted)
    ({'ami': ('x', 'y'), 'tami': ('x', 'z'), 'tzumi': ('x', 'z')}, {'x': 1, 'y': 0, 'z': 0}, 1.0, {'x': 1, 'y': 4, 'z': 0})

     Example run 1 iteration 2
    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> neighbors = [{"x":2, "y":4, "z":0}, {"x":3, "y":4, "z":0}]
    >>> initial_budgets={"ami":5, "tami":4, "tzumi":3}
    >>> combinations_courses_sorted = {'ami': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'tami': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'tzumi': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}
    >>> find_min_error_prices(instance, neighbors, initial_budgets, combinations_courses_sorted)
    ({'ami': ('y', 'z'), 'tami': ('x', 'z'), 'tzumi': ('x', 'z')}, {'x': 0, 'y': 0, 'z': 0}, 0.0, {'x': 2, 'y': 4, 'z': 0})
    """
    errors = []  # tuple of (allocation, excess_demand, norm, price)
    min_error_prices = []
    logger.debug("\nChecking the neighbors:")
    for neighbor in neighbors:
        logger.debug(f"neighbor: {neighbor}")
        allocations = student_best_bundles(neighbor.copy(), instance, initial_budgets, combinations_courses_sorted)
        allocation, excess_demand_vector, norma = min_excess_demand_for_allocation(instance, neighbor, allocations)
        logger.debug(f"excess demand: {excess_demand_vector}")
        logger.debug(f"norma = {norma}")
        errors.append((allocation, excess_demand_vector, norma, neighbor))
    logger.debug("")

    min_error_tuple = min(errors, key=lambda x: x[2])
    return min_error_tuple


def run_tabu_search(alloc: AllocationBuilder, **kwargs):
    beta = 0.001
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents, beta)
    return tabu_search(alloc, initial_budgets=initial_budgets, beta=beta, delta={0.34}, **kwargs)


def random_initial_budgets(num_of_agents: int, beta: float = 100) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [1, 1 + beta]
    initial_budgets = np.random.uniform(1, 1 + beta, num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}


if __name__ == "__main__":
    import doctest, sys

    print("\n", doctest.testmod(), "\n")
    # sys.exit(0)

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    import coloredlogs

    level_styles = {
        'debug': {'color': 'green'},
        'info': {'color': 'cyan'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'bold': True, 'background': 'white'}
    }
    coloredlogs.install(level='DEBUG', logger=logger, fmt='%(message)s', level_styles=level_styles)

    from fairpyx.adaptors import divide

    random_delta = {random.uniform(0.1, 1)}
    random_beta = random.uniform(1, 100)


    def random_initial_budgets(num):
        return {f"s{key}": random.uniform(1, 1 + random_beta) for key in range(1, num + 1)}

    random.seed(0)
    instance = Instance(
        valuations={'s1': {'c1': 275, 'c2': 79, 'c3': 59, 'c4': 63, 'c5': 54, 'c6': 226, 'c7': 133, 'c8': 110},
                    's2': {'c1': 105, 'c2': 17, 'c3': 222, 'c4': 202, 'c5': 227, 'c6': 89, 'c7': 30, 'c8': 107},
                    's3': {'c1': 265, 'c2': 120, 'c3': 37, 'c4': 230, 'c5': 160, 'c6': 44, 'c7': 30, 'c8': 113},
                    's4': {'c1': 194, 'c2': 132, 'c3': 224, 'c4': 77, 'c5': 29, 'c6': 230, 'c7': 62, 'c8': 52},
                    's5': {'c1': 174, 'c2': 89, 'c3': 229, 'c4': 249, 'c5': 24, 'c6': 83, 'c7': 99, 'c8': 52}},
        agent_capacities=5,
        item_capacities={'c1': 3.0, 'c2': 3.0, 'c3': 3.0, 'c4': 3.0, 'c5': 3.0, 'c6': 3.0, 'c7': 3.0, 'c8': 3.0})
    initial_budgets = {'s1': 1.0005695511898616, 's2': 1.0009070710569965, 's3': 1.000699704772071,
                       's4': 1.000078616581918, 's5': 1.0008131880118405}
    beta = 0.001
    divide(tabu_search, instance=instance, initial_budgets=initial_budgets, beta=beta, delta={0.34})