"""
Implement an " A-CEEI with (contested) EF-TB property" course allocation,

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
from enum import Enum
import logging
from fairpyx import Instance
from itertools import combinations


class EFTBStatus(Enum):
    NO_EF_TB = 0
    EF_TB = 1
    CONTESTED_EF_TB = 2


logger = logging.getLogger(__name__)


def excess_demand(instance, initial_budgets, prices, allocation):
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


def find_different_budgets(instance, initial_budgets, epsilon, delta, prices):
    max_k = (2 * epsilon / delta) + 1

    # Creating all possible combinations of prices
    combinations_set = set()
    # A matrix for keeping the budgets that give different bundles
    matrix_k = []

    students_names = list(instance._agent_capacities.keys())
    for student in range(instance.num_of_agents):
        # For each student, the course price combinations (in combinations_list)
        # are calculated according to the number of courses he needs to take
        capacity = instance._agent_capacities[students_names[student]]
        for r in range(1, capacity + 1):
            for combo in combinations(prices, r):
                combinations_set.add(sum(combo))
        combinations_list = list(combinations_set)

        # Setting the start and end index according to the definition
        index_start = initial_budgets[student] - epsilon
        index_end = initial_budgets[student] + epsilon
        # range_budget = (index_start, index_end + 1)

        # Keeping the various budgets for the current student
        row_student = [index_start]
        for combination in combinations_list:
            if len(row_student) + 1 > max_k:
                break
            if index_start < combination < index_end:
                row_student.append(combination)
                index_start = combination

        matrix_k.append(row_student)
    return matrix_k


# TODO: change the name?
def student_budget_per_bundle(different_budgets, prices, instance):
    # A matrix that says for each budget what is the bundle with the maximum utility that a student can take
    matrix_a = []

    # TODO: add this value when it stand in the requirements
    large_num = sum(prices[i] for i in range(len(prices)))

    students_names = list(instance._agent_capacities.keys())
    number_course = (i for i in range(len(prices)))

    for student in range(len(different_budgets)):
        # The combinations of the courses according to the student's capacity
        combination_list = []
        max_combination = None
        utility_max_combination = float('-inf')

        for budget in range(len(different_budgets[student])):
            capacity = instance._agent_capacities[students_names[student]]
            for r in range(1, capacity + 1):
                combination_list.extend(combinations(number_course, r))

            for combination in combination_list:
                sum_of_prices = sum(prices[i] for i in combination)
                if sum_of_prices <= budget:
                    utility_combination = sum(instance.valuations[students_names[student]][i] for i in combination)
                    if utility_combination > utility_max_combination:
                        max_combination = combination
                        utility_max_combination = utility_combination


def find_budget_perturbation(initial_budgets, epsilon, delta, prices, instance, t):
    different_budgets = find_different_budgets(instance, initial_budgets, epsilon, delta, prices)
    a = student_budget_per_bundle(different_budgets, prices, instance)


def find_ACEEI_with_EFTB(instance: Instance, initial_budgets: any, delta: float, epsilon: float, t: Enum):
    """
    "Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
     by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
     ALGORITHM 1: find an A-CEEI with (contested) EF-TB property

    :param instance: a fair-course-allocation instance
    :param initial_budgets: Students' initial budgets
    :param delta: The step size
    :param epsilon: maximum budget perturbation
    :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB

    :return final courses prices, final budgets, final distribution

     >>> from fairpyx.adaptors import divide

    >>> from fairpyx.utils.test_utils import stringify

    >>> instance = Instance(
    ...     valuations={"avi":{"x":1, "y":2, "z":4}, "beni":{"x":2, "y":3, "z":1}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {2, 3}
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = EFTBStatus.NO_EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ...     delta=delta, epsilon=epsilon, t=t))
    "{avi:['x','z'], beni:['y', 'z']}"

    >>> instance = Instance(
    ... valuations={"avi":{"x":5, "y":2, "z":1}, "beni":{"x":4, "y":1, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {3, 4}
    >>> delta = 0.5
    >>> epsilon = 1
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    "{avi:['y','z'], beni:['x', 'z']}"

    >>> instance = Instance(
    ...     valuations={"avi":{"x":5, "y":5, "z":1}, "beni":{"x":4, "y":6, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2})
    >>> initial_budgets = {5, 4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
        "{avi:['x','y'], beni:['y', 'z']}"

    >>> instance = Instance(
    ...     valuations={"avi":{"x":10, "y":20}, "beni":{"x":10, "y":20}},
    ...     agent_capacities=1,
    ...     item_capacities = {"x":1, "y":1})
    >>> initial_budgets = {1.1, 1}
    >>> delta = 0.1
    >>> epsilon = 0.2
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ...     delta=delta, epsilon=epsilon, t=t))
        "{avi:['y'], beni:['x']}"

    >>> instance = Instance(
    ... valuations={"avi":{"x":2}, "beni":{"x":3}},
    ... agent_capacities=1,
    ... item_capacities = {"x":1})
    >>> initial_budgets = {1.1, 1}
    >>> delta = 0.1
    >>> epsilon = 0.2
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,
    ... initial_budgets=initial_budgets,delta=delta, epsilon=epsilon, t=t))
    "{avi:['x'], beni:[]}"

    >>> instance = Instance(valuations={"avi":{"x":5, "y":4, "z":1},
    ...    "beni":{"x":4, "y":6, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {5, 4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = EFTBStatus.CONTESTED_EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
        "{avi:['x', 'z'], beni:['y', 'z']}"
    """
    allocation = [[0 for _ in range(instance.num_of_agents)] for _ in range(instance.num_of_items)]
    # 1) init prices vector to be 0
    prices = [0] * instance.num_of_items
    norma = 1
    while norma:
        # 2) 𝜖-budget perturbation
        new_budgets, norma, allocation, excess_demand = find_budget_perturbation(initial_budgets, epsilon, delta,
                                                                                 prices, instance, t)
        # 3) If ∥𝒛˜(𝒖,𝒄, 𝒑, 𝒃) ∥2 = 0, terminate with 𝒑* = 𝒑, 𝒃* = 𝒃
        if norma == 0:
            return allocation  # TODO: we need to return p*, b*
        # 4) update 𝒑 ← 𝒑 + 𝛿𝒛˜(𝒖,𝒄, 𝒑, 𝒃), then go back to step 2.
        prices = prices + delta * excess_demand


if __name__ == "__main__":
    import doctest

    instance = Instance(agent_capacities={"Alice": 2, "Bob": 2}, item_capacities={"c1": 1, "c2": 1, "c3": 2},
                        valuations={"Alice": {"c1": 1, "c2": 2, "c3": 4}, "Bob": {"c1": 2, "c2": 3, "c3": 1}})

    p = [0, 2, 0]
    b_0 = [2, 3]
    # print(find_different_budgets(instance, initial_budgets=b_0, epsilon=0.5, delta=0.5, prices=p ))
    # print()
    # doctest.testmod()
