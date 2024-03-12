"""
Implement an " A-CEEI with (contested) EF-TB property" course allocation,

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
from enum import Enum
import logging

from fairpyx import Instance, AllocationBuilder
from itertools import combinations
import linear_program as lp


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


def student_best_bundle_per_budget(prices: dict, instance: Instance, epsilon: any, initial_budgets: dict):
    """
    Return a dict that says for each budget what is the bundle with the maximum utility that a student can take

    :param different_budgets: different budgets that will give to every student different bundles
    :param prices: courses prices
    :param instance: a fair-course-allocation instance

    :return matrix_a: that says for each budget what is the bundle with the maximum utility that a student can take

    Example run 3 iteration 7
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":5, "z":1}, "Bob":{"x":4, "y":6, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2})
    >>> initial_budgets = {"Alice": 5, "Bob": 4}
    >>> epsilon = 2
    >>> prices = {"x": 2.5, "y": 0, "z": 0}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets)
    {'Alice': {3: ('x', 'y')}, 'Bob': {2.5: ('x', 'y'), 2: ('y', 'z')}}

    # Alice: 3-7: (10, [x,y] , p=2.5) (6, [x,z] p=2.5) (6, [y,z] p=0)
    # BOB: 2-6: (10, [x,y] p=2,5), (10, [y,z] p=0) (8, [x,z] p=2.5)

    Example run 6 iteration 5
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {"Alice": 5, "Bob": 4}
    >>> epsilon = 2
    >>> prices = {"x": 1.5, "y": 2, "z": 0}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets)
    {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}

    # Alice: 3-7 -> (9, [x,y], p=3.5) (6, [x,z], p=1.5) (5, [y,z], p=2) (5 , x , p=1.5) (4, y, p=2) (1, z, p=0)
    # Bob: 2-6 -> (10, [x,y]. p=3.5) , (9, [y,z], p=2) , (7, [x.z] , p=1.5) , (6, [y] p=1.5) , (4, [x]. p= 1.5), (3, [z]], p=0)


    >>> instance = Instance(
    ...     valuations={"Alice":{"x":1, "y":1, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {"Alice": 5}
    >>> epsilon = 0.1
    >>> prices = {"x": 2, "y": 2, "z": 5}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets)
    {'Alice': {4.9: ('x', 'y')}}

    """

    best_bundle_per_budget = {}

    for student_idx, student in enumerate(instance.agents):

        # Creating a list of combinations of courses up to the size of the student's capacity
        combinations_courses_list = []
        capacity = instance.agent_capacity(student)
        for r in range(1, capacity + 1):
            combinations_courses_list.extend(combinations(instance.items, r))

        #  We would like to meet the requirement of the number of courses a student needs, therefore if
        #  the current combination meets the requirement we will give it more weight
        large_num = instance.agent_maximum_value(student)

        # Define a lambda function that calculates the valuation of a combination
        valuation_function = lambda combination: instance.agent_bundle_value(student, combination) + (
            large_num if len(combination) == instance.agent_capacity(student) else 0)

        # Sort the combinations_set based on their valuations in descending order
        combinations_courses_sorted = sorted(combinations_courses_list, key=valuation_function, reverse=True)

        # Setting the min and max budget according to the definition
        min_budget = initial_budgets[student] - epsilon
        max_budget = initial_budgets[student] + epsilon

        # Sort the combinations of the courses in descending order according to utility. We went through the
        # budgets in descending order, for each budget we looked for the combination with the maximum value that
        # could be taken in that budget.
        min_price = float('inf')

        for combination in combinations_courses_sorted:
            price_combination = sum(prices[course] for course in combination)

            if price_combination <= max_budget:
                if price_combination <= min_budget:
                    best_bundle_per_budget.setdefault(student, {})[min_budget] = combination
                    break

                if price_combination < min_price:
                    min_price = price_combination
                    best_bundle_per_budget.setdefault(student, {})[price_combination] = combination

    return best_bundle_per_budget


def find_budget_perturbation(initial_budgets, epsilon, prices, instance, t):
    # return: new_budgets, norma, allocation, excess_demand
    map_student_to_best_bundle_per_budget = student_best_bundle_per_budget(prices, instance, epsilon, initial_budgets)
    new_budgets, clearing_error, excess_demand_per_course = lp.optimize_model(map_student_to_best_bundle_per_budget, instance, prices, t, initial_budgets)
    logger.info(f"new_budgets in find_budget_perturbation: {new_budgets}")
    return new_budgets, clearing_error, map_student_to_best_bundle_per_budget, excess_demand_per_course


def find_ACEEI_with_EFTB(alloc: AllocationBuilder, initial_budgets: dict, delta: float, epsilon: float, t: Enum):
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

    >>> logger.addHandler(logging.StreamHandler())
    >>> logger.setLevel(logging.INFO)

    >>> instance = Instance(
    ...     valuations={"avi":{"x":1, "y":2, "z":4}, "beni":{"x":2, "y":3, "z":1}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {"avi":2, "beni":3}
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = EFTBStatus.NO_EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    "{avi:['x', 'z'], beni:['y', 'z']}"

    >>> instance = Instance(
    ... valuations={"avi":{"x":5, "y":2, "z":1}, "beni":{"x":4, "y":1, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {"avi":3, "beni":4}
    >>> delta = 0.5
    >>> epsilon = 1
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    "{avi:['y', 'z'], beni:['x', 'z']}"

    >>> instance = Instance(
    ...     valuations={"avi":{"x":5, "y":5, "z":1}, "beni":{"x":4, "y":6, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2})
    >>> initial_budgets = {"avi":5, "beni":4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    "{avi:['x', 'y'], beni:['y', 'z']}"

    >>> instance = Instance(
    ...     valuations={"avi":{"x":10, "y":20}, "beni":{"x":10, "y":20}},
    ...     agent_capacities=1,
    ...     item_capacities = {"x":1, "y":1})
    >>> initial_budgets = {"avi":1.1, "beni":1}
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
    >>> initial_budgets = {"avi":1.1, "beni":1}
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
    >>> initial_budgets = {"avi":5, "beni":4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = EFTBStatus.CONTESTED_EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    "{avi:['x', 'z'], beni:['y', 'z']}"
    """
    # allocation = [[0 for _ in range(instance.num_of_agents)] for _ in range(instance.num_of_items)]
    # 1) init prices vector to be 0
    # print(instance.items)
    logger.info("Starting ACEEI")
    prices = {key: 0 for key in alloc.remaining_items()}
    clearing_error = 1
    new_budgets = {}
    while clearing_error:
        # 2) 𝜖-budget perturbation
        new_budgets, clearing_error, allocation, excess_demand_per_course = find_budget_perturbation(initial_budgets, epsilon, prices,
                                                                                 alloc.instance, t)
        logger.info(f"new budget: {new_budgets}")
        logger.info(f"clearing_error: {clearing_error}")
        logger.info(f"allocation: {allocation}")
        # 3) If ∥𝒛˜(𝒖,𝒄, 𝒑, 𝒃) ∥2 = 0, terminate with 𝒑* = 𝒑, 𝒃* = 𝒃
        if clearing_error == 0:
            logger.info(f"---------allocation in the if: {allocation}")

            # for student, (price, bundle) in new_budgets.items():
            #     alloc.give(student, bundle)
            # return alloc.sorted()
            break
            # return allocation  # TODO: we need to return p* = prices, b* = new_budgets -  in the logger
        # 4) update 𝒑 ← 𝒑 + 𝛿𝒛˜(𝒖,𝒄, 𝒑, 𝒃), then go back to step 2.
        for key in prices:
            prices[key] += delta * excess_demand_per_course[key]

    logger.info("Clearing error 0!")
    for student, (price, bundle) in new_budgets.items():
        logger.info("Giving %s to %s in price %s", bundle, student, price)
        alloc.give_bundle(student, bundle)


#   TODO: enter the bundle


if __name__ == "__main__":
    import doctest

    from fairpyx.adaptors import divide

    from fairpyx.utils.test_utils import stringify

    # print(doctest.run_docstring_examples(find_ACEEI_with_EFTB,globals()))


    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    instance = Instance(
       valuations={"avi":{"x":1, "y":2, "z":4}, "beni":{"x":2, "y":3, "z":1}},
       agent_capacities=2,
       item_capacities={"x":1, "y":1, "z":2})
    initial_budgets = {"avi":2, "beni":3}
    delta = 0.5
    epsilon = 0.5
    t = EFTBStatus.NO_EF_TB
    print(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets, delta=delta, epsilon=epsilon, t=t))


    # instance = Instance(
    #     valuations={"avi":{"x":2}, "beni":{"x":3}},
    #     agent_capacities=1,
    #     item_capacities = {"x":1})
    # initial_budgets = {"avi":1.1, "beni":1}
    # delta = 0.1
    # epsilon = 0.2
    # t = EFTBStatus.EF_TB
    # print(divide(find_ACEEI_with_EFTB, instance=instance,initial_budgets=initial_budgets,delta=delta, epsilon=epsilon, t=t))