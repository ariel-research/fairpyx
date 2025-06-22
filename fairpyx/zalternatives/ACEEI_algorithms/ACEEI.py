"""
    "Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
     by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
    link to the article: https://arxiv.org/pdf/2305.11406
     ALGORITHM 1: find an A-CEEI with (contested) EF-TB property


Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
from enum import Enum
import logging
from itertools import combinations

import numpy as np

from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms.ACEEI_algorithms import linear_program as lp
from fairpyx.algorithms.ACEEI_algorithms.log_capture_handler import LogCaptureHandler
from fairpyx.algorithms.ACEEI_algorithms.calculate_combinations import get_combinations_courses_sorted


class EFTBStatus(Enum):
    NO_EF_TB = 0
    EF_TB = 1
    CONTESTED_EF_TB = 2


logger = logging.getLogger(__name__)


# ---------------------The main function---------------------

def find_ACEEI_with_EFTB(alloc: AllocationBuilder, **kwargs):
    """
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

    >>> instance = Instance(valuations={'s1': {'c1': 184, 'c2': 172, 'c3': 62, 'c4': 50, 'c5': 84, 'c6': 75, 'c7': 37, 'c8': 39, 'c9': 80, 'c10': 54, 'c11': 69, 'c12': 93}, 's2': {'c1': 81, 'c2': 2, 'c3': 223, 'c4': 61, 'c5': 89, 'c6': 229, 'c7': 81, 'c8': 94, 'c9': 18, 'c10': 103, 'c11': 2, 'c12': 17}, 's3': {'c1': 178, 'c2': 44, 'c3': 210, 'c4': 78, 'c5': 49, 'c6': 174, 'c7': 59, 'c8': 23, 'c9': 101, 'c10': 43, 'c11': 33, 'c12': 7}, 's4': {'c1': 165, 'c2': 134, 'c3': 8, 'c4': 36, 'c5': 146, 'c6': 210, 'c7': 15, 'c8': 52, 'c9': 88, 'c10': 56, 'c11': 55, 'c12': 35}, 's5': {'c1': 42, 'c2': 21, 'c3': 155, 'c4': 82, 'c5': 122, 'c6': 146, 'c7': 75, 'c8': 51, 'c9': 91, 'c10': 81, 'c11': 61, 'c12': 72}, 's6': {'c1': 82, 'c2': 141, 'c3': 42, 'c4': 159, 'c5': 172, 'c6': 13, 'c7': 45, 'c8': 32, 'c9': 104, 'c10': 84, 'c11': 56, 'c12': 69}, 's7': {'c1': 188, 'c2': 192, 'c3': 96, 'c4': 7, 'c5': 36, 'c6': 36, 'c7': 44, 'c8': 129, 'c9': 26, 'c10': 33, 'c11': 85, 'c12': 127}, 's8': {'c1': 38, 'c2': 89, 'c3': 131, 'c4': 48, 'c5': 186, 'c6': 89, 'c7': 72, 'c8': 86, 'c9': 110, 'c10': 95, 'c11': 7, 'c12': 48}, 's9': {'c1': 34, 'c2': 223, 'c3': 115, 'c4': 144, 'c5': 64, 'c6': 75, 'c7': 61, 'c8': 0, 'c9': 82, 'c10': 36, 'c11': 89, 'c12': 76}, 's10': {'c1': 52, 'c2': 52, 'c3': 127, 'c4': 185, 'c5': 37, 'c6': 165, 'c7': 23, 'c8': 23, 'c9': 87, 'c10': 89, 'c11': 72, 'c12': 87}},
    ...     agent_capacities=5,
    ...     item_capacities={'c1': 5.0, 'c2': 5.0, 'c3': 5.0, 'c4': 5.0, 'c5': 5.0, 'c6': 5.0, 'c7': 5.0, 'c8': 5.0, 'c9': 5.0, 'c10': 5.0, 'c11': 5.0, 'c12': 5.0})
    >>> initial_budgets = {'s1': 0.1650725918656969, 's2': 0.16262501524662654, 's3': 0.3201931268150584, 's4': 0.2492903523388018, 's5': 0.8017230433275404, 's6': 0.4141205417185544, 's7': 0.6544436816508201, 's8': 0.37386229094484114, 's9': 0.18748235872379515, 's10': 0.6342641285976163}
    >>> delta = 0.5
    >>> epsilon = 3
    >>> t = EFTBStatus.EF_TB
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets,
    ... delta=delta, epsilon=epsilon, t=t))
    '{s1:[], s10:[], s2:[], s3:[], s4:[], s5:[], s6:[], s7:[], s8:[], s9:[]}'
    """
    # allocation = [[0 for _ in range(instance.num_of_agents)] for _ in range(instance.num_of_items)]
    # 1) init prices vector to be 0

    initial_budgets = kwargs.get('initial_budgets')
    delta = kwargs.get('delta')
    epsilon = kwargs.get('epsilon')
    t = kwargs.get('t')

    logger.info("ACEEI_algorithms algorithm with initial budgets = %s, delta = %s, epsilon = %s, t = %s",
                initial_budgets, delta,
                epsilon, t)

    prices = {key: 0 for key in alloc.remaining_items()}
    clearing_error = 1
    new_budgets = {}
    combinations_courses_sorted = get_combinations_courses_sorted(alloc.instance)

    while clearing_error:
        # 2) 𝜖-budget perturbation
        new_budgets, clearing_error, allocation, excess_demand_per_course = find_budget_perturbation(
            initial_budgets, epsilon, prices, alloc.instance, t, combinations_courses_sorted)

        if clearing_error is None:
            logger.info("Clearing error is None - No Solution")
            # raise ValueError("Clearing error is None")
            break
        # 3) If ∥𝒛˜(𝒖,𝒄, 𝒑, 𝒃) ∥2 = 0, terminate with 𝒑* = 𝒑, 𝒃* = 𝒃
        logger.info("Clearing error is %s", clearing_error)
        if np.allclose(clearing_error, 0):
            break
        # 4) update 𝒑 ← 𝒑 + 𝛿𝒛˜(𝒖,𝒄, 𝒑, 𝒃), then go back to step 2.
        for key in prices:
            prices[key] += delta * excess_demand_per_course[key]
        logger.info("Update prices to %s\n", prices)

    logger.info("Clearing error 0!")
    for student, (price, bundle) in new_budgets.items():
        logger.info(f"Giving {bundle} to {student}")
        alloc.give_bundle(student, bundle)

    # print the final budget (b* = new_budgets) for each student
    final_budget = ""
    for key, value in new_budgets.items():
        final_budget += f"{key}: {value[0]}, "

    # Remove the trailing comma and space
    final_budget = final_budget.rstrip(", ")
    logger.info(f"\nfinal budget b* = {final_budget}")
    # print the final price (p* = prices) for each course
    logger.info(f"\nfinal prices p* = {prices}")


# ---------------------helper functions:---------------------
def student_best_bundle_per_budget(prices: dict, instance: Instance, epsilon: any, initial_budgets: dict,
                                   combinations_courses_sorted: dict):
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
    >>> combinations_courses_sorted = {'Alice': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Bob': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)]}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets, combinations_courses_sorted)
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
    >>> combinations_courses_sorted = {'Alice': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Bob': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)]}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets, combinations_courses_sorted)
    {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}

    # Alice: 3-7 -> (9, [x,y], p=3.5) (6, [x,z], p=1.5) (5, [y,z], p=2) (5 , x , p=1.5) (4, y, p=2) (1, z, p=0)
    # Bob: 2-6 -> (10, [x,y]. p=3.5) , (9, [y,z], p=2) , (7, [x.z] , p=1.5) , (6, [y] p=1.5) , (4, [x]. p= 1.5), (3, [z]), p=0)


    >>> instance = Instance(
    ...     valuations={"Alice":{"x":1, "y":1, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> initial_budgets = {"Alice": 5}
    >>> epsilon = 0.1
    >>> prices = {"x": 2, "y": 2, "z": 5}
    >>> combinations_courses_sorted = {'Alice': [('x', 'z'), ('y', 'z'), ('x', 'y'), ('z',), ('x',), ('y',)]}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets, combinations_courses_sorted)
    {'Alice': {4.9: ('x', 'y')}}

    Example with a student with no bundle
    >>> instance = Instance(
    ...     valuations={"avi":{"x":5}, "beni":{"x":5}},
    ...     agent_capacities=1,
    ...     item_capacities={"x":1})
    >>> initial_budgets = {"avi": 1.1, "beni": 1}
    >>> epsilon = 0.2
    >>> prices = {"x": 1.3}
    >>> combinations_courses_sorted = {'avi': [('x',)], 'beni': [('x',)]}
    >>> student_best_bundle_per_budget(prices, instance, epsilon,initial_budgets, combinations_courses_sorted)
    {'avi': {1.3: ('x',)}, 'beni': {0: ()}}
    """

    logger.debug("    student_best_bundle_per_budget for initial budgets = %s, prices = %s, epsilon = %s",
                 initial_budgets, prices, epsilon)
    best_bundle_per_budget = {student: {} for student in instance.agents}

    # logger.info("START combinations")
    for student in instance.agents:
        # Setting the min and max budget according to the definition
        min_budget = initial_budgets[student] - epsilon
        max_budget = initial_budgets[student] + epsilon

        # Sort the combinations of the courses in descending order according to utility. We went through the
        # budgets in descending order, for each budget we looked for the combination with the maximum value that
        # could be taken in that budget.
        min_price = float('inf')

        for combination in combinations_courses_sorted[student]:
            price_of_combination = sum(prices[course] for course in combination)

            if price_of_combination > max_budget:
                continue  # bundle is too expensive for student - irrelevant

            if price_of_combination <= min_budget:
                best_bundle_per_budget[student][min_budget] = combination
                break

            if price_of_combination < min_price:
                min_price = price_of_combination
                best_bundle_per_budget[student][price_of_combination] = combination
        if best_bundle_per_budget[student] == {}:
            best_bundle_per_budget[student][0] = ()
        logger.debug("    for student %s, the best bundles are %s", student, best_bundle_per_budget[student])

    return best_bundle_per_budget


def find_budget_perturbation(initial_budgets: dict, epsilon: float, prices: dict, instance: Instance, t: Enum,
                             combinations_courses_sorted: dict):
    logger.debug("  find_budget_perturbation for initial budgets = %s, prices = %s, epsilon = %s", initial_budgets,
                 prices, epsilon)
    map_student_to_best_bundle_per_budget = student_best_bundle_per_budget(prices, instance, epsilon, initial_budgets,
                                                                           combinations_courses_sorted)
    new_budgets, clearing_error, excess_demand_per_course = lp.optimize_model(
        map_student_to_best_bundle_per_budget, instance, prices, t, initial_budgets)
    logger.debug(
        "  Budget perturbation with lowest clearing error: new_budgets = %s, clearing_error = %s, excess_demand_per_course = %s",
        new_budgets, clearing_error, excess_demand_per_course)

    return new_budgets, clearing_error, map_student_to_best_bundle_per_budget, excess_demand_per_course


def ACEEI_without_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.001, epsilon=0.3, t=EFTBStatus.NO_EF_TB,
                                **kwargs)


def ACEEI_with_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.001, epsilon=0.3, t=EFTBStatus.EF_TB,
                                **kwargs)


def ACEEI_with_contested_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.001, epsilon=0.3,
                                t=EFTBStatus.CONTESTED_EF_TB, **kwargs)


def random_initial_budgets(num_of_agents: int) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [0, 1)
    initial_budgets = np.random.rand(num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}


def check_envy_in_allocation(instance: Instance, allocation: dict, initial_budgets: dict, t: Enum, prices: dict):
    """
    Checks if there is any envy in the allocation.

    :param instance: a fair-course-allocation instance
    :param allocation: A dictionary with students as keys and lists of allocated items as values.
    :param initial_budgets: Students' initial budgets
    :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB
    :param prices: courses prices

    :return: True if there is any envy in the allocation, False otherwise.

    >>> instance = Instance(
    ...     valuations={"avi":{"x":1, "y":2}, "beni":{"x":2, "y":3}},
    ...     agent_capacities=1,
    ...     item_capacities={"x":1, "y":1})
    >>> allocation = {"avi":['x'], "beni":['y']}
    >>> initial_budgets = {"avi":3, "beni":2}
    >>> t = EFTBStatus.EF_TB
    >>> prices = {"x":1, "y":2}
    >>> check_envy_in_allocation(instance, allocation, initial_budgets, t, prices)
    True

    >>> instance = Instance(
    ...     valuations={"avi":{"x":1, "y":2}, "beni":{"x":2, "y":3}},
    ...     agent_capacities=1,
    ...     item_capacities={"x":1, "y":1})
    >>> allocation = {"avi":['x'], "beni":['y']}
    >>> initial_budgets = {"avi":1, "beni":2}
    >>> t = EFTBStatus.EF_TB
    >>> prices = {"x":1, "y":2}
    >>> check_envy_in_allocation(instance, allocation, initial_budgets, t, prices)
    False


    """
    allocation = adjusting_the_allocation_format(allocation, prices)
    for student1, student2 in combinations(instance.agents, 2):
        if initial_budgets[student1] > initial_budgets[student2]:
            if lp.check_envy(instance, student1, student2, allocation, t, prices):
                return True
    return False



def adjusting_the_allocation_format(allocation: dict, prices: dict):
    """
    Adjusts the allocation format by calculating the total price of allocated courses for each student.


    :param allocation: A dictionary where the keys are student names and the values are lists of course names
                        allocated to them.
    :param prices: A dictionary where the keys are course names and the values are their
                    corresponding prices.

    :return: A dictionary where the keys are student names and the values are dictionaries with the total price of
            allocated courses as the key and a tuple of course names as the value.

    >>> allocation = {"Alice":['x', 'z'], "Bob":['y', 'z']}
    >>> prices = {"x": 1.5, "y": 2, "z": 0}
    >>> adjusting_the_allocation_format(allocation, prices)
    {'Alice': {1.5: ('x', 'z')}, 'Bob': {2: ('y', 'z')}}
    """
    adjusted_allocation = {}
    for student, courses in allocation.items():
        total_price = sum(prices[course] for course in courses)
        adjusted_allocation[student] = {total_price: tuple(courses)}
    return adjusted_allocation


if __name__ == "__main__":
    import doctest

    print("\n", doctest.testmod(), "\n")
    # sys.exit(0)

    from fairpyx.adaptors import divide

    logger.setLevel(logging.DEBUG)
    lp.logger.setLevel(logging.WARNING)

    import coloredlogs

    level_styles = {
        'debug': {'color': 'green'},
        'info': {'color': 'cyan'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'bold': True, 'background': 'white'}
    }
    coloredlogs.install(level='DEBUG', logger=logger, fmt='%(message)s', level_styles=level_styles)

    # instance = Instance(
    #     valuations={"alice": {"CS161": 5, "ECON101": 3, "IR": 6},
    #                 "bob": {"CS161": 3, "ECON101": 2, "IR": 0},
    #                 "eve-1": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-2": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-3": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-4": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-5": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-6": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-7": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-8": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-9": {"CS161": 0, "ECON101": 10, "IR": 1},
    #                 "eve-10": {"CS161": 0, "ECON101": 10, "IR": 1}},
    #     agent_capacities=2,
    #     item_capacities={"CS161": 1, "ECON101": 10, "IR": 100})
    # initial_budgets = {"alice": 4.7, "bob": 4.4, "eve-1": 6, "eve-2": 1, "eve-3": 1, "eve-4": 1, "eve-5": 1, "eve-6": 1,
    #                    "eve-7": 1, "eve-8": 1, "eve-9": 1, "eve-10": 1}
    # delta = 0.5
    # epsilon = 0.5
    # t = EFTBStatus.EF_TB
    #
    # print(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets, delta=delta, epsilon=epsilon,
    #              t=t))
    #
    # log_capture_handler = LogCaptureHandler()
    # logging.getLogger().addHandler(log_capture_handler)
    #
    # instance = Instance(
    #     valuations={"alice": {"CS161": 5, "ECON101": 3, "IR": 6}, "bob": {"CS161": 3, "ECON101": 5, "IR": 0},
    #                 "eve": {"CS161": 1, "ECON101": 10, "IR": 0}},
    #     agent_capacities=2,
    #     item_capacities={"CS161": 1, "ECON101": 1, "IR": 100000})
    # initial_budgets = {"alice": 2, "bob": 1, "eve": 4}
    # delta = 0.5
    # epsilon = 0.5
    # t = EFTBStatus.EF_TB
    # print(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets, delta=delta, epsilon=epsilon,
    #              t=t))
    #
    # logs = log_capture_handler.get_logs()
    # print(f" --------logs----------\n{logs}")
    #
    # print(f"prices ==== {log_capture_handler.extract_prices()}")
