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
import numpy as np

from fairpyx import Instance, AllocationBuilder
from fairpyx.utils import linear_program as lp
from fairpyx.utils.calculate_combinations import get_combinations_courses_sorted


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
    """
    # allocation = [[0 for _ in range(instance.num_of_agents)] for _ in range(instance.num_of_items)]
    # 1) init prices vector to be 0

    initial_budgets = kwargs.get('initial_budgets')
    delta = kwargs.get('delta')
    epsilon = kwargs.get('epsilon')
    t = kwargs.get('t')

    logger.info("ACEEI algorithm with initial budgets = %s, delta = %s, epsilon = %s, t = %s", initial_budgets, delta,
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
            raise ValueError("Clearing error is None")
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
    if clearing_error is None:
        raise ValueError("Clearing error is None")
    return new_budgets, clearing_error, map_student_to_best_bundle_per_budget, excess_demand_per_course


def ACEEI_without_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.5, epsilon=3.0, t=EFTBStatus.NO_EF_TB,
                                **kwargs)


def ACEEI_with_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.5, epsilon=3.0, t=EFTBStatus.EF_TB,
                                **kwargs)


def ACEEI_with_contested_EFTB(alloc: AllocationBuilder, **kwargs):
    initial_budgets = random_initial_budgets(alloc.instance.num_of_agents)
    return find_ACEEI_with_EFTB(alloc, initial_budgets=initial_budgets, delta=0.5, epsilon=3.0,
                                t=EFTBStatus.CONTESTED_EF_TB, **kwargs)


def random_initial_budgets(num_of_agents: int) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [0, 1)
    initial_budgets = np.random.rand(num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}



if __name__ == "__main__":
    import doctest, sys

    print("\n", doctest.testmod(), "\n")
    # sys.exit(0)

    from fairpyx.adaptors import divide

    logger.setLevel(logging.INFO)
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

    instance = Instance(
        valuations={"alice": {"CS161": 5, "ECON101": 3, "IR": 6},
                    "bob": {"CS161": 3, "ECON101": 2, "IR": 0},
                    "eve-1": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-2": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-3": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-4": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-5": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-6": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-7": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-8": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-9": {"CS161": 0, "ECON101": 10, "IR": 1},
                    "eve-10": {"CS161": 0, "ECON101": 10, "IR": 1}},
        agent_capacities=2,
        item_capacities={"CS161": 1, "ECON101": 10, "IR": 100})
    initial_budgets = {"alice": 4.7, "bob": 4.4, "eve-1": 6, "eve-2": 1, "eve-3": 1, "eve-4": 1, "eve-5": 1, "eve-6": 1,
                       "eve-7": 1, "eve-8": 1, "eve-9": 1, "eve-10": 1}
    delta = 0.5
    epsilon = 0.5
    t = EFTBStatus.EF_TB
    
    print(divide(find_ACEEI_with_EFTB, instance=instance, initial_budgets=initial_budgets, delta=delta, epsilon=epsilon,
                 t=t))

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
