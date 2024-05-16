"""
Implement a "Find a profitable manipulation for a student",

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-05
"""
import logging
from enum import Enum
from fairpyx import Instance, AllocationBuilder


class criteria_for_profitable_manipulation(Enum):
    resampled_randomness = 0
    population = 1


logger = logging.getLogger(__name__)

# TODO ask erel about the mechanism type
# TODO ask erel how to test random
# TODO ask erel how to pass utilities in population (it is random)
# TODO ask erel handle the instance
# TODO ask erel about large tests

# TODO check if we need the utility
def find_profitable_manipulation(mechanism: callable, student: str, utility: dict,
                                 criteria: Enum, neu: float, alloc: AllocationBuilder, delta: float, epsilon: float, t: Enum):
    """
   "Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
    by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
    ALGORITHM 2: Find a profitable manipulation for a student

    :param mechanism: A randomized mechanism M for course-allocation
    :param student: The student who is being tested to see if he can manipulate
    :param utility: The student's utility
    :param criteria: The type of criteria for profitable manipulation
                                                 0 for resampled randomness
                                                 1 for population
    :param neu: a local update coefficient neu
    :param alloc: a fair-course-allocation instance
    :param initial_budgets: Students' initial budgets
    :param delta: The step size
    :param epsilon: maximum budget perturbation
    :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB

    return: The profitable manipulation

    >>> from fairpyx.algorithms.ACEEI import find_ACEEI_with_EFTB
    >>> from fairpyx.algorithms import ACEEI
    >>> logger.addHandler(logging.StreamHandler())
    >>> logger.setLevel(logging.INFO)

    Example run 1
    >>> mechanism = find_ACEEI_with_EFTB
    >>> student = "moti"
    >>> utility = {"x":1, "y":2, "z":4}
    >>> criteria = criteria_for_profitable_manipulation.resampled_randomness
    >>> neu = 2
    >>> instance = Instance(
    ...     valuations={"avi":{"x":3, "y":5, "z":1}, "beni":{"x":2, "y":3, "z":1}, "moti": {"x":1, "y":2, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":3})
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t)
    {"x":1, "y":2, "z":4}

    Example run 2
    >>> mechanism = find_ACEEI_with_EFTB
    >>> student = "moti"
    >>> utility = {"x":1, "y":2, "z":4}
    >>> criteria = criteria_for_profitable_manipulation.resampled_randomness
    >>> neu = 2
    >>> instance = Instance(
    ...     valuations={"avi":{"x":3, "y":5, "z":1}, "beni":{"x":2, "y":3, "z":1}, "moti": {"x":1, "y":2, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":3})
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t)
    {"x":1, "y":2, "z":4}


    Example run 4
    >>> mechanism = find_ACEEI_with_EFTB
    >>> student = "moti"
    >>> utility = {"x":6, "y":2}
    >>> criteria = criteria_for_profitable_manipulation.resampled_randomness
    >>> neu = 2
    >>> instance = Instance(
    ...     valuations={"avi":{"x":5, "y":3}, "moti": {"x":6, "y":2}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2})
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t)
    {"x":6, "y":2}

    Example run 5
    >>> mechanism = find_ACEEI_with_EFTB
    >>> student = "moti"
    >>> utility = {"x":1, "y":2, "z":5}
    >>> criteria = criteria_for_profitable_manipulation.population
    >>> neu = 2
    >>> instance = Instance(
    ...     valuations={"moti": {"x":1, "y":2, "z":5}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":3})
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t)
    {"x":1, "y":2, "z":5}

   """
