"""
Implement a "Find a profitable manipulation for a student",

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-05
"""
import logging
from enum import Enum

import numpy as np

from fairpyx import Instance, AllocationBuilder


class criteria_for_profitable_manipulation(Enum):
    resampled_randomness = 0
    population = 1


logger = logging.getLogger(__name__)


# TODO: ask erel how to change the instance for the criteria population - how to change moti values to the correct


def random_initial_budgets(instance: Instance, beta: float):
    return {agent: np.random.uniform(1 + (beta / 4), 1 + ((3 * beta) / 4)) for agent in instance.agents}


def find_profitable_manipulation(mechanism: callable, student: str, utility: dict,
                                 criteria: Enum, neu: float, instance: Instance, delta: float, epsilon: float, t: Enum, initial_budgets: dict,
                                 beta: float):
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
    :param beta: a parameter that determines the distribution of the initial budgets

    return: The profitable manipulation

    >>> from fairpyx.algorithms.ACEEI import find_ACEEI_with_EFTB
    >>> from fairpyx.algorithms import ACEEI, tabu_search
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
    >>> beta = 2
    >>> initial_budgets = random_initial_budgets(instance, beta)
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t, initial_budgets, beta)
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
    >>> beta = 2
    >>> initial_budgets = random_initial_budgets(instance, beta)
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t, initial_budgets, beta)
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
    >>> beta = 2
    >>> initial_budgets = random_initial_budgets(instance, beta)
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t, initial_budgets, beta)
    {"x":6, "y":2}

    Example run 5
    >>> mechanism = find_ACEEI_with_EFTB
    >>> student = "moti"
    >>> utility = {"x":1, "y":2, "z":5}
    >>> criteria = criteria_for_profitable_manipulation.population
    >>> neu = 2
    >>> instance = Instance.random_uniform(num_of_agents=3, num_of_items=3, agent_capacity_bounds=(2,2),
    ... item_capacity_bounds=(1,3), item_base_value_bounds=(1, 5),
    ... item_subjective_ratio_bounds=(1, 1.5),
    ... normalized_sum_of_values=1000)
    >>> beta = 2
    >>> initial_budgets = random_initial_budgets(instance, beta)
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = ACEEI.EFTBStatus.NO_EF_TB
    >>> find_profitable_manipulation(mechanism, student, utility, criteria, neu, instance, delta, epsilon, t, initial_budgets, beta)
    {"x":1, "y":2, "z":5}

   """
