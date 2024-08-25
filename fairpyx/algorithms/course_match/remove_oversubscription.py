"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation
Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Naama Shiponi and Ben Dabush
1/6/2024
"""
import logging
logger = logging.getLogger(__name__)
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.course_match.A_CEEI import (
    compute_surplus_demand_for_each_course,
    find_best_schedule,
    find_preference_order_for_each_student,
)

"""
Algorithm 2 : The algorithm makes sure that there are no courses that have more students registered than their capacity.
"""


def remove_oversubscription(
    allocation: AllocationBuilder,
    price_vector: dict,
    student_budgets: dict,
    epsilon: float = 0.1,
    compute_surplus_demand_for_each_course: callable = compute_surplus_demand_for_each_course,
):
    """
    Perform oversubscription elimination to adjust course prices.

    :param allocation: AllocationBuilder
    :param price_vector: Initial price vector (dict of floats)
    :param student_budgets: dict of student budgets (dict of floats)
    :param epsilon: Small value to determine when to stop binary search
    :param demand_function: Function that takes price vector and returns excess demand vector

    :return: Adjusted price vector (dict of floats)

    :pseudo code
    Input:  p* heuristic search solution price vector from Algorithm 1,
            ¯p scalar price greater than any budget,
            ε smallerthan budget differences,
            excess demand function d(p) that maps a price vector to the demand of a coursebeyond its maximum capacity.
    Output: Altered p* without oversubscription

    1:  j' ← argMax_j (d_j (p*))  # j' is the most oversubscribed course
    2:  while d_j'(p*) > 0 do
    3:      d* ← d_j'(p*)/2  # Perform binary search on the price of course j' until oversubscription equals (at most) d*
    4:      pl ← p*_j'
    5:      ph ← ¯p
    6:      repeat  # Our target price is always in the interval [pl ,ph], which we progressively shrink inhalf in each iteration of this loop
    7:          p*_j' ← (pl + ph )/2
    8:          if d_j'(p*) > d* then
    9:              pl ← p*_j'
    10:         else
    11:             ph ← p*_j'
    12:         end if
    13:     until ph - pl < ε
    14:     p*_j' ← ph  # Set to the higher price to be sure oversubscription is at most d*
    15:     j' ← argMax_j d_j(p*)  # Find the most oversubscribed course with the new prices
    16: end while

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                      "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                      "Tom": {"c1": 70, "c2": 30, "c3": 70}}
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {"c1": 1.2, "c2": 0.9, "c3": 1}
    >>> epsilon = 0.1
    >>> student_budgets = {"Alice": 2.2, "Bob": 2.1, "Tom": 2.0}
    >>> remove_oversubscription(allocation, price_vector, student_budgets, epsilon, compute_surplus_demand_for_each_course)
    {'c1': 2.0421875000000003, 'c2': 1.1515624999999998, 'c3': 2.0562500000000004}

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 3, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"c1": 3, "c2": 3, "c3": 3},
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                      "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                      "Tom": {"c1": 70, "c2": 30, "c3": 70}}
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {"c1": 1.2, "c2": 0.9, "c3": 1}
    >>> epsilon = 0.1
    >>> student_budgets = {"Alice": 2.2, "Bob": 2.1, "Tom": 2.0}
    >>> remove_oversubscription(allocation, price_vector, student_budgets, epsilon, compute_surplus_demand_for_each_course)
    {'c1': 1.2, 'c2': 0.9, 'c3': 1}

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 3, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                      "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                      "Tom": {"c1": 70, "c2": 30, "c3": 70}}
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {"c1": 0, "c2": 0, "c3": 0}
    >>> epsilon = 0.1
    >>> student_budgets = {"Alice": 2.2, "Bob": 2.1, "Tom": 2.0}
    >>> remove_oversubscription(allocation, price_vector, student_budgets, epsilon, compute_surplus_demand_for_each_course)
    {'c1': 2.0125, 'c2': 0.21113281250000004, 'c3': 2.0125}
    """
    max_budget = max(student_budgets.values()) + epsilon
    logger.debug('Max budget set to %g', max_budget)
    item_conflicts = {
        item: allocation.instance.item_conflicts(item)
        for item in allocation.instance.items
    }
    agent_conflicts = {
        agent: allocation.instance.agent_conflicts(agent)
        for agent in allocation.instance.agents
    }

    preferred_schedule = find_preference_order_for_each_student(
        allocation.instance._valuations,
        allocation.instance._agent_capacities,
        item_conflicts,
        agent_conflicts,
    )
    while True:
        excess_demands = compute_surplus_demand_for_each_course(price_vector, allocation, student_budgets, preferred_schedule)
        highest_demand_course = max(excess_demands, key=excess_demands.get)
        highest_demand = excess_demands[highest_demand_course]
        logger.debug('Highest demand course: %s with demand %g', highest_demand_course, highest_demand)
        if highest_demand <= 0:
            break

        d_star = highest_demand / 2
        low_price = price_vector[highest_demand_course]
        high_price = max_budget

        logger.info('Starting binary search for course %s', highest_demand_course)
        while high_price - low_price >= epsilon:
            p_mid = (low_price + high_price) / 2
            price_vector[highest_demand_course] = p_mid
            current_demand = compute_surplus_demand_for_each_course(price_vector, allocation, student_budgets, preferred_schedule)[highest_demand_course]
            logger.debug('Mid price set to %g, current demand %g', p_mid, current_demand)
            if current_demand > d_star:
                low_price = p_mid
                logger.debug('Current demand %g is greater than d_star %g, updating low_price to %g', current_demand, d_star, low_price)
            else:
                high_price = p_mid
                logger.debug('Current demand %g is less than or equal to d_star %g, updating high_price to %g', current_demand, d_star, high_price)

        price_vector[highest_demand_course] = high_price
        logger.info('Final price for course %s set to %g', highest_demand_course, high_price)
    logger.info('Final price vector after remove_oversubscription %s', price_vector, )

    return price_vector

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

    instance = Instance(
        agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
        item_capacities={"c1": 1, "c2": 1, "c3": 1},
        valuations={
            "Alice": {"c1": 50, "c2": 20, "c3": 80},
            "Bob": {"c1": 60, "c2": 40, "c3": 30},
            "Tom": {"c1": 70, "c2": 30, "c3": 70},
        },
    )
    allocation = AllocationBuilder(instance)
    price_vector = {"c1": 1.2, "c2": 0.9, "c3": 1}
    epsilon = 0.1
    student_budgets = {"Alice": 2.2, "Bob": 2.1, "Tom": 2.0}

    max_budget = max(student_budgets.values()) + 0.01  # ¯p scalar price greater than any budget
    # preferred_schedule = find_preferred_schedule(allocation.instance._valuations, allocation.instance._agent_capacities, allocation.instance.item_conflicts, allocation.instance.agent_conflicts)
    item_conflicts = {
        item: allocation.instance.item_conflicts(item)
        for item in allocation.instance.items
    }
    agent_conflicts = {
        agent: allocation.instance.agent_conflicts(agent)
        for agent in allocation.instance.agents
    }

    preferred_schedule = find_preference_order_for_each_student(
        allocation.instance._valuations,
        allocation.instance._agent_capacities,
        item_conflicts,
        agent_conflicts,
    )
    # print(preferred_schedule)
    # remove_oversubscription(allocation, price_vector, student_budgets, epsilon, compute_surplus_demand_for_each_course)
