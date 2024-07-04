"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from itertools import cycle

import logging
logger = logging.getLogger(__name__)


def TTC_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorithm: Allocate the given items to the given agents using the TTC protocol.

    TTC (top trading-cycle) assigns one course in each round to each student, the winning students are defined based on the students’ bid values.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of the fair course allocation problem(CAP).
    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    >>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    >>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    >>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(TTC_function, instance=instance)
    {'s1': ['c1'], 's2': ['c2']}
    """
    explanation_logger.info("\nAlgorithm TTC starts.\n")

    map_agent_to_best_item = {}               # dict of the max bids for each agent in specific iteration (from the article: max{i, b})
    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    logger.info("Max iterations: %d", max_iterations)
    for iteration in range(max_iterations):   # External loop of algorithm: in each iteration, each student gets 1 seat (if possible).
        logger.info("\n Iteration number: %d", iteration+1)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more courses with seats
            logger.info("There are no more agents (%d) or items(%d) ",len(alloc.remaining_agent_capacities), len(alloc.remaining_item_capacities))
            break
        agents_who_need_an_item_in_current_iteration = set(alloc.remaining_agents())  # only the agents that still need courses
        while agents_who_need_an_item_in_current_iteration:     # round i
            # 1. Create the dict map_agent_to_best_item
            agents_with_no_potential_items = set()  # agents that don't need courses to be removed later
            for current_agent in agents_who_need_an_item_in_current_iteration:    # check the course with the max bids for each agent in the set (who didnt get a course in this round)
                potential_items_for_agent = alloc.remaining_items_for_agent(current_agent)       # set of all the courses that have places sub the courses the agent got
                if len(potential_items_for_agent) == 0:
                    agents_with_no_potential_items.add(current_agent)
                else:
                    map_agent_to_best_item[current_agent] = max(potential_items_for_agent,
                                                          key=lambda item: alloc.effective_value(current_agent, item))  # for each agent save the course with the max bids from the potential courses only

            logger.debug("map_agent_to_best_item = %s", map_agent_to_best_item)
            for current_agent in agents_with_no_potential_items:  # remove agents that don't need courses
                logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", current_agent,
                            alloc.remaining_item_capacities, alloc.bundles[current_agent])
                alloc.remove_agent_from_loop(current_agent)
                agents_who_need_an_item_in_current_iteration.remove(current_agent)

            # 2. Allocate the remaining seats in each course:
            for course in list(alloc.remaining_items()):
                sorted_students_pointing_to_course = sorted(
                    [student for student in map_agent_to_best_item if map_agent_to_best_item[student] == course],
                    key=lambda student: alloc.effective_value(student,course),
                    reverse=True)  # sort the keys by their values (descending order)
                remaining_capacity = alloc.remaining_item_capacities[course]  # the amount of seats left in the current course
                sorted_students_who_can_get_course = sorted_students_pointing_to_course[:remaining_capacity]  # list of the student that can get the course
                for student in sorted_students_who_can_get_course:
                    alloc.give(student, course, logger)
                    agents_who_need_an_item_in_current_iteration.remove(student)    # removing the agent from the set (dont worry he will come back in the next round)
                    map_agent_to_best_item.pop(student, None)   # Delete student if exists in the dict

if __name__ == "__main__":
    import doctest, sys, numpy as np
    print("\n", doctest.testmod(), "\n")
    # sys.exit(1)

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    from fairpyx.adaptors import divide

    # s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}
    # s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}
    # s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}
    # s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}
    # s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}
    # instance = Instance(
    #     agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2},
    #     item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2},
    #     valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    # )
    # divide(TTC_function, instance=instance)

    np.random.seed(1)
    instance = Instance.random_uniform(
        num_of_agents=70, num_of_items=10, normalized_sum_of_values=100,
        agent_capacity_bounds=[2, 6],
        item_capacity_bounds=[20, 40],
        item_base_value_bounds=[1, 1000],
        item_subjective_ratio_bounds=[0.5, 1.5]
    )

    allocation = divide(TTC_function, instance=instance)

