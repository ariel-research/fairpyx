"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import numpy as np
import fairpyx

import logging
logger = logging.getLogger(__name__)

def SP_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 2: Allocate the given items to the given agents using the SP protocol.

    SP (Second Price) in each round distributes one course to each student, with the refund of the bids according to the price of the course.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    >>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    >>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    >>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(SP_function, instance=instance)
    {'s1': ['c1'], 's2': ['c2']}
    """

    explanation_logger.info("\nAlgorithm SP starts.\n")

    map_agent_to_best_item = {}  # dict of the max bids for each agent in specific iteration (from the article: max{i, b})
    map_student_to_his_sum_bids = {s: 0 for s in alloc.remaining_agents()}  # the amount of bids agent have from all the courses he got before

    # {'s1': {} , 's2': {} ...}
    map_student_to_course_with_no_seats_and_the_bids = {student: {} for student in alloc.remaining_agents()} # map students to courses he can't get for update bids
    all_agents = set(alloc.remaining_agents())
    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    explanation_logger.debug("Max iterations: %d", max_iterations)
    for iteration in range(max_iterations):  # External loop of algorithm: in each iteration, each student gets 1 seat (if possible).
        explanation_logger.info("\nIteration number: %d", iteration+1)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            explanation_logger.info("There are no more agents (%d) or items(%d) ",len(alloc.remaining_agent_capacities), len(alloc.remaining_item_capacities))
            break


        agents_who_need_an_item_in_current_iteration = set(alloc.remaining_agents())  # only the agents that still need courses

        # Find the difference between all_agents and agents_who_need_an_item_in_current_iteration
        agents_not_in_need = all_agents - agents_who_need_an_item_in_current_iteration
        # If you need the result as a list:
        agents_not_in_need_list = list(agents_not_in_need)
        for student in agents_not_in_need_list:
            explanation_logger.info("There are no more items you can get", agents=student)
        while agents_who_need_an_item_in_current_iteration:  # round i
            # 1. Create the dict map_agent_to_best_item
            agents_with_no_potential_items = set()
            for current_agent in agents_who_need_an_item_in_current_iteration:  # check the course with the max bids for each agent in the set (who didnt get a course in this round)


                potential_items_for_agent = alloc.remaining_items_for_agent(current_agent)       # set of all the courses that have places sub the courses the agent got
                if len(potential_items_for_agent) == 0:
                    agents_with_no_potential_items.add(current_agent)
                else:
                    explanation_logger.debug("\n the cources you can get are: %s", potential_items_for_agent,
                                            agents=current_agent)
                    map_agent_to_best_item[current_agent] = max(potential_items_for_agent,
                                                                key=lambda item: alloc.effective_value(current_agent, item))  # for each agent save the course with the max bids from the potential courses only
                    current_course = map_agent_to_best_item[current_agent]
                    pop_course_that_moved_bids = []
                    if current_agent in map_student_to_course_with_no_seats_and_the_bids:
                        for course_that_no_longer_potential in map_student_to_course_with_no_seats_and_the_bids[current_agent]:
                            # if the algorithm pass a course that the student can't get - add his bids if needed
                            if alloc.effective_value(current_agent, current_course) < map_student_to_course_with_no_seats_and_the_bids[current_agent][course_that_no_longer_potential]:
                                map_student_to_his_sum_bids[current_agent] += map_student_to_course_with_no_seats_and_the_bids[current_agent][course_that_no_longer_potential]
                                pop_course_that_moved_bids.append(course_that_no_longer_potential) #save the courses should delete
                for course in pop_course_that_moved_bids: # remove the courses that the algorithm add the bids
                    map_student_to_course_with_no_seats_and_the_bids[current_agent].pop(course, None)

            for current_agent in agents_with_no_potential_items:  # remove agents that don't need courses
                explanation_logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", current_agent,
                            alloc.remaining_item_capacities, alloc.bundles[current_agent], agents=current_agent)
                alloc.remove_agent_from_loop(current_agent)
                agents_who_need_an_item_in_current_iteration.remove(current_agent)

            # 2. Allocate the remaining seats in each course:
            for student, course in map_agent_to_best_item.items():  # update the bids of each student
                map_student_to_his_sum_bids[student] += alloc.effective_value(student, course)

            # create dict for each course of student that point on the course and their bids
            map_course_to_students_with_max_bids = {course:
                {student: map_student_to_his_sum_bids[student] for student in map_agent_to_best_item if map_agent_to_best_item[student] == course}
                for course in alloc.remaining_items()}


            # loop on the dict_by_course and for each course give it to as many students it can
            for course in map_course_to_students_with_max_bids:
                if course in alloc.remaining_item_capacities:
                    sorted_students_pointing_to_course = sorted(
                        [student for student in map_agent_to_best_item if map_agent_to_best_item[student] == course],
                        key=lambda student: map_student_to_his_sum_bids[student],
                        reverse=True)  # sort the keys by their values (descending order)
                    remaining_capacity = int(alloc.remaining_item_capacities[course])  # the amount of seats left in the current course
                    sorted_students_who_can_get_course = sorted_students_pointing_to_course[:remaining_capacity]  # list of the student that can get the course
                    price = 0  # the course price (0 if evert student that want the course this round can get it)
                    if len(sorted_students_pointing_to_course) > remaining_capacity:
                        price = map_student_to_his_sum_bids[sorted_students_pointing_to_course[remaining_capacity]]  # the amount of bids of the first studen that cant get the course
                    explanation_logger.debug("The price of course %s is:%d",course,price)
                    if len(sorted_students_pointing_to_course) >= remaining_capacity: # if there are equal or more students than capacity
                        for student in alloc.remaining_agents():
                            if student in sorted_students_pointing_to_course[remaining_capacity:]:
                                continue
                            map_student_to_course_with_no_seats_and_the_bids[student][course] = alloc.effective_value(student, course) # save the bids for each student for the courses that there aren't more seats
                    for student in sorted_students_who_can_get_course:
                        for conflict_course in alloc.instance.item_conflicts(course): #each course that have a conflict with the current course
                            if student in sorted_students_pointing_to_course[remaining_capacity:]:
                                continue
                            map_student_to_course_with_no_seats_and_the_bids[student][conflict_course] = alloc.effective_value(student, conflict_course) # save the bids for each student for the courses that have conflict
                        explanation_logger.info("you get course %s which you gave %d bids for the price: %d", course, map_student_to_his_sum_bids[student], price, agents=student)
                        alloc.give(student, course, logger)
                        agents_who_need_an_item_in_current_iteration.remove(student)  # removing the agent from the set (dont worry he will come back in the next round)
                        map_agent_to_best_item.pop(student, None)  # Delete student if exists in the dict
                        del map_course_to_students_with_max_bids[course][student]
                        map_student_to_his_sum_bids[student] -= price  # remove the price of the course from the bids of the student who got the course
                        map_student_to_course_with_no_seats_and_the_bids[student].pop(course, None) # remove the course from the dict because the student get this course


if __name__ == "__main__":
    import doctest, sys
    print("\n", doctest.testmod(), "\n")
    # sys.exit(1)

    #logger.addHandler(logging.StreamHandler())
    #logger.setLevel(logging.INFO)

    # from fairpyx.adaptors import divide
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
    # # divide(SP_function, instance=instance)
    #
    # np.random.seed(1)
    # instance = Instance.random_uniform(
    #     num_of_agents=70, num_of_items=10, normalized_sum_of_values=100,
    #     agent_capacity_bounds=[2, 6],
    #     item_capacity_bounds=[10, 30],
    #     item_base_value_bounds=[1, 1000],
    #     item_subjective_ratio_bounds=[0.5, 1.5]
    # )
    #
    # allocation = divide(SP_function, instance=instance)

    from fairpyx.adaptors import divide_random_instance, divide
    from fairpyx.explanations import ConsoleExplanationLogger, FilesExplanationLogger, StringsExplanationLogger

    num_of_agents = 5
    num_of_items = 3

    # console_explanation_logger = ConsoleExplanationLogger(level=logging.INFO)
    # files_explanation_logger = FilesExplanationLogger({
    #     f"s{i + 1}": f"logs/s{i + 1}.log"
    #     for i in range(num_of_agents)
    # }, mode='w', language="he")
    string_explanation_logger = StringsExplanationLogger([f"s{i + 1}" for i in range(num_of_agents)], level=logging.INFO)

    # print("\n\nIterated Maximum Matching without adjustments:")
    # divide_random_instance(algorithm=iterated_maximum_matching, adjust_utilities=False,
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12],
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)

    print("\n\nIterated Maximum Matching with adjustments:")
    divide_random_instance(algorithm=SP_function,
                           # explanation_logger=console_explanation_logger,
                           #    explanation_logger = files_explanation_logger,
                           explanation_logger=string_explanation_logger,
                           num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2, 5],
                           item_capacity_bounds=[3, 12],
                           item_base_value_bounds=[1, 100], item_subjective_ratio_bounds=[0.5, 1.5],
                           normalized_sum_of_values=100,
                           random_seed=1)

    # print(string_explanation_logger.map_agent_to_explanation())
    print(string_explanation_logger.map_agent_to_explanation()["s1"])
