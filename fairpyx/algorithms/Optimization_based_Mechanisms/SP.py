"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
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

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    for iteration in range(max_iterations):  # External loop of algorithm: in each iteration, each student gets 1 seat (if possible).
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            break
        agents_who_need_an_item_in_current_iteration = set(alloc.remaining_agents())  # only the agents that still need courses
        while agents_who_need_an_item_in_current_iteration:  # round i
            # 1. Create the dict map_agent_to_best_item
            agents_with_no_potential_items = set()
            for current_agent in agents_who_need_an_item_in_current_iteration:  # check the course with the max bids for each agent in the set (who didnt get a course in this round)


                potential_items_for_agent = alloc.remaining_items_for_agent(current_agent)       # set of all the courses that have places sub the courses the agent got
                if len(potential_items_for_agent) == 0:
                    agents_with_no_potential_items.add(current_agent)
                else:
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
                logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", current_agent,
                            alloc.remaining_item_capacities, alloc.bundles[current_agent])
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
                    remaining_capacity = alloc.remaining_item_capacities[course]  # the amount of seats left in the current course
                    sorted_students_who_can_get_course = sorted_students_pointing_to_course[:remaining_capacity]  # list of the student that can get the course
                    price = 0  # the course price (0 if evert student that want the course this round can get it)
                    if len(sorted_students_pointing_to_course) > remaining_capacity:
                        price = map_student_to_his_sum_bids[sorted_students_pointing_to_course[remaining_capacity]]  # the amount of bids of the first studen that cant get the course
                    if len(sorted_students_pointing_to_course) >= remaining_capacity: # if there are equal or more students than capacity
                        for student in alloc.remaining_agents():
                            map_student_to_course_with_no_seats_and_the_bids[student][course] = alloc.effective_value(student, course) # save the bids for each student for the courses that there aren't more seats
                    for student in sorted_students_who_can_get_course:
                        for conflict_course in alloc.instance.item_conflicts(course): #each course that have a conflict with the current course
                            map_student_to_course_with_no_seats_and_the_bids[student][conflict_course] = alloc.effective_value(student, conflict_course) # save the bids for each student for the courses that have conflict
                        alloc.give(student, course, logger)
                        agents_who_need_an_item_in_current_iteration.remove(student)  # removing the agent from the set (dont worry he will come back in the next round)
                        map_agent_to_best_item.pop(student, None)  # Delete student if exists in the dict
                        del map_course_to_students_with_max_bids[course][student]
                        map_student_to_his_sum_bids[student] -= price  # remove the price of the course from the bids of the student who got the course
                        map_student_to_course_with_no_seats_and_the_bids[student].pop(course, None) # remove the course from the dict because the student get this course


if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())