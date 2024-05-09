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

    # agent_order = []                                                    # list of all the instances of agents as many as their items capacities (from the article: sum of Ki)
    map_agent_to_best_item = {}                                           # dict of the max bids for each agent in specific iteration (from the article: max{i, b})
    map_course_to_students_with_max_bids = {c: {} for c in alloc.remaining_item_capacities}

    #for agent, capacity in alloc.remaining_agent_capacities.items():    # save the agent number of his needed courses (שומר את הסטודנט מספר פעמים כמספר הקורסים שהוא צריך)
    #    agent_order.extend([agent] * capacity)

    #set_agent = list(dict.fromkeys(agent_order))                        # set of the agents by thier order . each agent appears only once

    map_student_to_his_sum_bids = {s: 0 for s in alloc.remaining_agents()}

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())
    for iteration in range(max_iterations):  # External loop of algorithm: in each iteration, each student gets 1 seat (if possible).
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            break
        agents_who_need_an_item_in_current_iteration = set(alloc.remaining_agents())
        while agents_who_need_an_item_in_current_iteration:  # round i
            # 1. Create the dict map_agent_to_best_item
            agents_with_no_potential_items = set()
            for current_agent in agents_who_need_an_item_in_current_iteration:  # check the course with the max bids for each agent in the set (who didnt get a course in this round)
                # potential_items_for_agent = set(alloc.remaining_items()).difference(alloc.bundles[current_agent])  # set of all the courses that have places sub the courses the agent got (under the assumption that if agent doesnt want a course the course got 0 bids automaticlly)
                potential_items_for_agent = alloc.remaining_items_for_agent(current_agent)       # set of all the courses that have places sub the courses the agent got (under the assumption that if agent doesnt want a course the course got 0 bids automaticlly)
                if len(potential_items_for_agent) == 0:
                    agents_with_no_potential_items.add(current_agent)
                else:
                    map_agent_to_best_item[current_agent] = max(potential_items_for_agent,
                                                                key=lambda item: alloc.effective_value(current_agent,item))  # for each agent save the course with the max bids from the potential courses only

            for current_agent in agents_with_no_potential_items:
                logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", current_agent,
                            alloc.remaining_item_capacities, alloc.bundles[current_agent])
                alloc.remove_agent_from_loop(current_agent)
                agents_who_need_an_item_in_current_iteration.remove(current_agent)

            #map_student_to_his_sum_bids += [alloc.effective_value(student,course) for student, course in map_student_to_his_sum_bids.items()]
            for student, course in map_student_to_his_sum_bids.items():
                if student in map_agent_to_best_item:
                    map_student_to_his_sum_bids[student] += alloc.effective_value(student ,course)

            #create dict for each course of student that point on the course and their bids
            map_course_to_students_with_max_bids = {course:
                {student: map_student_to_his_sum_bids[student] for student in map_agent_to_best_item if map_agent_to_best_item[student] == course}
                for course in alloc.remaining_items()}

            #loop on the dict_by_course and for each course give it to as many studens it can
            for course in map_course_to_students_with_max_bids:
                if course in alloc.remaining_item_capacities:


                    sorted_students_pointing_to_course = sorted(
                        [student for student in map_agent_to_best_item if map_agent_to_best_item[student] == course],
                        key=lambda student: alloc.effective_value(student, course),
                        reverse=True)  # sort the keys by their values (descending order)
                    remaining_capacity = alloc.remaining_item_capacities[course]
                    sorted_students_who_can_get_course = sorted_students_pointing_to_course[:remaining_capacity]
                    price = 0
                    if len(sorted_students_pointing_to_course) > remaining_capacity:
                        price = map_student_to_his_sum_bids[sorted_students_pointing_to_course[remaining_capacity]]  # the amount of bids of the first studen that cant get the course
                    for student in sorted_students_who_can_get_course:
                        alloc.give(student, course, logger)
                        agents_who_need_an_item_in_current_iteration.remove(student)  # removing the agent from the set (dont worry he will come back in the next round)
                        map_agent_to_best_item.pop(student, None)  # Delete student if exists in the dict
                        #del map_course_to_students_with_max_bids[course][s]
                        map_student_to_his_sum_bids[student] -= price


                    #if len(map_course_to_students_with_max_bids[course]) == 0:
                    #    continue
                    #if len(map_course_to_students_with_max_bids[course]) <= remaining_capacity:
                        #for student in map_course_to_students_with_max_bids[course]:
                        #    alloc.give(student,course,logger)
                        #    agents_who_need_an_item_in_current_iteration.remove(student)                                            # removing the agent from the set (dont worry he will come back in the next round)
                        #    if student in map_agent_to_best_item:                                  # removing the agent from the max dict
                        #        del map_agent_to_best_item[student]
                            #del dict_by_course[c][s]
                    # else:
                    #     sorted_students = sorted(map_course_to_students_with_max_bids[course].keys(), key=lambda name: map_course_to_students_with_max_bids[course][name], reverse=True)      #sort the keys by their values (descending order)
                    #     price = map_student_to_his_sum_bids[sorted_students[remaining_capacity]]                   #the amount of bids of the first studen that cant get the course
                    #     for i in range(remaining_capacity):
                    #         s = sorted_students[i]
                    #         alloc.give(s, course, logger)
                    #         set_agent.remove(s)                                            # removing the agent from the set (dont worry he will come back in the next round)
                    #         if s in map_agent_to_best_item:                                  # removing the agent from the max dict
                    #             del map_agent_to_best_item[s]                                 # removing the agent from the max dict
                    #         del map_course_to_students_with_max_bids[course][s]
                    #         map_student_to_his_sum_bids[s] -= price

        # set_agent = list(dict.fromkeys(alloc.remaining_agents()))           #with this random in loop and k passes
        #set_agent = list(dict.fromkeys(agent_order))                       #with this the random and k diffrent got ValueError

if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())