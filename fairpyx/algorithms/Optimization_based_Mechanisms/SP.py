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

    agent_order = []                                                    # list of all the instances of agents as many as their items capacities (from the article: sum of Ki)
    map_agent_to_best_item = {}                                           # dict of the max bids for each agent in specific iteration (from the article: max{i, b})
    map_course_to_students_with_max_bids = {c: {} for c in alloc.remaining_item_capacities}

    for agent, capacity in alloc.remaining_agent_capacities.items():    # save the agent number of his needed courses (שומר את הסטודנט מספר פעמים כמספר הקורסים שהוא צריך)
        agent_order.extend([agent] * capacity)

    set_agent = list(dict.fromkeys(agent_order))                        # set of the agents by thier order . each agent appears only once

    dict_extra_bids = {s: 0 for s in set_agent}

    for agent in agent_order:
        if agent not in alloc.remaining_agent_capacities:  # to check if not all the agents got k courses
            continue

        while set_agent:  # round i
            for current_agent in set_agent:  # check the course with the max bids for each agent in the set (who didnt get a course in this round)
                if len(alloc.remaining_agent_capacities) == 0 or len(
                        alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
                    set_agent = []
                    break

                potential_items_for_agent = set(alloc.remaining_items()).difference(alloc.bundles[current_agent])  # set of all the courses that have places sub the courses the agent got (under the assumption that if agent doesnt want a course the course got 0 bids automaticlly)
                if len(potential_items_for_agent) == 0:
                    logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", current_agent,
                                alloc.remaining_item_capacities, alloc.bundles[current_agent])
                    if current_agent in agent_order and current_agent in alloc.remaining_agent_capacities:  # checking if the agent in dict before removing
                        alloc.remove_agent(current_agent)
                        agent_order.remove(current_agent)
                    continue

                map_agent_to_best_item[current_agent] = max(potential_items_for_agent,
                                                          key=lambda item: alloc.effective_value(current_agent,
                                                                                                 item))  # for each agent save the course with the max bids from the potential courses only
            # update bids for all student in dict_extra_bids
            for s in dict_extra_bids:
                if s in map_agent_to_best_item:
                    dict_extra_bids[s] += alloc.effective_value(s ,map_agent_to_best_item[s])

            #create dict for each course of student that point on the course and their bids
            map_course_to_students_with_max_bids = {course:
                {student: dict_extra_bids[student] for student in map_agent_to_best_item if map_agent_to_best_item[student] == course}
                for course in alloc.remaining_items()
            }
            # for c in map_course_to_students_with_bids:
            #     map_course_to_students_with_bids[c] = {s: dict_extra_bids[s] for s in map_agent_to_best_item if map_agent_to_best_item[s] == c}
            # for s in map_agent_to_best_item:
            #     if map_agent_to_best_item[s] == c:
            #         dict_s_to_c[s] = dict_extra_bids[s]
            # dict_by_course[c] = dict_s_to_c

            #loop on the dict_by_course and for each curse give it to as many studens it can
            for course in map_course_to_students_with_max_bids:
                if course in alloc.remaining_item_capacities:
                    c_capacity = alloc.remaining_item_capacities[course]
                    if len(map_course_to_students_with_max_bids[course]) == 0:
                        continue
                    elif len(map_course_to_students_with_max_bids[course]) <= c_capacity:
                        for s in map_course_to_students_with_max_bids[course]:
                            alloc.give(s,course,logger)
                            set_agent.remove(s)                                            # removing the agent from the set (dont worry he will come back in the next round)
                            if s in map_agent_to_best_item:                                  # removing the agent from the max dict
                                del map_agent_to_best_item[s]
                            #del dict_by_course[c][s]
                    else:
                        sorted_students = sorted(map_course_to_students_with_max_bids[course].keys(), key=lambda name: map_course_to_students_with_max_bids[course][name], reverse=True)      #sort the keys by their values (descending order)
                        price = dict_extra_bids[sorted_students[c_capacity]]                   #the amount of bids of the first studen that cant get the course
                        for i in range(c_capacity):
                            s = sorted_students[i]
                            alloc.give(s, course, logger)
                            set_agent.remove(s)                                            # removing the agent from the set (dont worry he will come back in the next round)
                            if s in map_agent_to_best_item:                                  # removing the agent from the max dict
                                del map_agent_to_best_item[s]                                 # removing the agent from the max dict
                            del map_course_to_students_with_max_bids[course][s]
                            dict_extra_bids[s] -= price

        set_agent = list(dict.fromkeys(alloc.remaining_agents()))           #with this random in loop and k passes
        #set_agent = list(dict.fromkeys(agent_order))                       #with this the random and k diffrent got ValueError

if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())