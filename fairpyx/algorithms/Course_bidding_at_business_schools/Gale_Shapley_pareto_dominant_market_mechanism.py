"""
"Course bidding at business schools", by Tayfun Sönmez and M. Utku Ünver (2010)
https://doi.org/10.1111/j.1468-2354.2009.00572.x

Allocate course seats using Gale-Shapley pareto-dominant market mechanism.

Programmer: Zachi Ben Shitrit
Since: 2024-05
"""

from fairpyx import AllocationBuilder
import numpy as np
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

def sort_and_tie_brake(input_dict: Dict[str, float], tie_braking_lottery: Dict[str, float], course_capacity: int) -> List[tuple[str, float]]:
    """
    Sorts a dictionary by its values in descending order and adds a number
    to the values of keys with the same value to break ties.
    Stops if the count surpasses the course's capacity and is not in a tie.

    Parameters:
    input_dict (Dict[str, float]): A dictionary with string keys and float values representing student bids.
    tie_braking_lottery (Dict[str, float]): A dictionary with string keys and float values for tie-breaking.
    course_capacity (int): The number of students allowed in the course.

    Returns:
    List[tuple[str, float]]: A list of tuples containing student names and their modified bids, sorted in descending order.
    """
    # Sort the dictionary by values in descending order
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Initialize previous value to track duplicate values
    previous_value = None
    prev_key = ""
    
    # Initialize a variable to track count
    count: int = 0
    
    # Iterate over the sorted dictionary and modify values
    for key in sorted_dict:
        current_value = sorted_dict[key]
        
        if current_value == previous_value:
            # If current value is the same as previous, add the number to both current and previous values
            sorted_dict[key] += tie_braking_lottery[key]
            sorted_dict[prev_key] += tie_braking_lottery[prev_key]
        elif count >= course_capacity:
            break
            
        # Update previous_value and prev_key to current_value and key for next iteration
        previous_value = sorted_dict[key]
        prev_key = key
    
    # Sort again after tie-breaking
    sorted_dict = (sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_dict


def gale_shapley(alloc: AllocationBuilder, course_order_per_student: Dict[str, List[str]], tie_braking_lottery: Dict[str, float]):
    """
    Allocate the given items to the given agents using the Gale-Shapley protocol.

    Parameters:
    alloc (AllocationBuilder): An allocation builder which tracks agent capacities, item capacities, and valuations.
    course_order_per_student (Dict[str, List[str]]): A dictionary that matches each agent to their course rankings indicating preferences.
    tie_braking_lottery (Dict[str, float]): A dictionary that matches each agent to their tie-breaking additive points (sampled from a uniform distribution [0,1]).

    Returns:
    Dict[str, List[str]]: A dictionary representing the final allocation of courses to students.

    Example:
    >>> from fairpyx import Instance, AllocationBuilder
    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 40, "c2": 60}
    >>> s2 = {"c1": 70, "c2": 30}
    >>> s3 = {"c1": 70, "c2": 30}
    >>> s4 = {"c1": 40, "c2": 60}
    >>> s5 = {"c1": 50, "c2": 50}
    >>> agent_capacities = {"Alice": 1, "Bob": 1, "Chana": 1, "Dana": 1, "Dor": 1}
    >>> course_capacities = {"c1": 3, "c2": 2}
    >>> valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    >>> course_order_per_student = {"Alice": ["c2", "c1"], "Bob": ["c1", "c2"], "Chana": ["c1", "c2"], "Dana": ["c2", "c1"], "Dor": ["c1", "c2"]}
    >>> tie_braking_lottery = {"Alice": 0.9, "Bob": 0.1, "Chana": 0.2, "Dana": 0.6, "Dor": 0.4}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(gale_shapley, instance=instance, course_order_per_student=course_order_per_student, tie_braking_lottery=tie_braking_lottery)
    {'Alice': ['c2'], 'Bob': ['c1'], 'Chana': ['c1'], 'Dana': ['c2'], 'Dor': ['c1']}
    """
    
    # Check if inputs are dictionaries
    input_to_check_types = [alloc.remaining_agent_capacities, alloc.remaining_item_capacities, course_order_per_student, tie_braking_lottery]
    for input_to_check in input_to_check_types:
        if(type(input_to_check) != dict):
            raise TypeError;
    
    if(alloc.remaining_agent_capacities == {} or alloc.remaining_item_capacities == {}):
        return {}
    was_an_offer_declined: bool = True
    course_to_on_hold_students: Dict[str, Dict[str, float]] = {course: {} for course in alloc.remaining_items()} 
    student_to_rejection_count: Dict[str, int] = {student: alloc.remaining_agent_capacities[student] for student in alloc.remaining_agents()}
    
    while(was_an_offer_declined):
        logger.info("Each student who is rejected from k > 0 courses in the previous step proposes to his best remaining k courses based on his stated preferences")
        for student in alloc.remaining_agents():
            student_capability: int = student_to_rejection_count[student]
            for index in range(student_capability):
                wanted_course = course_order_per_student[student].pop(index)
                if(wanted_course in course_to_on_hold_students):
                    if(student in course_to_on_hold_students[wanted_course]):
                        continue
                    try:
                        student_to_course_proposal = alloc.effective_value(student, wanted_course)
                        course_to_on_hold_students[wanted_course][student] = student_to_course_proposal
                    except Exception as e:
                        return {}
                    
        logger.info("Each course c considers the new proposals together with the proposals on hold and rejects all but the highest bidding Qc (the maximum capacity of students in course c) students")
        student_to_rejection_count = {student: 0 for student in alloc.remaining_agents()}
        for course_name in course_to_on_hold_students:
            course_capacity = alloc.remaining_item_capacities[course_name]
            course_to_offerings = course_to_on_hold_students[course_name]
            if len(course_to_offerings) == 0:
                continue
            elif len(course_to_offerings) <= course_capacity:
                was_an_offer_declined = False
                continue
            logger.info("In case there is a tie, the tie-breaking lottery is used to determine who is rejected and who will be kept on hold.")
            on_hold_students_sorted_and_tie_braked = sort_and_tie_brake(course_to_offerings, tie_braking_lottery, course_capacity)
            course_to_on_hold_students[course_name].clear()
            for key, value in on_hold_students_sorted_and_tie_braked[:course_capacity]:
                course_to_on_hold_students[course_name][key] = value
                
            rejected_students = on_hold_students_sorted_and_tie_braked[course_capacity:]
            for rejected_student, bid in rejected_students:
                student_to_rejection_count[rejected_student] += 1
            was_an_offer_declined = True
    
    logger.info("The procedure terminates when no proposal is rejected, and at this stage course assignments are finalized.")
    final_course_matchings = course_to_on_hold_students.items()
    for course_name, matching in final_course_matchings:
        for student, bid in matching.items():
            alloc.give(student, course_name, logger)

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
