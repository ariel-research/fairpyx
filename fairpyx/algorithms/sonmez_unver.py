"""
"Course bidding at business schools", by Tayfun Sönmez and M. Utku Ünver (2010)
https://doi.org/10.1111/j.1468-2354.2009.00572.x

Allocate course seats using Gale-Shapley pareto-dominant market mechanism.

Programmer: Zachi Ben Shitrit
Since: 2024-05
"""

from fairpyx import AllocationBuilder
import numpy as np
from typing import Dict, List, Union

import logging
logger = logging.getLogger(__name__)


def gale_shapley(alloc: AllocationBuilder, course_order_per_student: Union[Dict[str, List[str]], None] = None, tie_braking_lottery: Union[None, Dict[str, float]] = None):
    """
    Allocate the given items to the given agents using the Gale-Shapley protocol.

    Parameters:
    alloc (AllocationBuilder): An allocation builder which tracks agent capacities, item capacities, and valuations.
    course_order_per_student (Dict[str, List[str]]): A dictionary that matches each agent to their course rankings indicating preferences.
    tie_braking_lottery (Dict[str, float]): A dictionary that matches each agent to their tie-breaking additive points (sampled from a uniform distribution [0,1]).

    Returns:
    Dict[str, List[str]]: A dictionary representing the final allocation of courses to students.

    Naive Example:
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

    
    Example where the students course order does not align with the bids:
    >>> s1 = {"c1": 20, "c2": 15, "c3": 35, "c4": 10, "c5": 20}
    >>> s2 = {"c1": 30, "c2": 15, "c3": 20, "c4": 20, "c5": 15}
    >>> s3 = {"c1": 40, "c2": 10, "c3": 25, "c4": 10, "c5": 15}
    >>> s4 = {"c1": 10, "c2": 10, "c3": 15, "c4": 30, "c5": 35}
    >>> s5 = {"c1": 25, "c2": 20, "c3": 30, "c4": 10, "c5": 15}
    >>> agent_capacities = {"Alice": 3, "Bob": 3, "Chana": 3, "Dana": 3, "Dor": 3}
    >>> course_capacities = {"c1": 4, "c2": 4, "c3": 2, "c4": 3, "c5": 2}
    >>> valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    >>> course_order_per_student = {"Alice": ["c5", "c3", "c1", "c2", "c4"], "Bob": ["c1", "c4", "c5", "c2", "c3"], "Chana": ["c5", "c1", "c4", "c3", "c2"], "Dana": ["c3", "c4", "c1", "c5", "c2"], "Dor": ["c5", "c1", "c4", "c3", "c2"]}
    >>> tie_braking_lottery = {"Alice": 0.6, "Bob": 0.4, "Chana": 0.3, "Dana": 0.8, "Dor": 0.2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(gale_shapley, instance=instance, course_order_per_student=course_order_per_student, tie_braking_lottery=tie_braking_lottery)
    {'Alice': ['c1', 'c3', 'c5'], 'Bob': ['c1', 'c2', 'c4'], 'Chana': ['c1', 'c2', 'c4'], 'Dana': ['c2', 'c4', 'c5'], 'Dor': ['c1', 'c2', 'c3']}
    """
    
    # Check if inputs are dictionaries
    input_to_check_types = [alloc.remaining_agent_capacities, alloc.remaining_item_capacities]
    for input_to_check in input_to_check_types:
        if(type(input_to_check) != dict):
            raise TypeError(f"In the input {input_to_check}, Expected a dict, but got {type(input_to_check).__name__}")
    if tie_braking_lottery and type(tie_braking_lottery) != dict:
        raise TypeError(f"In the input tie_braking_lottery, Expected a dict or None, but got {type(tie_braking_lottery).__name__}")
    if not tie_braking_lottery:
        tie_braking_lottery = {student : np.random.uniform(low=0, high=1) for student in alloc.remaining_agents()}
    
    if not course_order_per_student:
        course_order_per_student = {student : generate_naive_course_order_for_student(student, alloc) for student in alloc.remaining_agents()}
        logger.info(f"Created course_order_per_student: {course_order_per_student}")
    
    was_an_offer_declined: bool = True
    course_to_on_hold_students: Dict[str, Dict[str, float]] = {course: {} for course in alloc.remaining_items()} 
    student_to_rejection_count: Dict[str, int] = {student: alloc.remaining_agent_capacities[student] for student in alloc.remaining_agents()}
    
    logger.info(f"We have {len(alloc.remaining_agents())} agents")
    logger.info(f"The students allocation capacities are: {alloc.remaining_agent_capacities}")
    logger.info(f"The courses capacities are: {alloc.remaining_item_capacities}")
    logger.info(f"The tie-braking lottery results are: {tie_braking_lottery}")
    for agent in alloc.remaining_agents():
        agent_bids = {course: alloc.effective_value(agent, course) for course in alloc.remaining_items()}
        logger.info(f"Student '{agent}' bids are: {agent_bids}")
        
    step = 0
    while(was_an_offer_declined):
        step += 1
        logger.info(f"\n *** Starting round #{step} ***")
        was_an_offer_declined = False
        logger.info("Each student who is rejected from k > 0 courses in the previous step proposes to his best remaining k courses based on his stated preferences")
        for student in alloc.remaining_agents():
            student_capability: int = student_to_rejection_count[student]
            for index in range(student_capability):
                if(not course_order_per_student[student]):
                    logger.info(f"Student {student} already proposed to all his desired courses")
                    continue
                wanted_course = course_order_per_student[student].pop(0)
                if(wanted_course in course_to_on_hold_students):
                    if(student in course_to_on_hold_students[wanted_course]):
                        continue
                    try:
                        student_to_course_proposal = alloc.effective_value(student, wanted_course)
                        course_to_on_hold_students[wanted_course][student] = student_to_course_proposal
                        logger.info(f"Student '{student} proposes to course {wanted_course} with a bid of {student_to_course_proposal}")
                    except Exception as e:
                        return {}
        
        logger.info("Each course c considers the new proposals together with the proposals on hold and rejects all but the highest bidding Qc (the maximum capacity of students in course c) students")
        student_to_rejection_count = {student: 0 for student in alloc.remaining_agents()}
        for course_name in course_to_on_hold_students:
            course_capacity = alloc.remaining_item_capacities[course_name]
            if(type(course_capacity) == np.float64):
                course_capacity = int(course_capacity)
            course_to_offerings = course_to_on_hold_students[course_name]
            logger.info(f"Course {course_name} considers the next offerings: {course_to_offerings}")
            if len(course_to_offerings) == 0:
                continue
            elif len(course_to_offerings) <= course_capacity:
                continue
            logger.info("In case there is a tie, the tie-breaking lottery is used to determine who is rejected and who will be kept on hold.")
            on_hold_students_sorted_and_tie_breaked = sort_and_tie_break(course_to_offerings, tie_braking_lottery)
            course_to_on_hold_students[course_name].clear()
            for key, value in on_hold_students_sorted_and_tie_breaked[:course_capacity]:
                course_to_on_hold_students[course_name][key] = value
                
            rejected_students = on_hold_students_sorted_and_tie_breaked[course_capacity:]
            for rejected_student, bid in rejected_students:
                logger.info(f"Agent '{rejected_student}' was rejected from course {course_name}")
                student_to_rejection_count[rejected_student] += 1
            was_an_offer_declined = True
    
    logger.info("The procedure terminates when no proposal is rejected, and at this stage course assignments are finalized.")
    final_course_matchings = course_to_on_hold_students.items()
    for course_name, matching in final_course_matchings:
        for student, bid in matching.items():
            alloc.give(student, course_name, logger)
    logger.info(f"The final course matchings are: {alloc.bundles}")


def sort_and_tie_break(input_dict: Dict[str, float], tie_braking_lottery: Dict[str, float]) -> List[tuple[str, float]]:
    """
    Sorts a dictionary by its values in descending order and adds a number
    to the values of keys with the same value to break ties.

    Parameters:
    input_dict (Dict[str, float]): A dictionary with string keys and float values representing student bids.
    tie_braking_lottery (Dict[str, float]): A dictionary with string keys and float values for tie-breaking.

    Returns:
    List[tuple[str, float]]: A list of tuples containing student names and their modified bids, sorted in descending order.
    
    Examples:
    >>> input_dict = {"Alice": 45, "Bob": 55, "Chana": 45, "Dana": 60}
    >>> tie_braking_lottery = {"Alice": 0.3, "Bob": 0.2, "Chana": 0.4, "Dana": 0.1}
    >>> sort_and_tie_break(input_dict, tie_braking_lottery)
    [('Dana', 60), ('Bob', 55), ('Chana', 45), ('Alice', 45)]
    """

    
    # Sort the dictionary by adjusted values in descending order
    sorted_dict = (sorted(input_dict.items(), key=lambda item: item[1] + tie_braking_lottery[item[0]], reverse=True))
    
    return sorted_dict


def generate_naive_course_order_for_student(student: str, alloc: AllocationBuilder) -> List[str]:
    """
    Generate a naive course order for a given student based on the effective value the student assigns to each course.
    
    Parameters:
    student (str): The student's name.
    alloc (AllocationBuilder): An allocation builder which tracks agent capacities, item capacities, and valuations.

    Returns:
    List[str]: A list of course names sorted by the effective value the student assigns to each course, in descending order.
    
    Example:
    >>> from fairpyx import Instance, AllocationBuilder
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
    >>> alloc = AllocationBuilder(instance)
    >>> generate_naive_course_order_for_student("Alice", alloc)
    ['c2', 'c1']
    >>> generate_naive_course_order_for_student("Bob", alloc)
    ['c1', 'c2']
    >>> generate_naive_course_order_for_student('Chana', alloc)
    ['c1', 'c2']
    >>> generate_naive_course_order_for_student('Dana', alloc)
    ['c2', 'c1']
    >>> generate_naive_course_order_for_student('Dor', alloc)
    ['c1', 'c2']
    """
    # Get all courses
    courses: List[str] = alloc.remaining_items()
    
    # Calculate the effective value of each course for the given student
    course_values: Dict[str, float] = {course: alloc.effective_value(student, course) for course in courses}
    
    # Sort the courses by their values in descending order
    sorted_courses = sorted(course_values.items(), key=lambda item: item[1], reverse=True)
    
    # Extract the course names from the sorted list of tuples
    sorted_course_names = [course for course, value in sorted_courses]
    
    return sorted_course_names

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    from fairpyx import Instance, divide

    s1 = {"c1": 20, "c2": 15, "c3": 35, "c4": 10, "c5": 20}
    s2 = {"c1": 30, "c2": 15, "c3": 20, "c4": 20, "c5": 15}
    s3 = {"c1": 40, "c2": 10, "c3": 25, "c4": 10, "c5": 15}
    s4 = {"c1": 10, "c2": 10, "c3": 15, "c4": 30, "c5": 35}
    s5 = {"c1": 25, "c2": 20, "c3": 30, "c4": 10, "c5": 15}
    agent_capacities = {"Alice": 3, "Bob": 3, "Chana": 3, "Dana": 3, "Dor": 3}
    course_capacities = {"c1": 4, "c2": 4, "c3": 2, "c4": 3, "c5": 2}
    valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    course_order_per_student = {"Alice": ["c5", "c3", "c1", "c2", "c4"], "Bob": ["c1", "c4", "c5", "c2", "c3"], "Chana": ["c5", "c1", "c4", "c3", "c2"], "Dana": ["c3", "c4", "c1", "c5", "c2"], "Dor": ["c5", "c1", "c4", "c3", "c2"]}
    tie_braking_lottery = {"Alice": 0.6, "Bob": 0.4, "Chana": 0.3, "Dana": 0.8, "Dor": 0.2}
    instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    divide(gale_shapley, instance=instance, course_order_per_student=course_order_per_student, tie_braking_lottery=tie_braking_lottery)
    {'Alice': ['c1', 'c3', 'c5'], 'Bob': ['c1', 'c2', 'c4'], 'Chana': ['c1', 'c2', 'c4'], 'Dana': ['c2', 'c4', 'c5'], 'Dor': ['c1', 'c2', 'c3']}

