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
from fairpyx.algorithms.course_match.A_CEEI import (
    compute_surplus_demand_for_each_course,
    find_best_schedule,
    find_preference_order_for_each_student,
)
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

"""
Algorithm 3: The algorithm is designed to refill all the courses that, following Algorithm 2, have space in them.
"""


def reduce_undersubscription(allocation: AllocationBuilder, price_vector: dict, student_budgets: dict, priorities_student_list: list) -> AllocationBuilder:
    """
    Perform automated aftermarket allocations with increased budget and restricted allocations.

    :param allocation: (AllocationBuilder) current course allocations
    :param price_vector: (dict) price vector for courses
    :param student_list: List of students ordered by their class year descending and budget surplus ascending
    :param student_budgets: Budget for each student

    :return: Updated course allocations
    """
    item_conflicts, agent_conflicts = calculate_conflicts(allocation)
    preferred_schedule = find_preference_order_for_each_student(allocation.instance._valuations, allocation.instance._agent_capacities, item_conflicts, agent_conflicts)
    logger.debug('Preferred schedule calculated: %s', preferred_schedule)

    # Calculate the demand for each course based on the price vector and student budgets
    course_demands_dict = compute_surplus_demand_for_each_course(price_vector, allocation, student_budgets, preferred_schedule)  
    logger.debug('Course demands calculated: %s', course_demands_dict)

    # Identify undersubscribed courses (courses with negative demand)
    capacity_undersubscribed_courses = {course: -1 * course_demand for course, course_demand in course_demands_dict.items() if course_demand < 0}
    logger.debug('Undersubscribed courses identified: %s', capacity_undersubscribed_courses)

    student_schedule = find_best_schedule(price_vector, student_budgets, preferred_schedule)
    student_schedule_dict = create_dictionary_of_schedules(student_schedule, allocation.instance.items, allocation.instance.agents)
    logger.debug('Initial student schedules: %s', student_schedule_dict)

    student_list = calculate_remaining_budgets(price_vector, student_budgets, student_schedule_dict, priorities_student_list, allocation)
    logger.debug('Student list with remaining budgets: %s', student_list)

    # Reoptimize student schedules to fill undersubscribed courses
    student_schedule_dict = reoptimize_student_schedules(allocation, price_vector, student_list, student_budgets, student_schedule_dict, capacity_undersubscribed_courses)

    # Update the allocation with the new student schedules
    for student, schedule in student_schedule_dict.items():
        allocation.give_bundle(student, schedule)
        logger.info('Updated allocation for student %s: %s', student, schedule)

    return allocation


def calculate_conflicts(allocation: AllocationBuilder) -> tuple:
    """
    Calculate conflicts for items and agents.

    :param allocation: (AllocationBuilder)

    :return: Tuple containing item conflicts and agent conflicts

    >>> instance = Instance(
    ...     agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
    ...     item_capacities={"c1": 2, "c2": 2, "c3": 2},
    ...     valuations={
    ...         "Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...         "Tom": {"c1": 70, "c2": 30, "c3": 70},
    ...     },
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> calculate_conflicts(allocation)
    ({'c1': set(), 'c2': set(), 'c3': set()}, {'Alice': set(), 'Bob': set(), 'Tom': set()})
    """
    item_conflicts = {
        item: allocation.instance.item_conflicts(item)
        for item in allocation.instance.items
    }
    agent_conflicts = {
        agent: allocation.instance.agent_conflicts(agent)
        for agent in allocation.instance.agents
    }
    
    logger.debug('Calculated item conflicts: %s', item_conflicts)
    logger.debug('Calculated agent conflicts: %s', agent_conflicts)
    return item_conflicts, agent_conflicts


def create_dictionary_of_schedules(student_schedule, course, students) -> dict:
    """
    Create a dictionary of student schedules.

    :param student_schedule: (list of list) schedule of students
    :param course: (list) list of courses
    :param students: (list) list of students

    :return: (dict) dictionary of student schedules

    >>> student_schedule = [[1, 0, 1], [1, 1, 0], [1, 0, 1]]
    >>> course = ["c1", "c2", "c3"]
    >>> students = ["Alice", "Bob", "Tom"]
    >>> create_dictionary_of_schedules(student_schedule, course, students)
    {'Alice': ['c1', 'c3'], 'Bob': ['c1', 'c2'], 'Tom': ['c1', 'c3']}
    """
    schedule_dict = {student: [course for j, course in enumerate(course) if student_schedule[i][j] == 1] for i, student in enumerate(students)}
    logger.debug('Created dictionary of schedules: %s', schedule_dict)
    return schedule_dict


def calculate_remaining_budgets(price_vector: dict, student_budgets: dict, student_courses: dict, priorities_student_list: list, alloc: AllocationBuilder) -> list:
    """
    Calculate remaining budget for each student and sort based on priority and remaining budget.

    :param price_vector: (dict) price vector for courses
    :param student_budgets: (dict) budget for each student
    :param student_courses: (dict) courses allocated to each student
    :param priorities_student_list: (list of lists) Each list represents students with higher priority than the group after it

    :return: List of tuples containing student and their remaining budget

    >>> instance = Instance(
    ...     agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
    ...     item_capacities={"c1": 2, "c2": 2, "c3": 2},
    ...     valuations={
    ...         "Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...         "Tom": {"c1": 70, "c2": 30, "c3": 70},
    ...     },
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {"c1": 1.2, "c2": 0.7, "c3": 1.3}
    >>> student_budgets = {"Alice": 2.2, "Bob": 1.4, "Tom": 2.6}
    >>> student_courses = {'Alice': ['c1', 'c2'], 'Bob': ['c1'], 'Tom': ['c1', 'c3']}
    >>> priorities_student_list = [["Alice"], ["Bob", "Tom"]]
    >>> calculate_remaining_budgets(price_vector, student_budgets, student_courses, priorities_student_list, allocation)
    [('Alice', 0.30000000000000027), ('Tom', 0.10000000000000009), ('Bob', 0.19999999999999996)]
    """
    remaining_budgets = []
    for student, courses in student_courses.items():
        total_cost = sum(price_vector[course] for course in courses)
        remaining_budget = student_budgets[student] - total_cost
        remaining_budgets.append((student, remaining_budget))
        logger.debug('Student %s, Courses: %s, Total Cost: %f, Remaining Budget: %f', student, courses, total_cost, remaining_budget)
    
    # Create a priority dictionary to map each student to their priority group index
    if len(priorities_student_list) == 0:
        priorities_student_list = [[agent for agent in alloc.remaining_agents()]]
    priority_dict = {}
    for priority_index, group in enumerate(priorities_student_list):
        for student in group:
            priority_dict[student] = priority_index
    
    # Sort first by priority and then by remaining budget within each priority group
    remaining_budgets.sort(key=lambda x: (priority_dict[x[0]], x[1]))

    logger.debug('Calculated remaining budgets: %s', remaining_budgets)
    return remaining_budgets


def reoptimize_student_schedules(allocation, price_vector, student_list, student_budgets, student_schedule_dict, capacity_undersubscribed_courses) -> dict:
    """
    Reoptimize student schedules to fill undersubscribed courses.

    :param allocation: (AllocationBuilder)
    :param price_vector: (dict) price vector for courses
    :param student_list: (list) list of students with their remaining budgets
    :param student_budgets: (dict) budget for each student
    :param student_schedule_dict: (dict) current schedules of students
    :param capacity_undersubscribed_courses: (dict) courses that are undersubscribed

    :return: Updated student schedules
    """
    not_done = True
    while not_done and len(capacity_undersubscribed_courses) != 0:
        not_done = False
        for student in student_list:
            current_bundle = list(student_schedule_dict[student[0]])
            current_bundle.extend(x for x in list(capacity_undersubscribed_courses.keys()) if x not in current_bundle)
            current_bundle.sort()
            student_budget = {student[0]: 1.1 * student_budgets[student[0]]}
            new_bundle = allocation_function(allocation, student[0], current_bundle, price_vector, student_budget)
            if is_new_bundle_better(allocation, student[0], student_schedule_dict[student[0]], new_bundle.get(student[0], {})):
                not_done = True
                update_student_schedule_dict(student, student_schedule_dict, new_bundle, capacity_undersubscribed_courses)
                logger.debug('Updated student %s schedule: %s', student[0], student_schedule_dict[student[0]])
                break  # Only one student changes their allocation in each pass
    logger.info("Finished reoptimization of student schedules %s",student_schedule_dict)    
    return student_schedule_dict


def update_student_schedule_dict(student, student_schedule_dict, new_bundle, capacity_undersubscribed_courses) -> None:
    """
    Update student schedule dictionary and capacity of undersubscribed courses.

    :param student: (tuple) student and their remaining budget
    :param student_schedule_dict: (dict) current schedules of students
    :param new_bundle: (dict) new schedules of students
    :param capacity_undersubscribed_courses: (dict) courses that are undersubscribed
    """
    diff_in_bundle = list(set(new_bundle.get(student[0])).symmetric_difference(set(student_schedule_dict[student[0]])))
    for course in diff_in_bundle:
        if course in student_schedule_dict[student[0]]:
            capacity_undersubscribed_courses[course] = capacity_undersubscribed_courses.get(course, 0) + 1
        else:
            capacity_undersubscribed_courses[course] -= 1
            if capacity_undersubscribed_courses[course] == 0:
                capacity_undersubscribed_courses.pop(course)
    student_schedule_dict.update({student[0]: new_bundle.get(student[0])})
    logger.debug('Updated undersubscribed course capacities: %s', capacity_undersubscribed_courses)

def allocation_function(allocation: AllocationBuilder, student: str, student_allocation: dict, price_vector: dict, student_budget: dict) -> dict:
    """
    Function to reoptimize student's schedule.

    :param allocation: (AllocationBuilder) current course allocations
    :param student: (str) name of student
    :param student_allocation: (dict) Schedule of student to reoptimize
    :param price_vector: (dict) price vector for courses
    :param student_budget: (dict) New student's budget

    :return: (dict) new course allocations
    """
    limited_student_valuations = filter_valuations_for_courses(allocation, student, student_allocation)
    item_conflicts, agent_conflicts = calculate_conflicts(allocation)
    agent_capacities = {student: allocation.instance._agent_capacities[student]}
    preferred_schedule = find_preference_order_for_each_student(limited_student_valuations, agent_capacities, item_conflicts, agent_conflicts)
    limited_price_vector = {course: price for course, price in price_vector.items() if course in student_allocation}
    new_allocation = find_best_schedule(limited_price_vector, student_budget, preferred_schedule)
    new_allocation_dict = create_dictionary_of_schedules(new_allocation, student_allocation, agent_capacities.keys())
    logger.debug('Reoptimized schedule for student %s: %s', student, new_allocation_dict)
    return new_allocation_dict

def filter_valuations_for_courses(allocation, student, student_allocation) -> dict:
    """
    Filter valuations for the courses in the student's allocation.

    :param allocation: (AllocationBuilder)
    :param student: (str) name of student
    :param student_allocation: (dict) Schedule of student to reoptimize

    :return: (dict) filtered valuations for the courses in the student's allocation

    >>> instance = Instance(
    ...     agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
    ...     item_capacities={"c1": 2, "c2": 2, "c3": 2},
    ...     valuations={
    ...         "Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...         "Tom": {"c1": 70, "c2": 30, "c3": 70},
    ...     },
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> student_allocation = {"c1", "c3"}
    >>> filter_valuations_for_courses(allocation, "Alice", student_allocation)
    {'Alice': {'c1': 50, 'c3': 80}}
    """
    filtered_valuations = {
        student: {
            course: valuations
            for course, valuations in allocation.instance._valuations.get(
                student, {}
            ).items()
            if course in student_allocation
        }
    }
    logger.debug('Filtered valuations for student %s: %s', student, filtered_valuations)
    return filtered_valuations

def is_new_bundle_better(allocation: AllocationBuilder, student: str, current_bundle: set, new_bundle: set) -> bool:
    """
    Check if the current bundle and new bundle are equal.

    :param allocation: (AllocationBuilder)
    :param student: (str) name of student
    :param current_bundle: (set) current course bundle
    :param new_bundle: (set) new course bundle

    :return: (bool) True if bundles are equal, False otherwise
    >>> instance = Instance(
    ...     agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
    ...     item_capacities={"c1": 2, "c2": 2, "c3": 2},
    ...     valuations={
    ...         "Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...         "Tom": {"c1": 70, "c2": 30, "c3": 70},
    ...     },
    ... )
    >>> allocation = AllocationBuilder(instance)
    >>> is_new_bundle_better(allocation, "Alice", ["c1", "c3"], ["c1", "c3"])
    False
    >>> is_new_bundle_better(allocation, "Alice", ["c1", "c3"], ["c2", "c3"])
    False
    >>> is_new_bundle_better(allocation, "Alice", ["c1"], ["c2", "c3"])
    True
    >>> is_new_bundle_better(allocation, "Alice", ["c3"], ["c1", "c2"])
    False
    """
    sum_valuations_cur = sum(valuations for course, valuations in allocation.instance._valuations.get(student, {}).items() if course in current_bundle)
    sum_valuations_new = sum(valuations for course, valuations in allocation.instance._valuations.get(student, {}).items() if course in new_bundle)
    
    logger.debug('Current bundle valuations for student %s: %g', student, sum_valuations_cur)
    logger.debug('New bundle valuations for student %s: %g', student, sum_valuations_new)

    if (sum_valuations_cur < sum_valuations_new) or (len(current_bundle) < len(new_bundle) and sum_valuations_cur <= sum_valuations_new):
        logger.info('New bundle is better for student %s.', student)
        return True
    logger.info('New bundle is not better for student %s.', student)
    return False






if __name__ == "__main__":
    # pass
    import doctest
    print(doctest.testmod())

#     instance = Instance(
#        agent_conflicts = {"Alice": [], "Bob": []},
#        item_conflicts = {"c1": [], "c2": [], "c3": []},
#        agent_capacities = {"Alice": 2, "Bob": 1},
#        item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
#        valuations = {"Alice": {"c1": 100, "c2": 60, "c3": 0},
#                      "Bob": {"c1": 0, "c2": 100, "c3": 0},
#  })
#     allocation = AllocationBuilder(instance)
#     student_budgets = {"Alice": 3.0, "Bob": 1.0}  
#     price_vector = {"c1": 2.0, "c2": 1.0, "c3": 5.0}
#     print(
#         reduce_undersubscription(
#             allocation,
#             price_vector,
#             student_budgets,
#         ).bundles
    # )

    # {'Alice': ['c1', 'c2'], 'Bob': ['c2']}

    # instance = Instance(
    #     agent_capacities={"Alice": 2, "Bob": 2, "Tom": 2},
    #     item_capacities={"c1": 2, "c2": 2, "c3": 2},
    #     valuations={
    #         "Alice": {"c1": 50, "c2": 20, "c3": 80},
    #         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    #         "Tom": {"c1": 70, "c2": 30, "c3": 70},
    #     },
    # )
    # allocation = AllocationBuilder(instance)
    # price_vector = {"c1": 1.26875, "c2": 0.9, "c3": 1.24375}
    # student_budgets = {"Alice": 2.2, "Bob": 2.1, "Tom": 2.0}
    # print(
    #     reduce_undersubscription(
    #         allocation,
    #         price_vector,
    #         student_budgets,
    #     ).bundles
    # )