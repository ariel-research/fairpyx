"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation
Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Naama Shiponi and Ben Dabush
1/6/2024
"""

from fairpyx.algorithms.CM.A_CEEI import (
    course_demands,
    find_best_schedule,
    find_preferred_schedule,
)
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

"""
Algorithm 3: The algorithm is designed to refill all the courses that, following Algorithm 2, have space in them.
"""


def reduce_undersubscription(allocation: AllocationBuilder, price_vector: dict, student_budgets: dict) -> AllocationBuilder:
    """
    Perform automated aftermarket allocations with increased budget and restricted allocations.

    :param allocation: (AllocationBuilder) current course allocations
    :param price_vector: (dict) price vector for courses
    :param student_list: List of students ordered by their class year descending and budget surplus ascending
    :param student_budgets: Budget for each student

    :return: Updated course allocations
    """
    item_conflicts, agent_conflicts = calculate_conflicts(allocation)
    preferred_schedule = find_preferred_schedule(allocation.instance._valuations, allocation.instance._agent_capacities, item_conflicts, agent_conflicts)

    # Calculate the demand for each course based on the price vector and student budgets
    course_demands_dict = course_demands(price_vector, allocation, student_budgets, preferred_schedule)

    # Identify undersubscribed courses (courses with negative demand)
    capacity_undersubscribed_courses = {course: -1 * course_demand for course, course_demand in course_demands_dict.items() if course_demand < 0}

    student_schedule = find_best_schedule(price_vector, student_budgets, preferred_schedule)
    student_schedule_dict = create_dictionary_of_schedules(student_schedule, allocation.instance.items, allocation.instance.agents)
    student_list = calculate_remaining_budgets(price_vector, student_budgets, student_schedule_dict)

    # Reoptimize student schedules to fill undersubscribed courses
    student_schedule_dict = reoptimize_student_schedules(allocation, price_vector, student_list, student_budgets, student_schedule_dict, capacity_undersubscribed_courses)

    # Update the allocation with the new student schedules
    for student, schedule in student_schedule_dict.items():
        allocation.give_bundle(student, schedule)

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
    return {student: [course for j, course in enumerate(course) if student_schedule[i][j] == 1] for i, student in enumerate(students)}


def calculate_remaining_budgets(price_vector, student_budgets, student_courses) -> list:
    """
    Calculate remaining budget for each student.

    :param price_vector: (dict) price vector for courses
    :param student_budgets: (dict) budget for each student
    :param student_courses: (dict) courses allocated to each student

    :return: List of tuples containing student and their remaining budget

    >>> price_vector = {"c1": 1.2, "c2": 0.7, "c3": 1.3}
    >>> student_budgets = {"Alice": 2.2, "Bob": 1.4, "Tom": 2.6}
    >>> student_courses = {'Alice': ['c1', 'c2'], 'Bob': ['c1'], 'Tom': ['c1', 'c3']}
    >>> calculate_remaining_budgets(price_vector, student_budgets, student_courses)
    [('Tom', 0.10000000000000009), ('Bob', 0.19999999999999996), ('Alice', 0.30000000000000027)]
    """
    remaining_budgets = []
    for student, courses in student_courses.items():
        total_cost = sum(price_vector[course] for course in courses)
        remaining_budget = student_budgets[student] - total_cost
        remaining_budgets.append((student, remaining_budget))
    
    # Sort the list of tuples by remaining budget
    remaining_budgets.sort(key=lambda x: x[1])
    
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
                break  # Only one student changes their allocation in each pass
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

def allocation_function(allocation: AllocationBuilder, student: str, student_allocation: dict, price_vector: dict, student_budget: dict) -> dict:
    """
    Function to reoptimize student's schedule.

    :param allocation: (AllocationBuilder) current course allocations
    :param student: (str) name of student
    :param student_allocation: (dict) Schedule of student to reoptimize
    :param price_vector: (dict) price vector for courses
    :param student_budget: (dict) New student's budget

    :return: (dict) new course allocations

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
    >>> price_vector = {"c1": 1.26875, "c2": 0.9, "c3": 1.24375}
    >>> student_allocation = {"c1", "c3"}
    >>> student_budget = {"Alice": 2.6}
    >>> allocation_function(allocation, "Alice", student_allocation, price_vector, student_budget)
    {'Alice': ['c1', 'c3']}
    """
    limited_student_valuations = filter_valuations_for_courses(allocation, student, student_allocation)
    item_conflicts, agent_conflicts = calculate_conflicts(allocation)
    agent_capacities = {student: allocation.instance._agent_capacities[student]}
    preferred_schedule = find_preferred_schedule(limited_student_valuations, agent_capacities, item_conflicts, agent_conflicts)
    limited_price_vector = {course: price for course, price in price_vector.items() if course in student_allocation}
    new_allocation = find_best_schedule(limited_price_vector, student_budget, preferred_schedule)
    new_allocation_dict = create_dictionary_of_schedules(new_allocation, student_allocation, agent_capacities.keys())
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
    return {
        student: {
            course: valuations
            for course, valuations in allocation.instance._valuations.get(
                student, {}
            ).items()
            if course in student_allocation
        }
    }

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
    >>> is_the_bundle_equal(allocation, "Alice", ["c1", "c3"], ["c1", "c3"])
    True
    >>> is_the_bundle_equal(allocation, "Alice", ["c1", "c3"], ["c2", "c3"])
    False
    >>> is_the_bundle_equal(allocation, "Alice", ["c1"], ["c2", "c3"])
    False
    """
    sum_valuations_cur = sum(valuations for course, valuations in allocation.instance._valuations.get(student, {}).items() if course in current_bundle)
    sum_valuations_new = sum(valuations for course, valuations in allocation.instance._valuations.get(student, {}).items() if course in new_bundle)
    
    if (sum_valuations_cur < sum_valuations_new) or (len(current_bundle) < len(new_bundle) and sum_valuations_cur <= sum_valuations_new):
        return True
        
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