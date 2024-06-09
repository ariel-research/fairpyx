"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation
Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Naama Shiponi and Ben Dabush
1/6/2024
"""

from fairpyx.algorithms.CM.A_CEEI import course_demands, find_best_schedule, find_preferred_schedule
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
"""
Algorithm 3: The algorithm is designed to refill all the courses that, following Algorithm 2, have space in them.
"""

def reduce_undersubscription(allocation: AllocationBuilder, price_vector: dict, student_list: list, alloction_function: callable, student_budgets):
    """
    Perform automated aftermarket allocations with increased budget and restricted allocations.

    :param alloction (AllocationBuilder) current course allocations
    :param price_vector
    :param instance: (Instance)
    :param student_list: List of students ordered by their class year descending and budget surplus ascending
    :param alloction_function: Function to reoptimize student's schedule given a set of courses and budget

    :return: Updated course allocations

    :pseudo code
    Input:  Input allocations x_ij = 1 if student i is taking course j,
            restricted demand functions x*_i (X_i , β_i ),
            S student sordered by class year descending and then by budget surplus ascending.
    Output: Altered allocations x_ij .
    1:  repeat
    2:      done ← true
    3:      u ← [Course j ∈ M: ∑_j x_ij < q_j]  # u is the set of currently undersubscribed courses
    4:      for Student i ∈ S do  # Iterate over the students in a ﬁxed order
    5:          x'_i ← x*_i (u U x_i , 1.1 · β_i )  # Reoptimize over a restricted set of courses with 10% more budget
    6:          if x_i diffrent x'_i then
    7:              done ← false
    8:              x_i ← x'_i
    9:          break  # Break out of the for loop, so that only one student changes his or her allocation in each pass
    10:         end if
    11:     end for
    12: until done  # Done only if we do a pass without any students changing their allocation

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1}, 
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                         "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> price_vector = [1.26875,0.9, 1.24375]
    >>> student_budgets = [2.2,2.1,2.0]
    >>> alloction = ({"Alice": [0,0,1]},
    ...                {"Bob": [1,0,0]},
    ...                {"Tom": [0,1,1]})
    >>> sort_student_list = [("Tom", 0.05625), ("Alice", 0.75625), ("Bob", 0.83125)]
    >>> reduce_undersubscription(alloction, price_vector, instance, sort_student_list, allocation_function)
    {{"Alice": [0,1,1]}, {"Bob": [0,0,1]}, {"Tom": [0,1,1]}}    
    """
    preferred_schedule = find_preferred_schedule(allocation)
    course_demands_dict = course_demands(price_vector, allocation, student_budgets, preferred_schedule)
    capacity_undersubscribed_courses = {course : -1*course_demand for course, course_demand in course_demands_dict.items() if course_demand < 0}
    student_schedule = find_best_schedule(price_vector, student_budgets, preferred_schedule)
    student_schedule_dict = {student: {} for student in allocation.instance.agents}
    student_schedule_dict = {student: {course for i, course in enumerate(allocation.instance.items) if student_schedule[i] == 1} for student in allocation.instance.agents}

    done = False
    while not done:
        done = True
        # Find undersubscribed courses
        # undersubscribed_courses = allocation.remaining_items()
        undersubscribed_courses = {course for course in capacity_undersubscribed_courses.keys()}
        
        
        for student in student_list:
            # Reoptimize over a restricted set of courses with 10% more budget
            current_bundle = student_schedule_dict[student]
            new_bundle = allocation_function(current_bundle.union(undersubscribed_courses), price_vector, 1.1 * student_budgets[student])

            if current_bundle != new_bundle:
                done = False
                # Calculate the difference between the current and new bundle
                diff_in_bundle = current_bundle.difference(new_bundle)

                for course in diff_in_bundle:
                    if course in student_schedule_dict[student]:
                        capacity_undersubscribed_courses[course] -= 1
                        if capacity_undersubscribed_courses[course] == 0:
                            capacity_undersubscribed_courses.pop(course)
                    else:
                        capacity_undersubscribed_courses[course] = 1
                student_schedule_dict[student] = new_bundle

                break  # Only one student changes their allocation in each pass
        
        for student, schedule in student_schedule_dict.items():
            allocation.give_bundle(student, schedule)
    return allocation

def allocation_function(student_allocation: tuple, price_vector: tuple, available_courses: tuple, student_budget: float, instance: Instance):   
    """
    function to reoptimize student's schedule.

    :praam instance
    :param student_allocation: Schedule of student to reoptimize
    :param price_vector (List of floats)
    :param available_courses: List of available courses for the student
    :param student_budget: New student's budget

    :return: List of new course allocations

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1, "c4": 1}, 
    ...   valuations       = {"Alice": {"c1": 90, "c2": 60, "c3": 50, "c4": 10},
    ... })
    >>> student_budget = 113
    >>> student_allocation = [1, 0, 0, 0]
    >>> price_vector = [103, 90, 72, 10]
    >>> available_courses = ["c4"]
    >>> allocation_function(student_allocation, price_vector, available_courses, student_budget, instance)
    [1, 0, 0, 1]

    # >>> instance = Instance(
    # ...   agent_capacities = {"Alice": 3}, 
    # ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1, "c4": 1}, 
    # ...   valuations       = {"Alice": {"c1": 90, "c2": 60, "c3": 50, "c4": 40},
    # ... })
    # >>> student_budget = 110
    # >>> student_allocation = [1, 0, 0, 0]
    # >>> price_vector = [103, 60, 72, 10]
    # >>> available_courses = ["c2","c4"]
    # >>> alloction_function(student_allocation, price_vector, available_courses, student_budget, instance)
    # [0, 1, 0, 1]

    # >>> instance = Instance(
    # ...   agent_capacities = {"Alice": 2}, 
    # ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1, "c4": 1}, 
    # ...   valuations       = {"Alice": {"c1": 90, "c2": 60, "c3": 50, "c4": 10},
    # ... })
    # >>> student_budget = 103
    # >>> student_allocation = [1, 0, 0, 0]
    # >>> price_vector = [103, 90, 72, 10]
    # >>> available_courses = ["c2"]
    # >>> alloction_function(student_allocation, price_vector, available_courses, student_budget, instance)
    # [1, 0, 0, 0]
    """
    # Implement a simple greedy algorithm to select courses within the budget
    course_indices = {course: idx for idx, course in enumerate(instance.valuations[student_allocation])}
    available_course_indices = [course_indices[course] for course in available_courses]

    new_allocation = student_allocation[:]
    total_cost = sum(new_allocation[i] * price_vector[i] for i in range(len(price_vector)))
    for course_idx in available_course_indices:
        if total_cost + price_vector[course_idx] <= student_budget:
            new_allocation[course_idx] = 1
            total_cost += price_vector[course_idx]

    return new_allocation


if __name__ == "__main__":
    import doctest
    doctest.testmod()
