"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""

import cvxpy

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import cvxpy as cp
logger = logging.getLogger(__name__)

def OC_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 5: Allocate the given items to the given agents using the OC protocol.

    in the OC algorithm for CAP, we maximize ordinal utility followed by maximizing cardinal utility among rank-maximal
    solutions, performing this two-part optimization once for the whole market.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 44, "c2": 39, "c3": 17}
    >>> s2 = {"c1": 50, "c2": 45, "c3": 5}
    >>> agent_capacities = {"s1": 2, "s2": 2}                                 # 4 seats required
    >>> course_capacities = {"c1": 2, "c2": 1, "c3": 2}                       # 5 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(OC_function, instance=instance)
    {'s1': ['c1', 'c3'], 's2': ['c1', 'c2']}
    """

    explanation_logger.info("\nAlgorithm OC starts.\n")

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)

    rank_mat = [[0 for _ in range(len(alloc.remaining_agents()))] for _ in range(len(alloc.remaining_items()))]

    for j, student in enumerate(alloc.remaining_agents()):
        map_courses_to_student = alloc.remaining_items_for_agent(student)
        sorted_courses = sorted(map_courses_to_student, key=lambda course: alloc.effective_value(student, course), )

        for i, course in enumerate(alloc.remaining_items()):
            if course in sorted_courses:
                rank_mat[i][j] = sorted_courses.index(course) + 1
            else:
                rank_mat[i][j] = len(sorted_courses) + 1

    objective_Z1 = cp.Maximize(cp.sum([rank_mat[i][j] * x[i, j]
                                        for i, course in enumerate(alloc.remaining_items())
                                        for j, student in enumerate(alloc.remaining_agents())
                                        if (student, course) not in alloc.remaining_conflicts]))

    constraints_Z1 = []
    # condition number 2:
    # check that the number of students who took course j does not exceed the capacity of the course
    for i, course in enumerate(alloc.remaining_items()):
        constraints_Z1.append(cp.sum(x[i, :]) <= alloc.remaining_item_capacities[course])

        # TODO: check if necessary
        # if the course have conflict
        for j, student in enumerate(alloc.remaining_agents()):
            if (student, course) in alloc.remaining_conflicts:
                constraints_Z1.append(x[i, j] == 0)

    # condition number 3:
    # Each student can take at most k courses
    for j, student in enumerate(alloc.remaining_agents()):
        constraints_Z1.append(cp.sum(x[:, j]) <= alloc.remaining_agent_capacities[student])

    # condition number 4:
    # TODO: if necessary because alloc check it

    problem = cp.Problem(objective_Z1, constraints=constraints_Z1)
    result_Z1 = problem.solve()

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)  # Is there a func which zero all the matrix?

    objective_Z2 = cp.Maximize(cp.sum([alloc.effective_value(student, course) * x[i, j]
                                        for i, course in enumerate(alloc.remaining_items())
                                        for j, student in enumerate(alloc.remaining_agents())
                                        if (student, course) not in alloc.remaining_conflicts]))

    # condition number 19:
    constraints_Z2 = []
    constraints_Z2.append(cp.sum([rank_mat[i][j] * x[i, j]
                                   for i, course in enumerate(alloc.remaining_items())
                                   for j, student in enumerate(alloc.remaining_agents())
                                   if (student, course) not in alloc.remaining_conflicts
                                   ]) == result_Z1)

    # condition number 2:
    # check that the number of students who took course j does not exceed the capacity of the course
    for i, course in enumerate(alloc.remaining_items()):
        constraints_Z2.append(cp.sum(x[i, :]) <= alloc.remaining_item_capacities[course])

        # TODO: check if necessary
        # if the course have conflict
        for j, student in enumerate(alloc.remaining_agents()):
            if (student, course) in alloc.remaining_conflicts:
                constraints_Z2.append(x[i, j] == 0)

    # condition number 3:
    # Each student can take at most k courses
    for j, student in enumerate(alloc.remaining_agents()):
        constraints_Z2.append(cp.sum(x[:, j]) <= alloc.remaining_agent_capacities[student])

    problem = cp.Problem(objective_Z2, constraints=constraints_Z2)
    result_Z2 = problem.solve()

    # Check if the optimization problem was successfully solved
    if result_Z2 is not None:
        # Extract the optimized values of x
        x_values = x.value

        # Initialize a dictionary where each student will have an empty list
        assign_map_courses_to_student = {student: [] for student in alloc.remaining_agents()}

        # Iterate over students and courses to populate the lists
        for j, student in enumerate(alloc.remaining_agents()):
            for i, course in enumerate(alloc.remaining_items()):
                if x_values[i, j] == 1:
                    assign_map_courses_to_student[student].append(course)

        # Assign the courses to students based on the dictionary
        for student, courses in assign_map_courses_to_student.items():
            for course in courses:
                alloc.give(student, course)

        optimal_value = problem.value
        explanation_logger.info("Optimal Objective Value:", optimal_value)
        # Now you can use this optimal value for further processing
    else:
        explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")


if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())

