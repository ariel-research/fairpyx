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

def TTC_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 3: Allocate the given items to the given agents using the TTC-O protocol.

    TTC-O assigns one course in each round to each student, the winning students are defined based on
     the students’ bid values. Uses linear planning for optimality.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).


    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    >>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    >>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    >>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(TTC_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}
    """
    explanation_logger.info("\nAlgorithm TTC-O starts.\n")

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    for iteration in range(max_iterations):
        x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)  # mat Xij

        i, j = 0, 0
        sums = 0
        for course in alloc.remaining_items():
            for student in alloc.remaining_agents():
                sums = x[i, j] * alloc.effective_value(student, course)
                j += 1
            j = 0
            i += 1

        obj = cp.Maximize(sums)

        constraints = []

        # condition number 7:
        # check that the number of students who took course j does not exceed the capacity of the course
        i = 0
        the_number_of_student_fit_the_course = 0
        for course in alloc.remaining_items():
            for j in range(len(alloc.remaining_agents())):
                the_number_of_student_fit_the_course += x[i, j]
            constraints.append(the_number_of_student_fit_the_course <= alloc.remaining_item_capacities[course])
            i += 1
            the_number_of_student_fit_the_course = 0

        # condition number 8:
        # check that each student receives only one course in each iteration
        the_number_of_courses_a_student_got = 0
        for i in range(len(alloc.remaining_agents())):
            for j in range(len(alloc.remaining_items())):
                the_number_of_courses_a_student_got += x[i, j]
            constraints.append(the_number_of_courses_a_student_got <= 1)
            the_number_of_courses_a_student_got = 0  # reset for the next column

        # map_agent_to_best_item = {}
        # for student in alloc.remaining_agents():
        #     map_agent_to_best_item[student] = max(alloc.remaining_items_for_agent(student),
        #                                           key=lambda item: alloc.effective_value(student, item))
        # obj = cp.Maximize(sum(alloc.effective_value(student, map_agent_to_best_item[student]) for student in map_agent_to_best_item))
        #
        # remaining_item_capacities_in_prev_iteretion = {}
        # for student in alloc.remaining_agents():
        #         remaining_item_capacities_in_prev_iteretion[student] = alloc.remaining_agent_capacities[student]




        # # condition number 7:
        # # check that the number of students who took course j does not exceed the capacity of the course
        # for course in alloc.remaining_items():
        #     count = 0
        #     for student in map_agent_to_best_item:
        #         if map_agent_to_best_item[student] == course:
        #             count += 1
        #     constraints.append(count <= alloc.remaining_item_capacities[course])
        #
        # # condition number 8:
        # # check that each student receives only one course in each iteration
        # for student in alloc.remaining_agents():
        #     constraints.append(remaining_item_capacities_in_prev_iteretion[student] - alloc.remaining_agent_capacities[student] <= 1)
        #
        # # condition number 9:
        # # we don't need

        problem = cp.Problem(obj, constraints=constraints)
        result = problem.solve()

        if result is not None:
            optimal_value = problem.value
            print("Optimal Objective Value:", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            print("Solver failed to find a solution or the problem is infeasible/unbounded.")



if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())