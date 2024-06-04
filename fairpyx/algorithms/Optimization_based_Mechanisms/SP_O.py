"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""
import cvxpy
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import cvxpy as cp
import logging
logger = logging.getLogger(__name__)

# check that the number of students who took course j does not exceed the capacity of the course
def notExceedtheCapacity(constraints, var, alloc):
    for j, course in enumerate(alloc.remaining_items()):
        constraints.append(cp.sum(var[j, :]) <= alloc.remaining_item_capacities[course])

        for i, student in enumerate(alloc.remaining_agents()):
            if (student, course) in alloc.remaining_conflicts:
                constraints.append(var[j, i] == 0)


# check that each student receives only one course in each iteration
def numberOfCourses(constraints, var, alloc, less_then):
    for i, student in enumerate(alloc.remaining_agents()):
        constraints.append(cp.sum(var[:, i]) <= less_then)

def alloctions(alloc, var, logger):
    # Extract the optimized values of x
    x_values = var.value
    logger.info("x_values - the optimum allocation: %s", x_values)

    # Initialize a dictionary where each student will have an empty list
    assign_map_courses_to_student = {student: [] for student in alloc.remaining_agents()}

    # Iterate over students and courses to populate the lists
    for i, student in enumerate(alloc.remaining_agents()):
        for j, course in enumerate(alloc.remaining_items()):
            if x_values[j, i] == 1:
                assign_map_courses_to_student[student].append(course)

    # Assign the courses to students based on the dictionary
    for student, courses in assign_map_courses_to_student.items():
        for course in courses:
            alloc.give(student, course)

def SP_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 4: Allocate the given items to the given agents using the SP-O protocol.

    SP-O in each round distributes one course to each student, with the refund of the bids according to the price of
    the course. Uses linear planning for optimality.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).


    #>>> from fairpyx.adaptors import divide
    #>>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    #>>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    #>>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    #>>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    #>>> valuations = {"s1": s1, "s2": s2}
    #>>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    #>>> divide(SP_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}
    """

    explanation_logger.info("\nAlgorithm SP-O starts.\n")

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in
                         alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    logger.info("Max iterations: %d", max_iterations)
    for iteration in range(max_iterations):
        logger.info("Iteration number: %d", iteration+1)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            logger.info("There are no more agents (%d) or items(%d) ", len(alloc.remaining_agent_capacities),len(alloc.remaining_item_capacities))
            break

        rank_mat = [[0 for _ in range(len(alloc.remaining_agents()))] for _ in range(len(alloc.remaining_items()))]

        for i, student in enumerate(alloc.remaining_agents()):
            map_courses_to_student = alloc.remaining_items_for_agent(student)
            sorted_courses = sorted(map_courses_to_student, key=lambda course: alloc.effective_value(student, course))

            for j, course in enumerate(alloc.remaining_items()):
                if course in sorted_courses:
                    rank_mat[j][i] = sorted_courses.index(course) + 1
                else:
                    rank_mat[j][i] = len(sorted_courses) + 1

        logger.info("Rank matrix: %s", rank_mat)

        x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)

        objective_Zt1 = cp.Maximize(cp.sum([rank_mat[j][i] * x[j, i]
                                            for j, course in enumerate(alloc.remaining_items())
                                            for i, student in enumerate(alloc.remaining_agents())
                                            if (student, course) not in alloc.remaining_conflicts]))

        constraints_Zt1 = []

        # condition number 7:
        notExceedtheCapacity(constraints_Zt1, x, alloc)

        # condition number 8:
        numberOfCourses(constraints_Zt1, x, alloc, 1)

        problem = cp.Problem(objective_Zt1, constraints=constraints_Zt1)
        result_Zt1 = problem.solve()  # This is the optimal value of program (6)(7)(8)(9).

        logger.info("result_Zt1 - the optimum ranking: %d", result_Zt1)

        # condition number 12:
        D = cvxpy.Variable()
        p = cvxpy.Variable(len(alloc.remaining_items()))
        v = cvxpy.Variable(len(alloc.remaining_agents()))
        objective_Wt1 = cp.Minimize(result_Zt1*D
                                   + cp.sum(alloc.remaining_item_capacities[course]*p[j] for j, course in enumerate(alloc.remaining_items()))
                                   + cp.sum(v[i] for i, student in enumerate(alloc.remaining_agents())))

        constraints_Wt1 = []

        # condition number 13 + 14:
        for j, course in enumerate(alloc.remaining_items()):
            for i, student in enumerate(alloc.remaining_agents()):
                constraints_Wt1.append(p[j]+v[i]+rank_mat[i][j]*D >= alloc.effective_value(student,course))
                constraints_Wt1.append(p[j] >= 0)
                constraints_Wt1.append(v[i] >= 0)

        problem = cp.Problem(objective_Wt1, constraints=constraints_Wt1)
        result_Wt1 = problem.solve()  # This is the optimal value of program (12)(13)(14).

        objective_Wt2 = cp.Minimize(cp.sum(alloc.remaining_item_capacities[course]* p[j]
                                           for j, course in enumerate(alloc.remaining_items())))

        constraints_Wt2 = []


        constraints_Wt2.append(result_Zt1*D
                               + cp.sum(alloc.remaining_item_capacities[course]* p[j]
                                   for j, course in enumerate(alloc.remaining_items()))
                               +cp.sum(v[i] for i, student in enumerate(alloc.remaining_agents())) == result_Wt1)

        for j in range(len(alloc.remaining_items())):
            for i in range(len(alloc.remaining_agents())):
                constraints_Wt2.append(p[j] >= 0)
                constraints_Wt2.append(v[i] >= 0)

        problem = cp.Problem(objective_Wt2, constraints=constraints_Wt2)
        result_Wt2 = problem.solve()  # This is the optimal price

        logger.info("result_Wt2 - the optimum price: %d", result_Wt2)

        # Check if the optimization problem was successfully solved
        if result_Zt1 is not None:
            alloctions(alloc, x, logger)

            optimal_value = problem.value
            explanation_logger.info("Optimal Objective Value:", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")


if __name__ == "__main__":
    #import doctest, sys
    #print(doctest.testmod())

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    from fairpyx.adaptors import divide
    s1 = {"c1": 40, "c2": 10, "c3": 20, "c4": 30}
    s2 = {"c1": 50, "c2": 10, "c3": 15, "c4": 25}
    s3 = {"c1": 60, "c2": 30, "c3": 2, "c4": 8}
    instance = Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2},
        item_capacities={"c1": 1, "c2": 2, "c3": 2, "c4": 1},
        valuations={"s1": s1, "s2": s2, "s3": s3}
    )
    divide(SP_O_function, instance=instance)