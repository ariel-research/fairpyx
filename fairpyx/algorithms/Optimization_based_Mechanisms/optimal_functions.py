import cvxpy
import cvxpy as cp

# check that the number of students who took course j does not exceed the capacity of the course
def notExceedtheCapacity(var, alloc):
    constraints = []
    for j, course in enumerate(alloc.remaining_items()):
        constraints.append(cp.sum(var[j, :]) <= alloc.remaining_item_capacities[course])

        for i, student in enumerate(alloc.remaining_agents()):
            if (student, course) in alloc.remaining_conflicts:
                constraints.append(var[j, i] == 0)
    return constraints


# check that each student receives only one course in each iteration
def numberOfCourses(var, alloc, less_then):
    constraints = []
    for i, student in enumerate(alloc.remaining_agents()):
        if less_then == 1:
            constraints.append(cp.sum(var[:, i]) <= less_then)
        else:
            constraints.append(cp.sum(var[:, i]) <= less_then[student])
    return constraints



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
            alloc.give(student, course, logger)


def createRankMat(alloc, logger):
    rank_mat = [[0 for _ in range(len(alloc.remaining_agents()))] for _ in range(len(alloc.remaining_items()))]

    for i, student in enumerate(alloc.remaining_agents()):
        map_courses_to_student = alloc.remaining_items_for_agent(student)
        sorted_courses = sorted(map_courses_to_student, key=lambda course: alloc.effective_value(student, course), )

        for j, course in enumerate(alloc.remaining_items()):
            if course in sorted_courses:
                rank_mat[j][i] = sorted_courses.index(course) + 1

    logger.info("Rank matrix: %s", rank_mat)
    return rank_mat

def SP_O_condition(alloc, result_Zt1, D, p, v):
    return result_Zt1 * D + cp.sum([alloc.remaining_item_capacities[course] * p[j] for j, course in enumerate(alloc.remaining_items())]) + cp.sum([v[i] for i, student in enumerate(alloc.remaining_agents())])

def conditions_14(alloc, v,p):
    constraints = []
    for j, course in enumerate(alloc.remaining_items()):
        for i, student in enumerate(alloc.remaining_agents()):
            constraints.append(p[j] >= 0)
            constraints.append(v[i] >= 0)
    return constraints

def sumOnRankMat(alloc, rank_mat, var):
    return cp.sum([rank_mat[j][i] * var[j, i]
                for j, course in enumerate(alloc.remaining_items())
                for i, student in enumerate(alloc.remaining_agents())
                if (student, course) not in alloc.remaining_conflicts])

#  if flag_if_use_alloc_in_func == 0 then using alloc.effective_value
#  if flag_if_use_alloc_in_func == 1 then using effective_value_with_price
def roundTTC_O(alloc, logger, func, flag_if_use_alloc_in_func):
    rank_mat = createRankMat(alloc, logger)

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)

    sum_rank = sumOnRankMat(alloc, rank_mat, x)

    objective_Zt1 = cp.Maximize(sum_rank)

    constraints_Zt1 = notExceedtheCapacity(x, alloc) + numberOfCourses(x, alloc, 1)

    problem = cp.Problem(objective_Zt1, constraints=constraints_Zt1)
    result_Zt1 = problem.solve()  # This is the optimal value of program (6)(7)(8)(9).
    logger.info("result_Zt1 - the optimum ranking: %d", result_Zt1)

    # Write and solve new program for Zt2 (10)(11)(7)(8)
    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())),
                       boolean=True)  # Is there a func which zero all the matrix?

    objective_Zt2 = cp.Maximize(cp.sum(
        [func(student, course) * x[j, i] if flag_if_use_alloc_in_func == 0 else func(alloc, student, course) * x[j, i]
         for j, course in enumerate(alloc.remaining_items())
         for i, student in enumerate(alloc.remaining_agents())
         if (student, course) not in alloc.remaining_conflicts]
    ))

    constraints_Zt2 = notExceedtheCapacity(x, alloc) + numberOfCourses(x, alloc, 1)

    constraints_Zt2.append(sum_rank == result_Zt1)

    try:
        problem = cp.Problem(objective_Zt2, constraints=constraints_Zt2)
        result_Zt2 = problem.solve()
        logger.info("result_Zt2 - the optimum bids: %d", result_Zt2)

    except Exception as e:
        logger.info("Solver failed: %s", str(e))
        logger.error("An error occurred: %s", str(e))
        raise

    return result_Zt1, result_Zt2, x, problem, rank_mat
