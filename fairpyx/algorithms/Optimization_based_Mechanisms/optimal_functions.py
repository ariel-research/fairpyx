import cvxpy
import cvxpy as cp
import concurrent.futures

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


# allocation course to student according the result of the linear programing
def give_items_according_to_allocation_matrix(alloc, allocation_matrix, explanation_logger, rank_mat):
    # Extract the optimized values of x
    x_values = allocation_matrix.value
    explanation_logger.debug("x_values - the optimum allocation:\n%s", x_values)

    # Initialize a dictionary where each student will have an empty list
    assign_map_courses_to_student = {student: [] for student in alloc.remaining_agents()}

    remaining_agents_list = list(alloc.remaining_agents())
    remaining_items_list = list(alloc.remaining_items())

    # Iterate over students and courses to populate the lists
    for i, student in enumerate(remaining_agents_list):
        for j, course in enumerate(remaining_items_list):
            # logger.info("x[%d,%d]=%s", j, i, x_values[j, i])
            if x_values[j, i] > 0.5:
                explanation_logger.info("x_values[%d, You] = %s, so you get course %s for %d bids", j, x_values[j, i], course, alloc.effective_value(student, course), agents=student)
                assign_map_courses_to_student[student].append(course)

    # logger.info("assign_map_courses_to_student: %s", assign_map_courses_to_student)

    # Assign the courses to students based on the dictionary
    for student, courses in assign_map_courses_to_student.items():
        for course in courses:
            # change the rank mat
            # before the last place in a course in given, remove the course from the rank_mat
            # otherwise change the rank of this course to 0
            if alloc.remaining_item_capacities[course] == 1:
                for j, course2 in enumerate(alloc.remaining_items()):
                    if course == course2:
                        del rank_mat[j]
                        break
            else:
                for j, course2 in enumerate(alloc.remaining_items()):
                    if course == course2:
                        break
                for i, student2 in enumerate(alloc.remaining_agents()):
                    if student == student2:
                        rank_mat[j][i] = 0
            alloc.give(student, course)  # , logger

    explanation_logger.debug("after alloc Rank matrix:\n%s", rank_mat)
    return rank_mat


# creating the rank matrix for the linear programing ((6) (17) in the article) using for TTC-O, SP-O and OC
def createRankMat(alloc, logger):
    rank_mat = [[0 for _ in range(len(alloc.remaining_agents()))] for _ in range(len(alloc.remaining_items()))]

    # sort the course to each student by the bids (best bids = higher rank)
    for i, student in enumerate(alloc.remaining_agents()):
        map_courses_to_student = alloc.remaining_items_for_agent(student)
        sorted_courses = sorted(map_courses_to_student, key=lambda course: alloc.effective_value(student, course))

        # fill the mat
        for j, course in enumerate(alloc.remaining_items()):
            if course in sorted_courses:
                rank_mat[j][i] = sorted_courses.index(course) + 1

    logger.debug("Rank matrix:\n%s", rank_mat)

    return rank_mat

# sum the optimal rank to be sure the optimal bids agree with the optimal rank (6) (10) (17) (19)
def sumOnRankMat(alloc, rank_mat, var):
    return cp.sum([rank_mat[j][i] * var[j, i]
                   for j, course in enumerate(alloc.remaining_items())
                   for i, student in enumerate(alloc.remaining_agents())
                   if (student, course) not in alloc.remaining_conflicts])

