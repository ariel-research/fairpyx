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

def allocate_course_to_student(student_idx_range, alloc, allocation_matrix, logger, remaining_items_list,remaining_agents_list):
    x_values = allocation_matrix.value
    logger.info("Thread handling students in range: %s", student_idx_range)

    logger.info("Number of remaining agents: %d", len(remaining_agents_list))
    logger.info("Number of remaining items: %d", len(remaining_items_list))

    for i in student_idx_range:
        logger.info("i: %d, len(remaining_agents_list): %s",i, len(remaining_agents_list))

        if i >= len(remaining_agents_list):
            logger.error("Index out of range: %d", i)
            continue
        student = remaining_agents_list[i]
        for j, course in enumerate(remaining_items_list):
            if x_values[j, i] == 1:
                alloc.give(student, course, logger)

def give_items_according_to_allocation_matrix_threaded(alloc, allocation_matrix, logger, executor, num_threads=4):
    num_students = len(alloc.remaining_agents())
    logger.info("num_students = %d", num_students)
    chunk_size = (num_students + num_threads - 1) // num_threads  # Ceiling division
    logger.info("chunk_size = %d", chunk_size)

    remaining_agents_list = list(alloc.remaining_agents())
    remaining_items_list = list(alloc.remaining_items())

    futures = []
    for i in range(0, num_students, chunk_size):
        student_idx_range = range(i, min(i + chunk_size, num_students))
        futures.append(executor.submit(allocate_course_to_student, student_idx_range, alloc, allocation_matrix, logger, remaining_items_list,remaining_agents_list))

    # Wait for all futures to complete
    concurrent.futures.wait(futures)

    for future in concurrent.futures.as_completed(futures):
        future.result()  # Ensure any exceptions are raised

# allocation course to student according the result of the linear programing
def give_items_according_to_allocation_matrix(alloc, allocation_matrix, logger):
    # Extract the optimized values of x
    x_values = allocation_matrix.value
    logger.info("x_values - the optimum allocation:\n%s", x_values)

    # Initialize a dictionary where each student will have an empty list
    assign_map_courses_to_student = {student: [] for student in alloc.remaining_agents()}

    remaining_agents_list = list(alloc.remaining_agents())
    remaining_items_list = list(alloc.remaining_items())

    # Iterate over students and courses to populate the lists
    for i, student in enumerate(remaining_agents_list):
        for j, course in enumerate(remaining_items_list):
            if x_values[j, i] == 1:
                assign_map_courses_to_student[student].append(course)

    # Assign the courses to students based on the dictionary
    for student, courses in assign_map_courses_to_student.items():
        for course in courses:
            alloc.give(student, course, logger)

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
