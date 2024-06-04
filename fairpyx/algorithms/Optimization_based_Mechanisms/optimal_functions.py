import cvxpy as cp

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
        if less_then == 1:
            constraints.append(cp.sum(var[:, i]) <= less_then)
        else:
            constraints.append(cp.sum(var[:, i]) <= less_then[student])


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

