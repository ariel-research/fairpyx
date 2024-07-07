from itertools import combinations

from fairpyx import Instance


def get_combinations_courses_sorted(instance: Instance, student: str):
    # Creating a list of combinations of courses up to the size of the student's capacity
    combinations_courses_list = []
    capacity = instance.agent_capacity(student)
    for r in range(1, capacity + 1):
        combinations_courses_list.extend(combinations(instance.items, r))
    # logger.info(f"FINISH combinations for {student}")

    #  We would like to meet the requirement of the number of courses a student needs, therefore if
    #  the current combination meets the requirement we will give it more weight
    large_num = instance.agent_maximum_value(student)

    # Define a lambda function that calculates the valuation of a combination
    valuation_function = lambda combination: instance.agent_bundle_value(student, combination) + (
        large_num if len(combination) == instance.agent_capacity(student) else 0)

    # Sort the combinations_set based on their valuations in descending order
    combinations_courses_sorted = sorted(combinations_courses_list, key=valuation_function, reverse=True)

    return combinations_courses_sorted