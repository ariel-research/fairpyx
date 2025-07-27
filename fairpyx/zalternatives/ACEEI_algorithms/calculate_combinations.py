from itertools import combinations
from fairpyx import Instance


def get_combinations_courses_sorted(instance: Instance):
    """
    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":4, "z":2}, "Bob":{"x":4, "y":3, "z":2}, "Eve":{"x":2, "y":4, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":2, "y":1, "z":3})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'Bob': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Eve': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}


    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":3, "w":2}, "Bob":{"x":5, "y":2, "z":4, "w":3}},
    ...     agent_capacities=3,
    ...     item_capacities={"x":1, "y":2, "z":1, "w":2})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'y', 'z'), ('x', 'y', 'w'), ('x', 'z', 'w'), ('y', 'z', 'w'), ('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('x',), ('z', 'w'), ('y',), ('z',), ('w',)], 'Bob': [('x', 'z', 'w'), ('x', 'y', 'z'), ('x', 'y', 'w'), ('y', 'z', 'w'), ('x', 'z'), ('x', 'w'), ('x', 'y'), ('z', 'w'), ('y', 'z'), ('x',), ('y', 'w'), ('z',), ('w',), ('y',)]}

    >>> instance = Instance(
    ...     valuations={"Alice":{"x":3, "y":3, "z":3, "w":3}, "Bob":{"x":3, "y":3, "z":3, "w":3}, "Eve":{"x":4, "y":4, "z":4, "w":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2, "w":1})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'Bob': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)], 'Eve': [('x', 'y'), ('x', 'z'), ('x', 'w'), ('y', 'z'), ('y', 'w'), ('z', 'w'), ('x',), ('y',), ('z',), ('w',)]}

    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":5, "z":1}, "Bob":{"x":4, "y":6, "z":4}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":2, "z":2})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Bob': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)]}

    >>> instance = Instance(
    ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'Bob': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)]}

    >>> instance = Instance(
    ...     valuations={"Alice":{"x":1, "y":1, "z":3}},
    ...     agent_capacities=2,
    ...     item_capacities={"x":1, "y":1, "z":2})
    >>> get_combinations_courses_sorted(instance)
    {'Alice': [('x', 'z'), ('y', 'z'), ('x', 'y'), ('z',), ('x',), ('y',)]}

    >>> instance = Instance(
    ...     valuations={"avi":{"x":5}, "beni":{"x":5}},
    ...     agent_capacities=1,
    ...     item_capacities={"x":1})
    >>> get_combinations_courses_sorted(instance)
    {'avi': [('x',)], 'beni': [('x',)]}

    >>> instance = Instance(
    ... valuations={"ami":{"x":3, "y":4, "z":2}, "tami":{"x":4, "y":3, "z":2}, "tzumi":{"x":2, "y":4, "z":3}},
    ... agent_capacities=2,
    ... item_capacities={"x":2, "y":1, "z":3})
    >>> get_combinations_courses_sorted(instance)
    {'ami': [('x', 'y'), ('y', 'z'), ('x', 'z'), ('y',), ('x',), ('z',)], 'tami': [('x', 'y'), ('x', 'z'), ('y', 'z'), ('x',), ('y',), ('z',)], 'tzumi': [('y', 'z'), ('x', 'y'), ('x', 'z'), ('y',), ('z',), ('x',)]}


    """
    combinations_courses = {student: [] for student in instance.agents}
    for student in instance.agents:
        combinations_for_student = []
        # Creating a list of combinations of courses up to the size of the student's capacity
        capacity = instance.agent_capacity(student)
        for r in range(1, capacity + 1):
            combinations_for_student.extend(combinations(instance.items, r))

        #  We would like to meet the requirement of the number of courses a student needs, therefore if
        #  the current combination meets the requirement we will give it more weight
        large_num = instance.agent_maximum_value(student)

        # Define a lambda function that calculates the valuation of a combination
        valuation_function = lambda combination: instance.agent_bundle_value(student, combination) + (
            large_num if len(combination) == instance.agent_capacity(student) else 0)

        # Sort the combinations_set based on their valuations in descending order
        combinations_for_student_sorted = sorted(combinations_for_student, key=valuation_function, reverse=True)
        combinations_courses[student] = combinations_for_student_sorted

    return combinations_courses


if __name__ == "__main__":
    import doctest, sys
    print("\n", doctest.testmod(), "\n")
