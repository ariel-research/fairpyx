"""
"On Achieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging

logger = logging.getLogger(__name__)



def Demote(matching, student_index, down, up):
    """
    Perform the demote operation to maintain stability.

    :param matching: The current matching dictionary
    :param student_index: The student being demoted
    :param down: The current college of the student
    :param up: The new college for the student
    """
    # Move student to college 'down' while reducing the number of students in 'up'
    # Set t to student_index
    t = student_index
    # Set p to 'down'
    p = down
    # Check if the student 't' is in college 'Cp-1'
    print( "Now demote student", t)
    print ("t ",t)
    print( "p " , p)
    print ("matching[p - 1]: ", matching[p - 1])
    if t not in matching[p - 1]:
        raise ValueError(f"Student {t} should be in matching to college {p - 1}")
    # Check that all colleges have at least one student
    for college, students in matching.items():
        if len(students) < 1:
            raise ValueError(f"All colleges must contain at least 1 student. College number {college} has only {len(students)} students.")

    # While p > up
    while p > up:
        print ("while loop")
        # Remove student 't' from college 'cp-1'
        matching[p - 1].remove(t)
        # Add student 't' to college 'cp'
        matching[p].append(t)
        # Decrement t and p
        t -= 1
        p -= 1
        print (matching)

    return matching

def FaSt(alloc: AllocationBuilder)-> dict:
    """
    FaSt algorithm: Find a leximin optimal stable matching under ranked isometric valuations.
    # def FaSt(instance: Instance, explanation_logger: ExplanationLogger = ExplanationLogger()):
    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    # The test is not the same as the running example we gave in Ex2.
    # We asked to change it to be with 3 courses and 7 students, like in algorithm 3 (FaSt-Gen algo).

    >>> from fairpyx.adaptors import divide
    >>> S = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set
    >>> C = {"c1", "c2", "c3"} #College set
    >>> V = {
    ...     "s1": {"c1": 1, "c3": 2, "c2": 3},
    ...     "s2": {"c2": 1, "c1": 2, "c3": 3},
    ...     "s3": {"c1": 1, "c3": 2, "c2": 3},
    ...     "s4": {"c3": 1, "c2": 2, "c1": 3},
    ...     "s5": {"c2": 1, "c3": 2, "c1": 3},
    ...     "s6": {"c3": 1, "c1": 2, "c2": 3},
    ...     "s7": {"c1": 1, "c2": 2, "c3": 3}
    ... } # V[i][j] is the valuation of Si for matching with Cj
    >>> instance = Instance(agents=S, items=C, _valuations=V)
    >>> divide(FaSt, instance=instance)
    {'s1': ['c1'], 's2': ['c2'], 's3': ['c1'], 's4': ['c3'], 's5': ['c3'], 's6': ['c3'], 's7': ['c2']}
    """
    S = alloc.instance.agents
    C = alloc.instance.items
    V = alloc.instance._valuations
    # Now V look like this:
    # "Alice": {"c1":2, "c2": 3, "c3": 4},
    # "Bob": {"c1": 4, "c2": 5, "c3": 6}

    # Initialize a stable matching
    matching = initialize_stable_matching(S, C, V)
    # Compute the initial leximin value and position array
    leximin_value, pos = compute_leximin_value(matching, V)

    # Iterate to find leximin optimal stable matching
    for i in range(len(S) - 1, -1, -1):
        for j in range(len(C) - 1, 0, -1):
            # Check if moving student i to college j improves the leximin value
            if can_improve_leximin(S[i], C[j], V, leximin_value):
                # If it does improve - perform the demote operation to maintain stability
                Demote(matching, S[i], C[j-1], C[j])
                # Recompute the leximin value and position array after the demotion
                leximin_value, pos = compute_leximin_value(matching, V)

    # Return the final stable matching
    return matching


def can_improve_leximin(student, college, V, leximin_value)-> bool:
    """
    Check if moving the student to the college improves the leximin value.

    :param student: The student being considered for reassignment
    :param college: The college being considered for the student's reassignment
    :param V: Valuation matrix where V[i][j] is the valuation of student i for college j
    :param leximin_value: The current leximin value
    :return: True if the new leximin value is an improvement, otherwise False
    """
    # Get the current value of the student for the new college
    current_value = V[student - 1][college - 1]  # assuming students and colleges are 1-indexed
    # Create a copy of the current leximin values
    new_values = leximin_value[:]
    # Remove the current value of the student in their current college from the leximin values
    new_values.remove(current_value)
    # Add the current value of the student for the new college to the leximin values
    new_values.append(current_value)
    # Sort the new leximin values to form the new leximin tuple
    new_values.sort()
    # Return True if the new leximin tuple is lexicographically greater than the current leximin tuple
    return new_values > leximin_value


def update_leximin_value(matching, V)-> list:
    # Update the leximin value after demotion
    values = []
    for college, students in matching.items():
        for student in students:
            student_index = student - 1  # assuming students are 1-indexed
            college_index = college - 1  # assuming colleges are 1-indexed
            values.append(V[student_index][college_index])
    values.sort()
    return values


def compute_leximin_value(matching, V)-> tuple:
    """
        Compute the leximin value of the current matching.

        This function calculates the leximin value of the current matching by evaluating the
        valuations of students for their assigned colleges. The leximin value is the sorted
        list of these valuations. It also returns the position array that tracks the positions
        of the valuations in the sorted list.

        :param matching: A dictionary representing the current matching where each college is a key and the value is a list of assigned students
        :param V: Valuation matrix where V[i][j] is the valuation of student i for college j
        :return: A tuple (values, pos) where values is the sorted list of valuations (leximin value) and pos is the position array
        """

    values = []# Initialize an empty list to store the valuations
    for college, students in matching.items():# Iterate over each college and its assigned students in the matching
        for student in students:# Iterate over each student assigned to the current college
            student_index = student - 1  # assuming students are 1-indexed
            college_index = college - 1  # assuming colleges are 1-indexed
            # Append the student's valuation for the current college to the values list
            values.append(V[student_index][college_index])
    # Sort the valuations in non-decreasing order to form the leximin tuple
    values.sort()
    pos = [0] * len(values)# Initialize the position array to track the positions of the valuations
    # Populate the position array with the index of each valuation
    for idx, value in enumerate(values):
        pos[idx] = idx
    # Return the sorted leximin values and the position array
    return values, pos


def initialize_stable_matching(S, C, V)-> dict:
    """
       Initialize a student optimal stable matching.
       This function creates an initial stable matching by assigning students to colleges based on
       their preferences. The first n - m + 1 students are assigned to the highest-ranked college,
       and the remaining students are assigned to the other colleges in sequence.

       :param S: List of students
       :param C: List of colleges
       :param V: Valuation matrix where V[i][j] is the valuation of student i for college j
       :return: A dictionary representing the initial stable matching where each college is a key and the value is a list of assigned students
       """
    # Get the number of students and colleges
    n = len(S)
    m = len(C)
    # Create an empty matching dictionary where each college has an empty list of assigned students
    matching = {c: [] for c in C}

    # Assign the first n - m + 1 students to the highest ranked college (C1)
    for i in range(n - m + 1):
        matching[C[0]].append(S[i])

    # Assign the remaining students to the other colleges in sequence
    for j in range(1, m):
        matching[C[j]].append(S[n - m + j])

    return matching# Return the initialized stable matching


if __name__ == "__main__":
    import doctest
    doctest.testmod()
