"""
"On Achieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging

logger = logging.getLogger(__name__)



def Demote(matching:dict, student_index:int, down_index:int, up_index:int)-> dict:
    """
    Demote algorithm: Adjust the matching by moving a student to a lower-ranked college
    while maintaining the invariant of a complete stable matching.
    The Demote algorithm is a helper function used within the FaSt algorithm to adjust the matching while maintaining stability.

    :param matching: the matchinf of the students with colleges.
    :param student_index: Index of the student to move.
    :param down_index: Index of the college to move the student to.
    :param up_index: Index of the upper bound college.

        # The test is the same as the running example we gave in Ex2.
#    ...   valuations       = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}},

    >>> matching = {1: [1, 6], 2: [2, 3],3: [4, 5]}
    >>> UP = 1
    >>> DOWN = 3
    >>> I = 2
    >>> Demote(matching, I, DOWN, UP)
    {1: [6], 2: [3, 1], 3: [4, 5, 2]}"""
    # Move student to college 'down' while reducing the number of students in 'up'
    # Set t to student_index
    t = student_index
    # Set p to 'down'
    p = down_index
    if t not in matching[p - 1]:
        raise ValueError(f"Student {t} should be in matching to college {p - 1}")
        # Check that all colleges have at least one students
    # for college, students in matching.items():
    #     if len(students) < 1:
    #         raise ValueError(f"All colleges must contain at least 1 student. College number {college} has only {len(students)} students.")

    # While p > up
    while p > up_index:
        # Remove student 't' from college 'cp-1'
        matching[p - 1].remove(t)
        # Add student 't' to college 'cp'
        matching[p].append(t)
        # Decrement t and p
        t -= 1
        p -= 1

    return matching #Return the matching after the change


def get_leximin_tuple(matching, V):
    """
    Generate the leximin tuple based on the given matching and evaluations,
    including the sum of valuations for each college.

    :param matching: The current matching dictionary
    :param V: The evaluations matrix
    :return: Leximin tuple
    """
    leximin_tuple = []
    college_sums = []

    # Iterate over each college in the matching
    for college, students in matching.items():
        college_sum = 0
        # For each student in the college, append their valuation for the college to the leximin tuple
        for student in students:
            valuation = V[student - 1][college - 1]
            leximin_tuple.append(valuation)
            college_sum += valuation
        college_sums.append(college_sum)
    # Append the college sums to the leximin tuple
    leximin_tuple.extend(college_sums)

    # Sort the leximin tuple in descending order
    leximin_tuple.sort(reverse=False)

    return leximin_tuple


def get_unsorted_leximin_tuple(matching, V):
    """
        Generate the leximin tuple based on the given matching and evaluations,
        including the sum of valuations for each college.

        :param matching: The current matching dictionary
        :param V: The evaluations matrix
        :return: UNSORTED Leximin tuple
        """
    leximin_tuple = []
    college_sums = []

    # Iterate over each college in the matching
    for college, students in matching.items():
        college_sum = 0
        # For each student in the college, append their valuation for the college to the leximin tuple
        for student in students:
            valuation = V[student - 1][college - 1]
            leximin_tuple.append(valuation)
            college_sum += valuation
        college_sums.append(college_sum)
    # Append the college sums to the leximin tuple
    leximin_tuple.extend(college_sums)
    return leximin_tuple


def build_pos_array(matching, V):
    """
    Build the pos array based on the leximin tuple and the matching.

    :param leximin_tuple: The leximin tuple
    :param matching: The current matching dictionary
    :param V: The evaluations matrix
    :return: Pos array
    """
    pos = []  # Initialize pos array
    student_index = 0
    college_index = 0
    leximin_unsorted_tuple = get_unsorted_leximin_tuple(matching, V)
    leximin_sorted_tuple = sorted(leximin_unsorted_tuple)
    while student_index < len(V):
        pos_value = leximin_sorted_tuple.index(leximin_unsorted_tuple[student_index])
        pos.append(pos_value)
        student_index += 1
    while college_index < len(matching):
        pos_value = leximin_sorted_tuple.index(leximin_unsorted_tuple[student_index + college_index])
        pos.append(pos_value)
        college_index += 1
    return pos


def create_L(matching):
    """
            Create the L list based on the matching.
        :param matching: The current matching
        :return: L list
        """
    L = []

    # Create a list of tuples (college, student)
    for college, students in matching.items():
        for student in students:
            L.append((college, student))

    return L


def build_college_values(matching, V):
    """
       Build the college_values dictionary that sums the students' valuations for each college.

       :param matching: The current matching dictionary
       :param V: The evaluations matrix
       :return: College values dictionary
       """
    college_values = {}

    # Iterate over each college in the matching
    for college, students in matching.items():
        college_sum = sum(V[student - 1][college - 1] for student in students)
        college_values[college] = college_sum

    return college_values


def initialize_matching(n, m):
    """
       Initialize the first stable matching.

       :param n: Number of students
       :param m: Number of colleges
       :return: Initial stable matching
       """
    initial_matching = {k: [] for k in range(1, m + 1)}  # Create a dictionary for the matching
    # Assign the first (n - m + 1) students to the first college (c1)
    for student in range(1, n - m + 2):
        initial_matching[1].append(student)
    # Assign each remaining student to the subsequent colleges (c2, c3, ...)
    for j in range(2, m + 1):
        initial_matching[j].append(n - m + j)
    return initial_matching

def convert_valuations_to_matrix(valuations):
    """
    Convert the dictionary of valuations to a matrix format.

    :param valuations: Dictionary of valuations
    :return: Matrix of valuations
    """
    students = sorted(valuations.keys())  # Sort student keys to maintain order
    colleges = sorted(valuations[students[0]].keys())  # Sort college keys to maintain order

    V = []
    for student in students:
        V.append([valuations[student][college] for college in colleges])

    return V

def FaSt(alloc: AllocationBuilder)-> dict:
    """
    FaSt algorithm: Find a leximin optimal stable matching under ranked isometric valuations.
    # def FaSt(instance: Instance, explanation_logger: ExplanationLogger = ExplanationLogger()):
    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    # The test is not the same as the running example we gave in Ex2.
    # We asked to change it to be with 3 courses and 7 students, like in algorithm 3 (FaSt-Gen algo).

    >>> from fairpyx.adaptors import divide
    >>> agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
    >>> items = {"c1", "c2", "c3"} #College set=C
    >>> valuation= {"S1": {"c1": 9, "c2": 8, "c3": 7},
    ... "S2": {"c1": 8, "c2": 7, "c3":6},
    ... "S3": {"c1": 7, "c2": 6, "c3":5},
    ... "S4": {"c1": 6, "c2": 5, "c3":4},
    ... "S5": {"c1": 5, "c2": 4, "c3":3},
    ... "S6": {"c1": 4, "c2": 3, "c3":2},
    ... "S7": {"c1": 3, "c2": 2, "c3":1}}# V[i][j] is the valuation of Si for matching with Cj
    >>> ins = Instance(agents=agents, items=items, valuations=valuation)
    >>> alloc = AllocationBuilder(instance=ins)
    >>> FaSt(alloc=alloc)
    {1: [1,2,3], 2: [5, 4], 3: [7, 6]}"""

    S = alloc.instance.agents
    C = alloc.instance.items
    V = alloc.instance._valuations
    # Now V look like this:
    # "Alice": {"c1":2, "c2": 3, "c3": 4},
    # "Bob": {"c1": 4, "c2": 5, "c3": 6}
    n=len(S)# number of students
    m = len(C)  # number of colleges
    i = n - 1  # start from the last student
    j = m - 1  # start from the last college
    # Initialize the first stable matching
    initial_matching = initialize_matching(n, m)
    # Convert Valuations to only numerical matrix
    V= convert_valuations_to_matrix(V)

    lex_tupl=get_leximin_tuple(initial_matching,V)

# Initialize the leximin tuple L and position array pos
    pos= build_pos_array(initial_matching, V)

    L=create_L(initial_matching)

    college_values=build_college_values(initial_matching,V)
    print("i: ", i)
    print("j: ", j)
    index = 1
    while i > j - 1 and j > 0:

        print("******** Iteration number ", index, "********")
        print("i: ", i)
        print("j: ", j)
        print("college_values[j+1]: ", college_values[j + 1])
        print("V[i-1][j]: ", V[i - 1][j])
        print("college_values: ", college_values)
        if college_values[j + 1] >= V[i - 1][j]:  ###need to update after each iteration
            j -= 1
        else:
            if college_values[j + 1] < V[i - 1][j]:
                print("V[i-1][j]:", V[i - 1][j])
                # if V[i][j - 1] > L[j - 1]:
                initial_matching = Demote(initial_matching, i, j + 1, 1)
                print("initial_matching after demote:", initial_matching)
            else:
                if V[i][j - 1] < college_values[j]:
                    j -= 1
                else:
                    # Lookahead
                    k = i
                    t = pos[i]
                    µ_prime = initial_matching.copy()
                    while k > j - 1:
                        if V[k][j - 1] > L[t - 1]:
                            i = k
                            initial_matching = Demote(µ_prime, k, j, 1)
                            break
                        elif V[k][j - 1] < college_values[j]:
                            j -= 1
                            break
                        else:
                            µ_prime = Demote(µ_prime, k, j, 1)
                            k -= 1
                            t += 1
                    if k == j - 1 and initial_matching != µ_prime:
                        j -= 1
        # Updates
        college_values = build_college_values(initial_matching, V)  # Update the college values
        lex_tupl = get_leximin_tuple(initial_matching, V)
        print("lex_tupl: ", lex_tupl)
        L = create_L(initial_matching)
        print("L:", L)  ################todo : update POS
        pos = build_pos_array(initial_matching, V)
        print("pos:", pos)

        i -= 1
        index += 1
        print("END while :")
        print("i: ", i)
        print("j: ", j)

    return initial_matching


if __name__ == "__main__":
    import doctest
    doctest.testmod()
