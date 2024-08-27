"""
"On Achieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
"""

import random
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import copy # For deep copy use

#Object to insert the relevant data 
logger = logging.getLogger("data")


def Demote(matching:dict, student_index:int, down_index:int, up_index:int)-> dict:
    """
    Demote algorithm: Adjust the matching by moving a student to a lower-ranked college
    while maintaining the invariant of a complete stable matching.
    The Demote algorithm is a helper function used within the FaSt algorithm to adjust the matching while maintaining stability.

    :param matching: the matchinf of the students with colleges.
    :param student_index: Index of the student to move.
    :param down_index: Index of the college to move the student to.
    :param up_index: Index of the upper bound college.

        #*** The test is the same as the running example we gave in Ex2.***
#    ...   valuations       = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}},

    >>> matching = {1: [1, 6], 2: [2, 3],3: [4, 5]}
    >>> UP = 1
    >>> DOWN = 3
    >>> I = 2
    >>> Demote(matching, I, DOWN, UP)
    {1: [6], 2: [3, 1], 3: [4, 5, 2]}"""
    # Check if matching is empty
    if not matching:
        raise ValueError("Demote algorithm failed on empty input")

    # Move student to college 'down' while reducing the number of students in 'up'
    # Set t to student_index
    t = student_index
    # Set p to 'down'
    p = down_index
    logger.info("matching: %s",matching)
    # Check if student 't' is in college 'Cp-1'
    if t not in matching[p - 1]:
        raise ValueError(f"Student {t} should be in matching to college {p - 1}")
    # Check that all colleges have at least one students
    for college, students in matching.items():
        if len(students) < 1:
            raise ValueError(f"All colleges must contain at least 1 student. College number {college} has only {len(students)} students.")

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
    >>> matching = {1: [1, 2, 3, 4], 2: [5], 3: [7, 6]}
    >>> V = [[], [0, 9, 8, 7], [0, 8, 7, 6], [0, 7, 6, 5], [0, 6, 5, 4], [0, 5, 4, 3], [0, 4, 3, 2], [0, 3, 2, 1]]
    >>> get_leximin_tuple(matching, V)
    [1, 2, 3, 4, 4, 6, 7, 8, 9, 30]
    """

    leximin_tuple=get_unsorted_leximin_tuple(matching, V)
    # Sort the leximin tuple in descending order
    leximin_tuple.sort(reverse=False)

    return leximin_tuple

def get_unsorted_leximin_tuple(matching, V):
    """
        Generate the leximin tuple based on the given matching and evaluations,
        including the sum of valuations for each college.
        Using in calculate pos array.

        :param matching: The current matching dictionary
        :param V: The evaluations matrix
        :return: UNSORTED Leximin tuple
    >>> matching = {1: [1, 2, 3, 4], 2: [5], 3: [7, 6]}
    >>> V = [[], [0, 9, 8, 7], [0, 8, 7, 6], [0, 7, 6, 5], [0, 6, 5, 4], [0, 5, 4, 3], [0, 4, 3, 2], [0, 3, 2, 1]]
    >>> get_unsorted_leximin_tuple(matching, V)
    [9, 8, 7, 6, 4, 1, 2, 30, 4, 3]
        """
    leximin_tuple = []
    college_sums = []

    # Iterate over each college in the matching
    for college, students in matching.items():
        college_sum = 0
        # For each student in the college, append their valuation for the college to the leximin tuple
        for student in students:
            valuation = V[student][college]
            leximin_tuple.append(valuation)
            college_sum += valuation
        college_sums.append(college_sum)
    # Append the college sums to the leximin tuple
    leximin_tuple.extend(college_sums)
    return leximin_tuple

def build_pos_array(matching, V):
    """
    Build the pos array based on the leximin tuple and the matching.
    For example:
    Unsorted Leximin Tuple: [**9**, 8, 7, 6, 5, 3, 1, 35, 3, 1] -> V of S1 is 9
    Sorted Leximin Tuple:   [1, 1, 3, 3, 5, 6, 7, 8, 9,35]-> 9 is in index 8
    pos=[8,7, 6,5,4,3,1,9,2,0]

    :param leximin_tuple: The leximin tuple
    :param matching: The current matching dictionary
    :param V: The evaluations matrix
    :return: Pos array
    >>> matching = {1: [1, 2, 3, 4], 2: [5], 3: [7, 6]}
    >>> V = [[], [0, 9, 8, 7], [0, 8, 7, 6], [0, 7, 6, 5], [0, 6, 5, 4], [0, 5, 4, 3], [0, 4, 3, 2], [0, 3, 2, 1]]
    >>> build_pos_array(matching, V)
    [8, 7, 6, 5, 3, 0, 1, 9, 3, 2]
    """
    pos = []  # Initialize pos array
    student_index = 0
    college_index = 0
    # Get the unsorted leximin tuple
    leximin_unsorted_tuple = get_unsorted_leximin_tuple(matching, V)
    # Get the sorted leximin tuple
    leximin_sorted_tuple = sorted(leximin_unsorted_tuple)
    # Build pos array for students
    while student_index < len(V)-1:# -1 because i added an element to V
        pos_value = leximin_sorted_tuple.index(leximin_unsorted_tuple[student_index])
        pos.append(pos_value)
        student_index += 1
    # Build pos array for colleges
    while college_index < len(matching):
        pos_value = leximin_sorted_tuple.index(leximin_unsorted_tuple[student_index + college_index])
        pos.append(pos_value)
        college_index += 1
    return pos

def build_college_values(matching, V):
    """
       Build the college_values dictionary that sums the students' valuations for each college.
        For example:
        c1: [9, 8, 7, 6, 5] =35
        c2: [3] =3
        c3: [1] =1
       :param matching: The current matching dictionary
       :param V: The evaluations matrix
       :return: College values dictionary
       >>> matching = {1: [1, 2, 3, 4], 2: [5], 3: [7, 6]}
       >>> V = [[], [0, 9, 8, 7], [0, 8, 7, 6], [0, 7, 6, 5], [0, 6, 5, 4], [0, 5, 4, 3], [0, 4, 3, 2], [0, 3, 2, 1]]
       >>> build_college_values(matching, V)
       {0: 0, 1: 30, 2: 4, 3: 3}
       """
    college_values = {0 : 0}

    # Iterate over each college in the matching
    for college, students in matching.items():
        college_sum = sum(V[student][college] for student in students)
        college_values[college] = college_sum

    return college_values

def initialize_matching(n, m):
    """
    Initialize the first stable matching.

    :param n: Number of students
    :param m: Number of colleges
    :return: Initial stable matching

    >>> n = 7
    >>> m = 3
    >>> initialize_matching(n, m)
    {1: [1, 2, 3, 4, 5], 2: [6], 3: [7]}"""
    initial_matching = {k: [] for k in range(1, m + 1)}  # Create a dictionary for the matching
    # Assign the first (n - m + 1) students to the first college (c1)
    for student in range(1, n - m + 2):
        initial_matching[1].append(student)
    # Assign each remaining student to the subsequent colleges (c2, c3, ...)
    for j in range(2, m + 1):# 2 because we started from C2
        initial_matching[j].append(n - m + j)
    return initial_matching

def convert_valuations_to_matrix(valuations):
    """
    Convert the dictionary of valuations to a matrix format.
    To be the same as in the algo .

    :param valuations: Dictionary of valuations
    :return: Matrix of valuations
    >>> valuations={'S1': {'c1': 9, 'c2': 8, 'c3': 7}, 'S2': {'c1': 8, 'c2': 7, 'c3': 6}, 'S3': {'c1': 7, 'c2': 6, 'c3': 5}, 'S4': {'c1': 6, 'c2': 5, 'c3': 4}, 'S5': {'c1': 5, 'c2': 4, 'c3': 3}, 'S6': {'c1': 4, 'c2': 3, 'c3': 2}, 'S7': {'c1': 3, 'c2': 2, 'c3': 1}}
    >>> convert_valuations_to_matrix(valuations)
    [[], [0, 9, 8, 7], [0, 8, 7, 6], [0, 7, 6, 5], [0, 6, 5, 4], [0, 5, 4, 3], [0, 4, 3, 2], [0, 3, 2, 1]]
    """
    students = sorted(valuations.keys())  # Sort student keys to maintain order
    colleges = sorted(valuations[students[0]].keys())  # Sort college keys to maintain order
    V = []
    V.append([])
    for student in students:
        V.append([0] + [valuations[student][college] for college in colleges])
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
    {1: [1, 2], 2: [4, 3], 3: [7, 6, 5]}"""
  # this is the prev matching that i understand to be wrong because of indexes problem  {1: [1, 2, 3], 2: [5, 4], 3: [7, 6]}"""
    S = alloc.instance.agents
    C = alloc.instance.items
    V = alloc.instance._valuations
    # Now V look like this:
    # "Alice": {"c1":2, "c2": 3, "c3": 4},
    # "Bob": {"c1": 4, "c2": 5, "c3": 6}
    logger.info('FaSt(%s,%s,%s)',S,C,V)
    n=len(S)# number of students
    m = len(C)  # number of colleges
    i = n - 1  # start from the last student
    j = m # start from the last college
    # Initialize the first stable matching
    initial_matching = initialize_matching(n, m)
    # Convert Valuations to only numerical matrix
    V= convert_valuations_to_matrix(V)

    # Initialize the leximin tuple
    lex_tupl=get_leximin_tuple(initial_matching,V)

    # Initialize the position array pos, F, college_values
    pos= build_pos_array(initial_matching, V)
    college_values=build_college_values(initial_matching,V)
    logger.debug('Initial i:%d', i)
    logger.debug('Initial j:%d', j)
     # Initialize F as a list of two lists: one for students, one for colleges
    F_students = []
    F_colleges = []
    F_students.append(n)  # Add sn to the student list in F
    #logger.debug('Initialized F_students: %s, F_colleges: %s',  F_stduents, F_colleges)

    logger.debug('\n**initial_matching %s**', initial_matching)

    iteration = 1   # For logging
    while i > j - 1 and j > 1:
        logger.debug('\n**Iteration number %d**', iteration)
        logger.debug('Current i:%d', i)
        logger.debug('Current j:%d  ', j)
    
        logger.debug('V: %s',  V)
        logger.debug('V[i][j-1]: %d',  V[i][j-1])
        logger.debug('college_values:%s ', college_values)
        logger.debug('college_values[j]: %d',  college_values[j])
        # IMPORTANT! in the variable college_values we use in j and not j-1 because it build like this:  {1: 35, 2: 3, 3: 1}
        # So this: college_values[j] indeed gave us the right index ! [i.e. different structure!]
        if college_values[j] >= V[i][j-1]:  # In the algo the college_values is actually v
            j -= 1
        else:
            logger.info("index i:%s", i )
            logger.info("index j: %s", j)
            logger.debug('V[i][j]: %d',  V[i][j])
            if V[i][j] > college_values[j]: #Raw 11 in the article- different indixes because of different structures.
                initial_matching = Demote(initial_matching, i, j, 1)
                logger.debug('initial_matching after demote: %s',  initial_matching)

            else:
                if V[i][j] < college_values[j]:#Raw 14
                    j -= 1
                else:
                    # Lookahead
                    k = i
                    t = pos[i]
                    µ_prime = copy.deepcopy(initial_matching) # Deep copy
                    logger.debug('k: %s',  k)
                    logger.debug('t: %s',  t)
                    logger.debug('V[k][j]: %s',  V[k][j])
                    logger.debug('lex_tupl[t]: %s',  lex_tupl[t])
                    logger.debug('i: %s',  i)

                    while k > j - 1:
                        if V[k][j] > lex_tupl[t]:
                            i = k
                            logger.debug('Before demote: µ=%s, µ_prime=%s',  initial_matching, µ_prime)
                            initial_matching = Demote(copy.deepcopy(µ_prime), k, j, 1)
                            logger.debug('After demote: µ=%s, µ_prime=%s',  initial_matching, µ_prime)
                            break
                        else:
                            if V[i][j] < college_values[j]:# raw 24 in the article
                                j -= 1
                                break
                            else:
                                µ_prime = Demote(copy.deepcopy(µ_prime), k, j, 1)
                                k -= 1
                                t += 1
                    logger.debug('k:%s ,j: %s', k,j)
                    logger.debug('matching new: %s',  µ_prime)
                    logger.debug('initial_matching: %s',  initial_matching)

                    if k == j - 1 and initial_matching != µ_prime:
                        j -= 1
        # Updates
        # Update the college values
        college_values = build_college_values(initial_matching, V)  
        # Update leximin tuple 
        lex_tupl = get_leximin_tuple(initial_matching, V) 
        logger.debug('lex_tupl: %s',  lex_tupl)

        # Update position array
        pos = build_pos_array(initial_matching, V) 
        logger.debug('pos: %s',  pos)

         # Update F
        # Insert all students from i to n
        for student in range(i, n + 1):
            if student not in F_students:
                F_students.append(student)
        # Insert all colleges from j+1 to m
        for college in range(j + 1, m + 1):
            if college not in F_colleges:
                F_colleges.append(college)

        logger.debug('Updated F_students: %s, F_colleges: %s',  F_students, F_colleges)

        i -= 1
        iteration += 1
        logger.debug('END while, i: %d, j: %d',i, j)
    logger.debug(f"  final match:    {initial_matching}")
    return initial_matching

if __name__ == "__main__":
    console=logging.StreamHandler() #writes to stderr (= cerr)
    logger.handlers=[console] # we want the logs to be written to console
    # Change logger level
    logger.setLevel(logging.DEBUG) # Set logger level to DEBUG
    import doctest
    print(doctest.testmod())
