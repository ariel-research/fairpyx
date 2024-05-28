"""
"On Achieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging

logger = logging.getLogger(__name__)

def Demote(matching:dict, student_index:int, down:int, up:int):
    """
    Demote algorithm: Adjust the matching by moving a student to a lower-ranked college
    while maintaining the invariant of a complete stable matching.
    The Demote algorithm is a helper function used within the FaSt algorithm to adjust the matching while maintaining stability.

    :param matching: the matchinf of the students with colleges.
    :param student_index: Index of the student to move.
    :param down_index: Index of the college to move the student to.
    :param up_index: Index of the upper bound college.

        # The test is the same as the running example we gave in Ex2.

    >>> from fairpyx import AllocationBuilder
    >>> alloc = AllocationBuilder(agent_capacities={"s1": 1, "s2": 1, "s3": 1, "s4": 1, "s5": 1}, item_capacities={"c1": 1, "c2": 2, "c3": 2})
    >>> alloc.add_allocation(0, 0)  # s1 -> c1
    >>> alloc.add_allocation(1, 1)  # s2 -> c2
    >>> alloc.add_allocation(2, 1)  # s3 -> c2
    >>> alloc.add_allocation(3, 2)  # s4 -> c3
    >>> alloc.add_allocation(4, 2)  # s5 -> c3
    >>> Demote(alloc, 2, 2, 1)
    >>> alloc.get_allocation()
    {'s1': ['c1'], 's2': ['c2'], 's3': ['c3'], 's4': ['c3'], 's5': ['c2']}
    """
    return 0


def FaSt(alloc: AllocationBuilder):
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
    >>> instance = Instance(agents=S, items=C, valuations=V)
    >>> divide(FaSt, instance=instance)
    {'s1': ['c1'], 's2': ['c2'], 's3': ['c1'], 's4': ['c3'], 's5': ['c3'], 's6': ['c3'], 's7': ['c2']}
    """
    return 0



if __name__ == "__main__":
    import doctest
    doctest.testmod()
