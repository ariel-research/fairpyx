"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
logger = logging.getLogger(__name__)


def Demote():
    """
    Demote algorithm: Adjust the matching by moving a student to a lower-ranked college
    while maintaining the invariant of a complete stable matching.
    The Demote algorithm is a helper function used within the FaSt algorithm to adjust the matching while maintaining stability.

    # def Demote(alloc: AllocationBuilder, student_index: int, down_index: int, up_index: int):
    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity.
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
if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())
