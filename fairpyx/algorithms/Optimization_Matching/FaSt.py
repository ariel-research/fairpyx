"""
"On Achieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

Programmer: Hadar Bitan, Yuval Ben-Simhon
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging

logger = logging.getLogger(__name__)

def FaSt():
    """
    FaSt algorithm: Find a leximin optimal stable matching under ranked isometric valuations.
# def FaSt(instance: Instance, explanation_logger: ExplanationLogger = ExplanationLogger()):
    :param instance: Instance of ranked isometric valuations <S,C,V>

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
    >>> instance = Instance(S, C, V)
    >>> FaSt(instance)
    >>> alloc = AllocationBuilder(agent_capacities={s: 1 for s in S}, item_capacities={c: 1 for c in C}, valuations=V)
    >>> alloc.get_allocation()
    {'s1': ['c1'], 's2': ['c2'], 's3': ['c1'], 's4': ['c3'], 's5': ['c3'], 's6': ['c3'], 's7': ['c2']}
    """

if __name__ == "__main__":
    import doctest
    doctest.testmod()
