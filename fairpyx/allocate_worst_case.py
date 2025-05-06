"""
An implementation of the algorithms in:
"On Worst-Case Allocations in the Presence of Indivisible Goods"
by Evangelos Markakis and Christos-Alexandros Psomas (2011).
https://link.springer.com/chapter/10.1007/978-3-642-25510-6_24
http://pages.cs.aueb.gr/~markakis/research/wine11-Vn.pdf

Programmer: Ibrahem Hurani
Date: 2025-05-06
"""

from fairpyx import AllocationBuilder

def algorithm1_worst_case_allocation(alloc: AllocationBuilder) -> None:
    """
    Algorithm 1: Allocates items such that each agent receives a bundle worth at least their worst-case guarantee.

    This algorithm incrementally builds bundles for agents based on their preferences until one agent reaches
    their worst-case guarantee. That agent receives the bundle, then we normalize the remaining agents and recurse.

    אלגוריתם זה בונה חבילות עד שסוכן מקבל לפחות את ההבטחה הגרועה שלו, מקצה לו, מנרמל וממשיך רקורסיבית.

    :param alloc: AllocationBuilder — the current allocation state and remaining instance.
    :return: None — allocation is done in-place inside `alloc`.

    Example 1 :
    >>> from fairpyx import Instance, divide
    >>> instance1 = Instance(
    ...     valuations={
    ...         "A": {"1": 6, "2": 3, "3": 1},
    ...         "B": {"1": 2, "2": 5, "3": 5}
    ...     }
    ... )
    >>> alloc1 = divide(instance1, algorithm1_worst_case_allocation)
    >>> set(alloc1.bundles["A"]) == {"1", "3"} or set(alloc1.bundles["A"]) == {"3", "1"}
    True
    >>> set(alloc1.bundles["B"]) == {"2"}
    True

    Example 2 :
    >>> instance2 = Instance(
    ...     valuations={
    ...         "A": {"1": 7, "2": 2, "3": 1, "4": 1},
    ...         "B": {"1": 3, "2": 6, "3": 1, "4": 2},
    ...         "C": {"1": 2, "2": 3, "3": 5, "4": 5}
    ...     }
    ... )
    >>> alloc2 = divide(instance2, algorithm1_worst_case_allocation)
    >>> set(alloc2.bundles["C"]) == {"3", "4"}
    True
    >>> sorted(alloc2.bundles["A"] + alloc2.bundles["B"] + alloc2.bundles["C"]) == ["1", "2", "3", "4"]
    True
    """
    return  # Empty implementation
