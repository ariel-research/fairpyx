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

    The threshold guarantee value Vn(α) follows the definition from the paper:

    For any integer n ≥ 2, and for α ∈ [0,1]:

        - If α = 0:
            Vn(α) = 1 / n

        - If α ≥ 1 / (n - 1):
            Vn(α) = 0

        - Otherwise, for each k ∈ ℕ, define:
            I(n, k) = [ (k+1) / (k((k+1)n - 1)), 1 / (k n - 1) ]
            NI(n, k) = ( 1 / ((k+1)n - 1), (k+1) / (k((k+1)n - 1)) )

          Then:
            Vn(α) = 1 - k(n - 1)α       if α ∈ I(n, k)
            Vn(α) = 1 - ((k+1)(n-1)) / ((k+1)n - 1)     if α ∈ NI(n, k)

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
    >>> alloc1.bundles["A"]
    ['1', '3']
    # Explanation:
    # A has values [6,3,1], sum=10, max=6 ⇒ α = 6/10 = 0.6
    # Since α ∈ NI(2,1), Vn(α) = 1 - (2×1)/(2×2 -1) = 1 - 2/3 = 1/3
    # Threshold to reach: 10 × 1/3 = 3.33
    # Bundle ['1','3'] = 6 + 1 = 7 ≥ 3.33 ⇒ OK
    >>> sumA = sum(instance1.valuations["A"].values())
    >>> alphaA = maxA / sumA
    >>> valA >= vnA
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
    >>> alloc2.bundles["A"]
    ['1']
    >>> alloc2.bundles["B"]
    ['2']
    >>> alloc2.bundles["C"]
    ['3', '4']
    >>> maxC = max(instance2.valuations["C"].values())
    >>> sumC = sum(instance2.valuations["C"].values())
    >>> alphaC = maxC / sumC
    >>> vnC = compute_vn(alphaC, n=3)
    >>> valC = sum(instance2.valuations["C"][i] for i in alloc2.bundles["C"])
    >>> valC >= vnC
    True

    Example 3 :
    >>> instance3 = Instance(
    ...     valuations={
    ...         "Alice": {"a": 10, "b": 3, "c": 1, "d": 1, "e": 2},
    ...         "Bob": {"a": 5, "b": 8, "c": 3, "d": 2, "e": 2},
    ...         "Carol": {"a": 1, "b": 3, "c": 9, "d": 6, "e": 5},
    ...         "Dave": {"a": 2, "b": 2, "c": 2, "d": 8, "e": 6}
    ...     }
    ... )
    >>> alloc3 = divide(instance3, algorithm1_worst_case_allocation)
    >>> alloc3.bundles["Alice"]
    ['a', 'd']
    >>> alloc3.bundles["Bob"]
    ['b']
    >>> alloc3.bundles["Carol"]
    ['c', 'e']
    >>> maxCarol = max(instance3.valuations["Carol"].values())
    >>> sumCarol = sum(instance3.valuations["Carol"].values())
    >>> alphaCarol = maxCarol / sumCarol
    >>> vnCarol = compute_vn(alphaCarol, n=4)
    >>> valCarol = sum(instance3.valuations["Carol"][i] for i in alloc3.bundles["Carol"])
    >>> valCarol >= vnCarol
    True

    Example 4 :
    >>> instance4 = Instance(
    ...     valuations={
    ...         "A": {"1": 9, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1},
    ...         "B": {"1": 1, "2": 9, "3": 1, "4": 1, "5": 1, "6": 1},
    ...         "C": {"1": 1, "2": 1, "3": 9, "4": 1, "5": 1, "6": 1},
    ...         "D": {"1": 1, "2": 1, "3": 1, "4": 9, "5": 1, "6": 1},
    ...         "E": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 9, "6": 1},
    ...         "F": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 9}
    ...     }
    ... )
    >>> alloc4 = divide(instance4, algorithm1_worst_case_allocation)
    >>> alloc4.bundles["A"]
    ['1']
    >>> alloc4.bundles["B"]
    ['2']
    >>> alloc4.bundles["C"]
    ['3']
    >>> alloc4.bundles["D"]
    ['4']
    >>> alloc4.bundles["E"]
    ['5']
    >>> alloc4.bundles["F"]
    ['6']
    >>> maxF = max(instance4.valuations["F"].values())
    >>> sumF = sum(instance4.valuations["F"].values())
    >>> alphaF = maxF / sumF
    >>> vnF = compute_vn(alphaF, n=6)
    >>> valF = sum(instance4.valuations["F"][i] for i in alloc4.bundles["F"])
    >>> valF >= vnF
    True

    # Manual explanation of Example 4:
    # - F values: [1,1,1,1,1,9], max=9, sum=14, α = 9/14 ≈ 0.642857
    # - (n - 1)α = 5 × 0.642857 ≈ 3.214 → k = floor(1 / 3.214) = 0 → reset to k = 1
    # - threshold = (k+1) / (k((k+1)n - 1)) = 2 / (1*(12 - 1)) = 2 / 11 ≈ 0.1818
    # - Since α > threshold, we are in I(n,k), and Vn(α) = 0 (because α > 1/(n-1) = 1/5)
    # - Hence, even a value of 9 is valid (≫ 0)

    Vn(alpha) Examples:
    >>> compute_vn(0, 3)
    0.3333333333333333
    >>> compute_vn(0.3, 3)
    0.2
    >>> compute_vn(0.45, 3)
    0.10000000000000009
    >>> compute_vn(0.6, 3)
    0.0
    >>> compute_vn(1 / (2 * 2), 3)
    0.0
    """
    return  # Empty implementation

"""
Note on normalization:
The input alpha must be a normalized value in the range [0,1],
representing the fraction of the highest-valued item among the total value
of the agent's valuation function.

For example, if an agent values items as {"1": 6, "2": 3, "3": 1},
then alpha should be computed as:

    alpha = max(values) / sum(values) = 6 / (6 + 3 + 1) = 0.6

This is because the function Vn(alpha) is defined on the domain [0,1] as per the paper:

    Vn : [0,1] → [0,1/n]

Sending raw values (e.g., 6) instead of normalized ratios will lead to incorrect results.
"""

def compute_vn(alpha: float, n: int) -> float:
    """ 
    Computes the worst-case guarantee value Vn(alpha) for a given agent.

    This function implements the piecewise definition from Definition 1 in the paper:
    - If alpha == 0 → Vn(alpha) = 1 / n
    - If alpha >= 1 / (n-1) → Vn(alpha) = 0
    - Else → determine k and calculate based on whether alpha ∈ I(n,k) or NI(n,k)

    :param alpha: The value of the largest single item for the agent.
    :param n: The total number of agents.
    :return: The worst-case guaranteed value Vn(alpha).
    """
    if alpha == 0:
        return 1.0 / n
    if alpha >= 1.0 / (n - 1):
        return 0.0
    inv = 1.0 / ((n - 1) * alpha)
    k_floor = int(inv)
    if k_floor < 1:
        k_floor = 1
    cand = (1.0 / alpha + 1.0) / n - 1.0
    k = k_floor
    if abs(round(cand) - cand) < 1e-9:
        k_candidate = int(round(cand))
        if k_candidate >= 1:
            k = k_candidate
    threshold = (k + 1) / (k * (((k + 1) * n) - 1))
    if alpha < threshold:
        return 1.0 - ((k + 1) * (n - 1)) / (((k + 1) * n) - 1)
    else:
        return 1.0 - k * (n - 1) * alpha
