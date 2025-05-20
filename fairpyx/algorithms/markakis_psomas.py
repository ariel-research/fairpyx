"""
An implementation of the algorithms in:
"On Worst-Case Allocations in the Presence of Indivisible Goods"
by Evangelos Markakis and Christos-Alexandros Psomas (2011).
https://link.springer.com/chapter/10.1007/978-3-642-25510-6_24
http://pages.cs.aueb.gr/~markakis/research/wine11-Vn.pdf

Programmer: Ibrahem Hurani
Date: 2025-05-06
"""

from fairpyx import AllocationBuilder,divide,Instance

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
     Vn(alpha) Examples:
    >>> compute_vn(0, 3)
    0.3333333333333333
    >>> compute_vn(0.3, 3)
    0.19999999999999996
    >>> compute_vn(0.6, 3)
    0.0
    """
    if n<=1:
        return 0.0
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

    Example 1:
    >>> from fairpyx import Instance, divide
    >>> instance1 = Instance(valuations={"A": {"1": 6, "2": 3, "3": 1}, "B": {"1": 2, "2": 5, "3": 5}})
    >>> alloc1 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance1)
    >>> alloc1["A"]
    ['1']
    >>> alloc1["B"]
    ['2', '3']
  

    # Explanation:
    # A has values [6,3,1], sum=10, max=6 ⇒ α = 6/10 = 0.6
    # Since α ∈ NI(2,1), Vn(α) = 1 - (2×1)/(2×2 -1) = 1 - 2/3 = 1/3
    # Threshold to reach: 10 × 1/3 = 3.33
    # Bundle ['1'] = 6  ≥ 3.33 ⇒ OK

    Example 2 :
    >>> instance2 = Instance(valuations={"A": {"1": 7, "2": 2, "3": 1, "4": 1}, "B": {"1": 3, "2": 6, "3": 1, "4": 2}, "C": {"1": 2, "2": 3, "3": 5, "4": 5}})
    >>> alloc2 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance2)
    >>> sorted(alloc2["A"])
    ['1']
    >>> sorted(alloc2["B"])
    ['2']
    >>> sorted(alloc2["C"])
    ['3', '4']
    >>> maxC = max(instance2.valuations["C"].values())
    >>> sumC = sum(instance2.valuations["C"].values())
    >>> alphaC = maxC / sumC
    >>> vnC = compute_vn(alphaC, n=3)
    >>> valC = sum(instance2.valuations["C"][i] for i in alloc2["C"])
    >>> valC >= vnC
    True

    Example 3:
    >>> instance3 = Instance(valuations={"Alice": {"a": 10, "b": 3, "c": 1, "d": 1, "e": 2}, "Bob": {"a": 5, "b": 8, "c": 3, "d": 2, "e": 2}, "Carol": {"a": 1, "b": 3, "c": 9, "d": 6, "e": 5}, "Dave": {"a": 2, "b": 2, "c": 2, "d": 8, "e": 6}})
    >>> alloc3 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance3)
    >>> sorted(alloc3["Alice"])
    ['a', 'd']
    >>> sorted(alloc3["Bob"])
    ['b']
    >>> sorted(alloc3["Carol"])
    ['c', 'e']
    >>> maxCarol = max(instance3.valuations["Carol"].values())
    >>> sumCarol = sum(instance3.valuations["Carol"].values())
    >>> alphaCarol = maxCarol / sumCarol
    >>> vnCarol = compute_vn(alphaCarol, n=4)
    >>> valCarol = sum(instance3.valuations["Carol"][i] for i in alloc3["Carol"])
    >>> valCarol >= vnCarol
    True

    Example 4:
    >>> instance4 = Instance(valuations={"A": {"1": 9, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1}, "B": {"1": 1, "2": 9, "3": 1, "4": 1, "5": 1, "6": 1}, "C": {"1": 1, "2": 1, "3": 9, "4": 1, "5": 1, "6": 1}, "D": {"1": 1, "2": 1, "3": 1, "4": 9, "5": 1, "6": 1}, "E": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 9, "6": 1}, "F": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 9}})
    >>> alloc4 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance4)
    >>> [alloc4[a] for a in sorted(alloc4.keys())]
    [['1'], ['2'], ['3'], ['4'], ['5'], ['6']]
    >>> maxF = max(instance4.valuations["F"].values())
    >>> sumF = sum(instance4.valuations["F"].values())
    >>> alphaF = maxF / sumF
    >>> vnF = compute_vn(alphaF, n=6)
    >>> valF = sum(instance4.valuations["F"][i] for i in alloc4["F"])
    >>> valF >= vnF
    True

    # Manual explanation of Example 4:
    # - F values: [1,1,1,1,1,9], max=9, sum=14, α = 9/14 ≈ 0.642857
    # - (n - 1)α = 5 × 0.642857 ≈ 3.214 → k = floor(1 / 3.214) = 0 → reset to k = 1
    # - threshold = (k+1) / (k((k+1)n - 1)) = 2 / (1*(12 - 1)) = 2 / 11 ≈ 0.1818
    # - Since α > threshold, we are in I(n,k), and Vn(α) = 0 (because α > 1/(n-1) = 1/5)
    # - Hence, even a value of 9 is valid (≫ 0)

    """
    
    if alloc is None:
        return {}
    n = len(alloc.remaining_agents())
    if n==1:
        agent = next(iter(alloc.remaining_agents()))
        items = list(alloc.remaining_items_for_agent(agent))
        alloc.give_bundle(agent, items)
        return
   
    # Step 1: Build Si bundles
    bundles = {}
    for agent in alloc.remaining_agents():
        items = sorted(
            alloc.remaining_items_for_agent(agent),
            key=lambda item: alloc.effective_value(agent, item),
            reverse=True
            # effective_value:
            # היא משמשת כדי להחזיר את שווי הפריט item עבור הסוכן agent,
            #  תוך התחשבות בהקצאות הקודמות, בקונפליקטים ובמגבלות של משקל/כמות.
        )
        bundle = []
        value = 0
        max_value = max([alloc.effective_value(agent, item) for item in items], default=0)
        total_value = sum([alloc.effective_value(agent, item) for item in items])
        alpha = max_value / total_value if total_value > 0 else 0
        Vn_alpha = compute_vn(alpha, n)
        Vn_alpha_i = Vn_alpha * total_value

        for item in items:
            v = alloc.effective_value(agent, item)
            if v == float('-inf'):
                continue
            bundle.append(item)
            value += v
            if value >= Vn_alpha_i:
                break
        bundles[agent] = (bundle, value, Vn_alpha_i)

    # Step 2: Choose an agent whose bundle passes the threshold->Vn_Alpha_i
    for agent, (bundle, value,Vn_alpha_i) in bundles.items():
        if value >= Vn_alpha_i:
            alloc.give_bundle(agent, bundle)
            break
    else:
        return  # No agent met their threshold — should not happen with valid inputs

    remaining_agents = [a for a in alloc.remaining_agents() if a != agent]
    remaining_items = [i for i in alloc.remaining_items() if i not in bundle]

    # Step 3: Handle the base case
    if len(remaining_agents) == 1:
        last_agent = remaining_agents[0]
        alloc.give_bundle(last_agent, remaining_items)
        return

    # Step 4: Recursive step with normalized instance
    instance = alloc.instance
    reduced_valuations = {}
    for a in remaining_agents:
        vals = {}
        remaining_total = sum(instance.agent_item_value(a, i) for i in remaining_items)
        denom = remaining_total
        for i in remaining_items:
            original = instance.agent_item_value(a, i)
            vals[i] = original / denom if denom > 0 else 0
        reduced_valuations[a] = vals

    reduced_instance = Instance(
        valuations=reduced_valuations,
        agent_capacities={a: instance.agent_capacity(a) for a in remaining_agents},
        item_capacities={i: instance.item_capacity(i) for i in remaining_items},
        item_weights={i: instance.item_weight(i) for i in remaining_items}
    )

    reduced_alloc = AllocationBuilder(reduced_instance)
    algorithm1_worst_case_allocation(reduced_alloc)
    for a in reduced_alloc.bundles:
        alloc.give_bundle(a, reduced_alloc.bundles[a])


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