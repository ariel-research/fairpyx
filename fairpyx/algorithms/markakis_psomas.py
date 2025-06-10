"""
An implementation of the algorithms in:
"On Worst-Case Allocations in the Presence of Indivisible Goods"
by Evangelos Markakis and Christos-Alexandros Psomas (2011).
https://link.springer.com/chapter/10.1007/978-3-642-25510-6_24
http://pages.cs.aueb.gr/~markakis/research/wine11-Vn.pdf

Programmer: Ibrahem Hurani
Date: 2025-05-06
"""
import logging
import math

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
    """
    if n <= 1:
        return 0.0
    if alpha == 0:
        return 1.0 / n
    
    for k in range(1, 1000):
        I_left=(k + 1) / (k * ((k + 1) * n - 1))
        I_right=1 / (k * n - 1)
        NI_left= 1 / ((k + 1) * n - 1)
        NI_right=(k + 1) / (k * ((k + 1) * n - 1))
        # Check I(n,k) range (closed interval)
        if I_left <= alpha <= I_right or math.isclose(alpha, I_left) or math.isclose(alpha, I_right):
            return 1 - k * (n - 1) * alpha
        # Check NI(n,k) range (open interval)
        if NI_left < alpha < NI_right or math.isclose(alpha, NI_left) or math.isclose(alpha, NI_right):
            return 1 - ((k + 1) * (n - 1)) / ((k + 1) * n - 1)
    return 0.0


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
    >>> from fairpyx.algorithms.markakis_psomas import algorithm1_worst_case_allocation, compute_vn
    >>> instance1 = Instance(valuations={"A": {"1": 6, "2": 3, "3": 1}, "B": {"1": 2, "2": 5, "3": 5}})
    >>> alloc1 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance1)
    >>> sorted(alloc1["A"])
    ['1']
    >>> sorted(alloc1["B"])
    ['2', '3']
    
    Example 2:
    >>> instance2 = Instance(valuations={"A": {"1": 7, "2": 2, "3": 1, "4": 1}, "B": {"1": 3, "2": 6, "3": 1, "4": 2}, "C": {"1": 2, "2": 3, "3": 5, "4": 5}})
    >>> alloc2 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance2)
    >>> sorted(alloc2["A"])
    ['1']
    >>> sorted(alloc2["B"])
    ['2']
    >>> sorted(alloc2["C"])
    ['3', '4']
   
    Example 3:
    >>> instance3 = Instance(valuations={"Alice": {"a": 10, "b": 3, "c": 1, "d": 1, "e": 2}, "Bob": {"a": 5, "b": 8, "c": 3, "d": 2, "e": 2}, "Carol": {"a": 1, "b": 3, "c": 9, "d": 6, "e": 5}, "Dave": {"a": 2, "b": 2, "c": 2, "d": 8, "e": 6}})
    >>> alloc3 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance3)
    >>> sorted(alloc3["Alice"])
    ['a']
    >>> sorted(alloc3["Bob"])
    ['b']
    >>> sorted(alloc3["Carol"])
    ['c']
    >>> sorted(alloc3["Dave"])
    ['d', 'e']
   

    Example 4:
    >>> instance4 = Instance(valuations={
    ...     "A": {"1": 9, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1},
    ...     "B": {"1": 1, "2": 9, "3": 1, "4": 1, "5": 1, "6": 1},
    ...     "C": {"1": 1, "2": 1, "3": 9, "4": 1, "5": 1, "6": 1},
    ...     "D": {"1": 1, "2": 1, "3": 1, "4": 9, "5": 1, "6": 1},
    ...     "E": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 9, "6": 1},
    ...     "F": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 9}
    ... })
    >>> alloc4 = divide(algorithm=algorithm1_worst_case_allocation, instance=instance4)
    >>> [sorted(alloc4[a]) for a in sorted(alloc4)]
    [['1'], ['2'], ['3'], ['4'], ['5'], ['6']]
   

    # Manual explanation of Example 4:
    # - F values: [1,1,1,1,1,9], max=9, sum=14, α = 9/14 ≈ 0.642857
    # - (n - 1)α = 5 × 0.642857 ≈ 3.214 → k = floor(1 / 3.214) = 0 → reset to k = 1
    # - threshold = (k+1) / (k((k+1)n - 1)) = 2 / (1*(12 - 1)) = 2 / 11 ≈ 0.1818
    # - Since α > threshold, we are in I(n,k), and Vn(α) = 0 (because α > 1/(n-1) = 1/5)
    # - Hence, even a value of 9 is valid (≫ 0)

    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    if alloc is None:
        return {}

    n = len(alloc.remaining_agents())
    logger.info(f"\nNew recursive call with {n} agents")

    if n == 1:
        agent = next(iter(alloc.remaining_agents()))
        items = list(alloc.remaining_items_for_agent(agent))
        logger.info(f"Only one agent '{agent}' remains — giving all items: {items}")
        alloc.give_bundle(agent, items)
        return

   # Step 1: Construct Si incrementally (lines 1–8 in algorithm)
    bundles = {agent: [] for agent in alloc.remaining_agents()}
    values = {agent: 0 for agent in alloc.remaining_agents()}
    Vn_alpha_i = {}
    pointers = {agent: 0 for agent in alloc.remaining_agents()}
    max_values = {}
    # Precompute thresholds for each agent
    for agent in alloc.remaining_agents():
        items = alloc.remaining_items_for_agent(agent)
        values_list = [alloc.effective_value(agent, item) for item in items]
        max_val = max(values_list, default=0)
        total_val = sum(values_list)
        alpha = max_val / total_val if total_val > 0 else 0
        Vn_alpha = compute_vn(alpha, n)
        Vn_alpha_i[agent] = Vn_alpha * total_val
        max_values[agent] = max_val  # Store max value for fallback
        logger.info(f"Agent '{agent}': max={max_val:.2f}, sum={total_val:.2f}, alpha={alpha:.3f}, Vn(α)={Vn_alpha:.3f}, threshold={Vn_alpha_i[agent]:.2f}")

    # Pre-sort items for each agent
    sorted_items = {
        agent: sorted(alloc.remaining_items_for_agent(agent), key=lambda item: alloc.effective_value(agent, item), reverse=True)
        for agent in alloc.remaining_agents()
    }
    

    # Add items until at least one agent meets the minimum value
    while all(values[agent] <= Vn_alpha_i[agent] for agent in alloc.remaining_agents()):
        for agent in alloc.remaining_agents():
            if pointers[agent] < len(sorted_items[agent]):
                item = sorted_items[agent][pointers[agent]]
                val = alloc.effective_value(agent, item)
                bundles[agent].append(item)
                values[agent] += val
                pointers[agent] += 1
                
            else:
                # If no more items, use current bundle
                pass

        # Check if we can break early
        if all(pointers[agent] >= len(sorted_items[agent]) for agent in alloc.remaining_agents()):
            break
    
    # Choose agent who meets the threshold and has non-empty bundle
    chosen_agent = None
    for agent in alloc.remaining_agents():
        if values[agent] >= Vn_alpha_i[agent]:
            chosen_agent = agent
            logger.info(f"is  agent '{chosen_agent}' ")
            break
    

    # Allocate the bundle
    bundle = bundles[chosen_agent]
    logger.info(f"Assigning bundle {bundle} to agent '{chosen_agent}' (value={values[chosen_agent]:.2f} ≥ min={Vn_alpha_i[agent]:.2f})")
    alloc.give_bundle(chosen_agent, bundle)
    alloc.remove_agent_from_loop(chosen_agent)
    
    # Recursive call
    algorithm1_worst_case_allocation(alloc=alloc)

    


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

if __name__ == "__main__":
    valuations={"A": {"1": 7, "2": 2, "3": 1, "4": 1}, "B": {"1": 3, "2": 6, "3": 1, "4": 2}, "C": {"1": 2, "2": 3, "3": 5, "4": 5}}  
    instance = Instance(valuations=valuations)

    allocation = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    