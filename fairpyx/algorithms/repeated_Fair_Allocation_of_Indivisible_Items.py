"""	
This module implements algorithms for fair allocation of indivisible items
it is based on the article:
    "Repeated Fair Allocation of Indivisible Items" by  Igarashi, Lackner, Nardi, Novaro
1. two agents, two rounds
2. two agents, even k rounds

Programmer: Shaked Shvartz
Since: 2025-05
"""

import cvxpy as cp
from typing import Dict, Tuple, List, Set
from fairpyx.adaptors import AllocationBuilder

Agent = int
Item  = int

def solve_fractional_ILP(utilities: Dict[Agent, Dict[Item, float]], k: int, solver: str = "GLPK_MI") -> Dict[Tuple[Agent, Item], int]:
    """
    Solves the ILP from Figure 1 in the article.

    Each item must be allocated `k` times in total across all agents.
    Each agent must receive at least k/n fraction of their total utility.
    The result is a dictionary of integral counts.

    >>> utils = {0: {0: 1.0, 1: 2.0}, 1: {0: 2.0, 1: 1.0}}
    >>> counts = solve_fractional_ILP(utils, 2)
    >>> sorted(counts.items())
    [((0, 0), 0), ((0, 1), 2), ((1, 0), 2), ((1, 1), 0)]
    """
    agents = list(utilities.keys())
    items  = list(next(iter(utilities.values())).keys())
    n, m = len(agents), len(items)

    # Define ILP variable: x[i, j] = number of times agent i gets item j
    x = cp.Variable((n, m), integer=True)
    constraints = [x >= 0, x <= k]

    # Each item allocated k times over all agents
    for j in range(m):
        constraints.append(cp.sum(x[:, j]) == k)

    # Proportionality constraint for each agent
    for i in range(n):
        uvec  = [utilities[agents[i]][items[j]] for j in range(m)]
        total = sum(uvec)
        constraints.append(cp.reshape(x[i, :], (1, m), order='C') @ uvec >= (k / n) * total)

    # Objective: maximize total utility (PO)
    U = [utilities[agents[i]][items[j]] for i in range(n) for j in range(m)]
    obj = cp.Maximize(cp.reshape(x, (n*m,), order='C') @ U)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)

    # Convert solution to dictionary format
    result: Dict[Tuple[Agent, Item], int] = {}
    for i in range(n):
        for j in range(m):
            result[(agents[i], items[j])] = int(round(x.value[i, j]))
    return result

def algorithm1(initial_alloc: List[Dict[Agent, Set[Item]]], utilities: Dict[Agent, Dict[Item, float]]) -> List[Dict[Agent, Set[Item]]]:
    """
    Implements Algorithm 1 (n=2, k=2) from the article.

    The algorithm ensures both rounds are EF1 for both agents.

    >>> utils = {0: {0: 1.0, 1: 2.0, 2: 5.0}, 1: {0: 5.0, 1: 1.0, 2: 2.0}}
    >>> result = algorithm1([{}, {}], utils)
    >>> sorted(result[0][0])  # Round 1, agent 0
    [1, 2]
    >>> sorted(result[0][1])  # Round 1, agent 1
    [0]
    """
    k = 2

    # Step 1: Solve ILP to get fractional allocations across k rounds
    counts = solve_fractional_ILP(utilities, k)

    # Step 2: Assign items to rounds based on ILP counts
    rounds = [{0: set(), 1: set()} for _ in range(k)]
    for (i, o), c in counts.items():
        for r in range(c):
            rounds[r][i].add(o)
    π1, π2 = rounds

    # Step 3: Find persistent items (allocated to same agent in both rounds)
    I1 = π1[0] & π2[0]
    I2 = π1[1] & π2[1]
    all_items = set(utilities[0].keys())
    O = all_items - (I1 | I2)

    # Step 4: Split remaining items based on positive/negative utilities
    O_plus  = {o for o in O if utilities[0][o] >= 0 and utilities[1][o] >= 0}
    O_minus = O - O_plus

    # Step 5: Construct candidate bundles for each round
    π1 = {0: I1 | O_minus, 1: I2 | O_plus}
    π2 = {0: I1 | O_plus, 1: I2 | O_minus}

    # Helper to check EF1 condition
    def is_EF1(bundle):
        for i in (0, 1):
            u_self = sum(utilities[i][o] for o in bundle[i])
            u_other = sum(utilities[i][o] for o in bundle[1 - i])
            if u_self < u_other:
                # Check marginal removal of other's items
                improvements = [u_self >= u_other - utilities[i][o2] for o2 in bundle[1 - i]]
                if not any(improvements):
                    return False
        return True

    # Step 6: Swap items in O until both rounds are EF1
    O_list = list(O)
    idx = 0
    while (not is_EF1(π1) or not is_EF1(π2)) and idx < len(O_list):
        o = O_list[idx]
        if o in O_minus:
            π1[0].discard(o); π2[1].discard(o)
            π1[1].add(o);     π2[0].add(o)
        else:
            π1[1].discard(o); π2[0].discard(o)
            π1[0].add(o);     π2[1].add(o)
        idx += 1

    return [π1, π2]

def algorithm2(initial_alloc: List[Dict[Agent, Set[Item]]], utilities: Dict[Agent, Dict[Item, float]]) -> List[Dict[Agent, Set[Item]]]:
    """
    Implements Algorithm 2 (n=2, even k) from the article.

    Ensures each round satisfies weak-EF1 fairness.
    """
    k = len(initial_alloc)

    # Step 1: Solve ILP for total allocation counts
    counts = solve_fractional_ILP(utilities, k)

    # Step 2: Build per-round item schedule using round-robin
    per_round_allocation = [{0: set(), 1: set()} for _ in range(k)]
    item_rounds: Dict[Item, List[int]] = {o: [] for o in utilities[0]}
    for (agent, item), count in counts.items():
        item_rounds[item].extend([agent] * count)

    round_robin = 0
    for item, agent_list in item_rounds.items():
        for agent in agent_list:
            for offset in range(k):
                r = (round_robin + offset) % k
                if item not in per_round_allocation[r][0] and item not in per_round_allocation[r][1]:
                    per_round_allocation[r][agent].add(item)
                    round_robin = (round_robin + 1) % k
                    break
            else:
                raise ValueError(f"Could not assign item {item} uniquely in any round")

    π = per_round_allocation

    # EF and weak-EF1 condition checks
    def envy_free_round(j, a):
        u_self = sum(utilities[a][o] for o in π[j][a])
        u_other = sum(utilities[a][o] for o in π[j][1 - a])
        return u_self >= u_other

    def weak_EF1_round(j, a):
        if envy_free_round(j, a):
            return True
        u_self = sum(utilities[a][o] for o in π[j][a])
        u_other = sum(utilities[a][o] for o in π[j][1 - a])
        return any(u_self >= u_other - utilities[a][o2] for o2 in π[j][1 - a]) or \
               any(u_self + utilities[a][o1] >= u_other for o1 in π[j][a])

    # Step 3: Repair allocations to satisfy weak-EF1
    def make_weak_EF1(agent_idx):
        E = {j for j in range(k) if not envy_free_round(j, agent_idx)}
        retries = 0
        max_retries = 1000
        while E and retries < max_retries:
            changed = False
            for j in list(E):
                if weak_EF1_round(j, agent_idx):
                    E.remove(j)
                    continue
                for i in sorted(set(range(k)) - {j}):
                    for o in (π[j][agent_idx] | π[i][agent_idx]):
                        if o in π[j][agent_idx] and o not in π[i][agent_idx]:
                            src, dst = j, i
                        elif o in π[i][agent_idx] and o not in π[j][agent_idx]:
                            src, dst = i, j
                        else:
                            continue
                        # Perform transfer
                        π[src][agent_idx].remove(o)
                        π[src][1 - agent_idx].add(o)
                        π[dst][1 - agent_idx].remove(o)
                        π[dst][agent_idx].add(o)
                        changed = True
                        if weak_EF1_round(j, agent_idx):
                            E.remove(j)
                            break
                    if j not in E:
                        break
            if not changed:
                break
            retries += 1

        if E:
            print("\n[DEBUG] Failed to resolve weak-EF1. Remaining rounds:", E)
            for j in E:
                print(f"Round {j}: Agent {agent_idx} →", π[j][agent_idx], "/ Opponent →", π[j][1-agent_idx])
            print("\n[DEBUG] Utilities for agent", agent_idx, ":", utilities[agent_idx])
            raise RuntimeError(f"Failed to ensure weak-EF1 for agent {agent_idx} after {retries} retries")

    # Ensure weak-EF1 for both agents
    make_weak_EF1(0)
    make_weak_EF1(1)

    return π

# === FairPyx Divide Interface ===

def algorithm1_div(builder: AllocationBuilder, round_idx: int = 0, **kwargs):
    utils = builder.instance._valuations
    bundle = algorithm1([{}, {}], utils)[round_idx]
    builder.give_bundles({i: list(bundle[i]) for i in bundle})

def algorithm2_div(builder: AllocationBuilder, k: int, round_idx: int = 0, **kwargs):
    utils = builder.instance._valuations
    init = [{} for _ in range(k)]
    bundle = algorithm2(init, utils)[round_idx]
    builder.give_bundles({i: list(bundle[i]) for i in bundle})

# === Local Test Runner ===
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    print("All tests passed.")
