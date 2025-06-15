"""
Repeated Fair Allocation of Indivisible Items
=============================================
Implementation of the two–agent algorithms from

    Igarashi · Lackner · Nardi · Novaro (2024)
    "Repeated Fair Allocation of Indivisible Items"

Algorithm 1  – n = 2, k = 2         (per-round  EF1  +  PO overall)  
Algorithm 2  – n = 2, k even        (per-round weak-EF1 +  PO overall)

Author :  Shaked Shvartz
Since :  2025-05
"""

from   typing import Dict, Tuple, List, Set
import cvxpy as cp
import numpy as np
from   fairpyx.adaptors import AllocationBuilder, divide
import logging
log = logging.getLogger("fairpyx.rfaoii")
# ------------------------------------------------------------------

Agent  = int | str              # may come from JSON as str
Item   = int | str
Bundle = Set[Item]
OneDayAllocation = Dict[Agent, Bundle]



# ------------------------------------------------------------------
# 0-ter.  Public fairness predicates (Definition 2 and weak-EF1) and round-robin scheduler
# ------------------------------------------------------------------
def _to_set(x):
    """Ensure we work with a real set."""
    return x if isinstance(x, set) else set(x)


def EF1_holds(
    allocation: OneDayAllocation,
    agent: Agent,
    utilities: Dict[Agent, Dict[Item, float]],
) -> bool:
    """
    Return **True** iff Definition 2 (EF1) holds for *agent*.

    Only the return value is asserted; the log message is ignored by doctest.

    Examples
    --------
    >>> utils = {0:{0:4, 1:1}, 1:{0:1, 1:4}}
    >>> EF1_holds({0:{0}, 1:{1}}, 0, utils)          # no envy
    True
    >>> utils = {0:{0:5, 1:1}, 1:{0:6, 1:2}}
    >>> EF1_holds({0:{1}, 1:{0}}, 0, utils)          # remove opp-item 0
    True
    >>> utils = {0:{0:1, 1:-10}, 1:{0:1, 1:10}}
    >>> EF1_holds({0:{1}, 1:{0}}, 0, utils)          # still envies
    False
    """
    A = _to_set(allocation[agent])
    B = _to_set(allocation[1 - agent])
    u_self  = sum(utilities[agent][o] for o in A)
    u_other = sum(utilities[agent][o] for o in B)

    if u_self >= u_other:
        log.info("Agent %d: no envy (u_self=%s ≥ u_other=%s)", agent, u_self, u_other)
        return True

    for o in A | B:
        u_self_after  = u_self  - (utilities[agent][o] if o in A else 0)
        u_other_after = u_other - (utilities[agent][o] if o in B else 0)
        if u_self_after >= u_other_after:
            side = "own" if o in A else "opp"
            log.info("Agent %d: EF1 by removing %s item %r", agent, side, o)
            return True

    log.info("Agent %d: **violates EF1** (u_self=%s, u_other=%s)", agent, u_self, u_other)
    return False


# ---------------------------------------------------------------------------
# weak-EF1 (Definition 6) helper  + doctests
# ---------------------------------------------------------------------------
def weak_EF1_holds(
    allocation: OneDayAllocation,
    agent: Agent,
    utilities: Dict[Agent, Dict[Item, float]],
) -> bool:
    """
    Return **True** iff Definition 6 (weak-EF1) holds for *agent*.

    Examples
    --------
    >>> utils = {0:{0:10, 1:1}, 1:{0:6, 1:8}}
    >>> weak_EF1_holds({0:{1}, 1:{0}}, 0, utils)     # take opp-item 0
    True
    >>> weak_EF1_holds({0:{1}, 1:{0}}, 1, utils)     # already no envy
    True
    >>> utils = {0:{0:9, 1:8}, 1:{0:1, 1:2}}
    >>> weak_EF1_holds({0:{0,1}, 1:set()}, 1, utils) # take item 1 from opp
    True
    """
    A = _to_set(allocation[agent])
    B = _to_set(allocation[1 - agent])
    u_self  = sum(utilities[agent][o] for o in A)
    u_other = sum(utilities[agent][o] for o in B)

    if u_self >= u_other:
        log.info("Agent %d: no envy (u_self=%s ≥ u_other=%s)", agent, u_self, u_other)
        return True

    for o in A | B:
        val = utilities[agent][o]
        if o in B:                                   # ADD to self / REMOVE from other
            if u_self + val >= u_other - val:
                log.info("Agent %d: weak-EF1 by taking %r from opp", agent, o)
                return True
        else:                                        # REMOVE from self / ADD to other
            if u_self - val >= u_other + val:
                log.info("Agent %d: weak-EF1 by giving %r to opp", agent, o)
                return True

    log.info("Agent %d: **violates weak-EF1** (u_self=%s, u_other=%s)", agent, u_self, u_other)
    return False

def round_robin_from_counts(
    counts: Dict[Tuple[Agent, Item], int],   # output of solve_fractional_ILP
    k: int                                   # number of rounds
) -> List[OneDayAllocation]:
    r"""
    Deterministically expand the integer solution of the ILP into *k*
    concrete rounds using a simple round-robin walk.

    Guarantees
    ----------
    * Each item appears exactly the requested number of times.
    * No round contains duplicate copies of the *same* item.

    Examples
    --------
    >>> counts = {(0, 'A'): 2, (1, 'B'): 2}
    >>> round_robin_from_counts(counts, k=2)
    [{0: {'A'}, 1: {'B'}}, {0: {'A'}, 1: {'B'}}]

    >>> counts = {(0, 'x'): 3, (1, 'y'): 3}
    >>> _ = round_robin_from_counts(counts, k=3)   # runs without error
    """
    # ----------------------------------------------------------------------
    # 1) infer the *actual* agent labels   (could be '0','1' if JSON encoded)
    # ----------------------------------------------------------------------
    AGENTS: Set[Agent] = {ag for ag, _ in counts}           # e.g. {'0','1'}

    # ----------------------------------------------------------------------
    # 2) owners[item]  =  [agent, agent, …]  (one entry per requested copy)
    # ----------------------------------------------------------------------
    owners: Dict[Item, List[Agent]] = {}
    for (ag, it), c in counts.items():
        owners.setdefault(it, []).extend([ag] * c)

    # ----------------------------------------------------------------------
    # 3) start with *k* fully-keyed empty rounds
    # ----------------------------------------------------------------------
    rounds: List[OneDayAllocation] = [
        {ag: set() for ag in AGENTS}          # every agent key present
        for _ in range(k)
    ]

    # ----------------------------------------------------------------------
    # 4) place the copies round–robin
    # ----------------------------------------------------------------------
    ptr = 0                                   # “current round” pointer
    for it, owner_list in owners.items():     # deterministic by insertion
        for ag in owner_list:                 # keep ILP order
            for off in range(k):              # probe k rounds cyclically
                r = (ptr + off) % k
                # safe test: is *it* already used in that round?
                if all(it not in bundle for bundle in rounds[r].values()):
                    rounds[r][ag].add(it)
                    ptr = (r + 1) % k        # next copy starts AFTER this one
                    break
            else:
                # Would imply > k copies for the same item – contradicts ILP.
                raise RuntimeError(f"Could not place item {it!r}")

    return rounds




# ---------------------------------------------------------------------------
# 1.  Fractional ILP   (Figure 1 of the paper)
# ---------------------------------------------------------------------------
def solve_fractional_ILP(
    utilities: Dict[Agent, Dict[Item, float]],
    k: int,
    solver: str = "GLPK_MI",
) -> Dict[Tuple[Agent, Item], int]:
    """
    Return counts  x[(i,o)]  that maximise  Σ_{i,o}  u_i(o)·x_{i,o}
    subject to
        (1)  Σ_i x_{i,o}   = k                     ∀ item o
        (2)  0 ≤ x_{i,o} ≤ k                       ∀ i,o
        (3)  Σ_o u_i(o)·x_{i,o} ≥ (k/n)·Σ_o u_i(o) ∀ agent i   (proportionality)

    Works for any number of agents/items; we just use it with n=2 in the
    “Repeated Fair Allocation of Indivisible Items” algorithms.

    doctest:
    >>> utils = {0: {0: 11, 1: 22}, 1: {0: 22, 1: 11}}
    >>> solve_fractional_ILP(utils, 2)
    {(0, 0): 0, (0, 1): 2, (1, 0): 2, (1, 1): 0}
    >>> utils = {0: {0: 11, 1: 22}, 1: {0: 11, 1: 22}}
    >>> solve_fractional_ILP(utils, 2)
    {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    """
    agents = list(utilities)                       # e.g. [0,1]
    items  = list(next(iter(utilities.values())))  # e.g. [0,1,2]
    n, m   = len(agents), len(items)
    log.info("Solving ILP for %d agents, %d items, k=%d", n, m, k)

    # --- build a plain NumPy utility matrix U[i,o] ----------------------------
    U_mat = np.array([[utilities[i][o] for o in items] for i in agents])
    log.debug("Utility matrix U:\n%s", U_mat)

    # --- decision variables ---------------------------------------------------
    # x[i,o] = how many copies of item o go to agent i   (integer in {0,…,k})
    x = cp.Variable((n, m), integer=True)
    log.debug("Decision variables x[i,o]:\n%s", x)

    # --- constraints ----------------------------------------------------------
    constraints = [
        x >= 0,
        x <= k,
    ]

    # (1) every item allocated exactly k times (across all agents)
    constraints.extend(
        cp.sum(x[:, j]) == k  for j in range(m)
    )

    # (3) proportionality for each agent
    for i in range(n):
        total_utility_i = U_mat[i, :].sum()
        constraints.append(
            cp.sum(cp.multiply(U_mat[i, :], x[i, :])) >= (k / n) * total_utility_i
        )

    # --- objective  max Σ_{i,o}  u_i(o)*x_{i,o}  ------------------------------
    objective = cp.Maximize(cp.sum(cp.multiply(U_mat, x)))
    log.debug("Objective function: %s", objective)
    log.debug("Constraints:\n%s", constraints)

    # --- solve ----------------------------------------------------------------
    cp.Problem(objective, constraints).solve(solver=solver)
    log.info("ILP solved: status=%s, objective value=%.2f", cp.Problem(objective, constraints).status, objective.value)

    # --- pack result back into a dictionary -----------------------------------
    return {
        (agents[i], items[j]): int(round(x.value[i, j]))
        for i in range(n) for j in range(m)
    }


# ---------------------------------------------------------------------------
# 2.  Algorithm 1  (n = 2 , k = 2)
# ---------------------------------------------------------------------------

def algorithm1(
    utilities: Dict[Agent, Dict[Item, float]],
) -> List[OneDayAllocation]:
    """
    Compute the 2-round sequence returned by Algorithm 1
    (per-round EF1, Pareto-optimal overall).

    Parameters
    ----------
    utilities  : dict   (2 × m additive utilities)

    Returns
    -------
    π          : list length == 2, each element is {agent → set(items)}


    doctest:
    >>> utils = {0: {0: 10, 1: 1}, 1: {0: 6, 1: 8}}
    >>> allocs = algorithm1(utils)
    >>> allocs[0]  # round 1
    {0: {0}, 1: {1}}
    >>> allocs[1]  # round 2
    {0: {0}, 1: {1}}
    """
    k = 2                      # fixed by the algorithm’s design
    counts = solve_fractional_ILP(utilities, k)
    π1, π2 = round_robin_from_counts(counts, k)
    log.info("Initial allocation π1: %s", π1)
    log.info("Initial allocation π2: %s", π2)

    # --- persistent + optional items --------------------------------------------
    I1 = π1[0] & π2[0]          # always agent 0
    I2 = π1[1] & π2[1]          # always agent 1
    log.debug("Persistent items I1: %s, I2: %s", I1, I2)
    all_items = set(utilities[0])
    O  = all_items - (I1 | I2)  # items distributed once each
    O = {o for o in O if utilities[0][o] != 0 or utilities[1][o] != 0}
    log.debug("Objective items O: %s", O)
    O_plus  = {o for o in O if utilities[0][o] > 0 and utilities[1][o] > 0}
    log.debug("Objective goods O_plus: %s", O_plus)
    O_minus = O - O_plus
    assert O_minus == {o for o in O if utilities[0][o] < 0 or utilities[1][o] < 0}, \
        "O_minus must contain only chores (negative utility for at least one agent)"
    log.debug("Objective chores O_minus: %s", O_minus)
    π1 = {0: I1 | O_minus, 1: I2 | O_plus}
    π2 = {0: I1 | O_plus, 1: I2 | O_minus}
    log.info("Initial π1: %s", π1)
    log.info("Initial π2: %s", π2)
    # --- EF1 checker (mixed goods / chores, exact Def. 2) -----------------------
    def EF1(allocation):
        return EF1_holds(allocation, 0, utilities) and EF1_holds(allocation, 1, utilities)


    # --- swap loop --------------------------------------------------------------
    swap_pool = list(O)
    idx = 0
    while (not EF1(π1) or not EF1(π2)) and idx < len(swap_pool):
        o = swap_pool[idx]
        if o in O_minus:                           # chore-swap
            π1[0].discard(o); π2[1].discard(o)
            π1[1].add(o);     π2[0].add(o)
            log.debug("Chore-swap: %r", o)
        else:                                     # good-swap
            π1[1].discard(o); π2[0].discard(o)
            π1[0].add(o);     π2[1].add(o)
            log.debug("Good-swap: %r", o)
        idx += 1
        log.debug("After swap %d: π1=%s, π2=%s", idx, π1, π2)
    log.info("Final π1: %s", π1)
    log.info("Final π2: %s", π2)
    return [π1, π2]


# ---------------------------------------------------------------------------
# 3.  Algorithm 2  (n = 2 , k even)   
# ---------------------------------------------------------------------------
def algorithm2(
    k: int,
    utilities: Dict[Agent, Dict[Item, float]],
) -> List[OneDayAllocation]:
    """
    Return a k-round sequence that satisfies weak-EF1 in every round
    and is Pareto-optimal overall   (Algorithm 2 from the paper).

    Parameters
    ----------
    k          : even int   (number of rounds)
    utilities  : dict       (usual 2 × m utility matrix)

    Returns
    -------
    π          : List[OneDayAllocation]   length == k


    doctest:
    >>> utils = {0: {0: 10, 1: 1}, 1: {0: 6, 1: 8}}
    >>> allocs = algorithm2(4, utils)  # k=4 rounds
    >>> allocs[0]  # round 1
    {0: {0}, 1: {1}}
    >>> allocs[1]  # round 2
    {0: {0}, 1: {1}}
    >>> allocs[2]  # round 3
    {0: {0}, 1: {1}}
    >>> allocs[3]  # round 4
    {0: {0}, 1: {1}}
    """
    counts = solve_fractional_ILP(utilities, k)

    # ❶ build the initial schedule (shared helper shown earlier)
    π = round_robin_from_counts(counts, k)

    # ---------- helper predicates ---------------------------------------------
    def envy_free(r,a):
        uS = sum(utilities[a][o] for o in π[r][a])
        uO = sum(utilities[a][o] for o in π[r][1-a])
        log.debug("Round %d, agent %d: u_self=%s, u_other=%s", r, a, uS, uO)
        return uS >= uO

    def weak_EF1(r, a):
        return weak_EF1_holds(π[r], a, utilities)
    # ---------- adjustment loop (paper’s pseudo-code) --------------------------
    def adjust(a:int):  
        E = {r for r in range(k) if not envy_free(r,a)}
        log.debug("Adjusting agent %d, initial not envy-free rounds: %s", a, E)
        F = set(range(k)) - E
        log.debug("Envy-free rounds for agent %d: %s", a, F)
        while any(not weak_EF1(r,a) for r in E):
            log.debug("Not envy-free rounds for agent %d: %s", a, E)
            log.debug("Envy-free rounds for agent %d: %s", a, F)
            j = next(r for r in E if not weak_EF1(r,a))
            log.debug("Adjusting round %d for agent %d", j, a)
            while not weak_EF1(j,a):
                i = min(F)
                try:
                    o = next(iter(x for x in π[i][a] - π[j][a] if utilities[a][x] > 0), None)
                    if o is not None:
                        src1,dst1,src2,dst2 = i,j,j,i
                    else:
                        o = next(iter(x for x in π[j][a] - π[i][a] if utilities[a][x] < 0), None)
                        src1,dst1,src2,dst2 = j,i,i,j
                        assert (o is not None), "Lemma 17 violated"
                    log.debug("Found item %r to swap from %d to %d", o, src1, dst1)
                except StopIteration:
                    # no good/chore found – use *any* transferable item
                    if π[j][a] - π[i][a]:
                        o = next(iter(π[j][a] - π[i][a]))
                        src1,dst1,src2,dst2 = j,i,i,j
                    else:
                        o = next(iter(π[i][a] - π[j][a]))
                        src1,dst1,src2,dst2 = i,j,j,i
                # move o
                π[src1][a].remove(o);   π[src1][1-a].add(o)
                π[dst1][1-a].remove(o); π[dst1][a].add(o)

                π[src2][1-a].remove(o); π[src2][a].add(o)
                π[dst2][a].remove(o);   π[dst2][1-a].add(o)
                log.debug("Moved item %r from %d to %d", o, src1, dst1)
                log.debug("After move: π[%d]=%s, π[%d]=%s", src1, π[src1], dst1, π[dst1])

                if not envy_free(i,a):
                    F.remove(i); E.add(i)
                    log.debug("Round %d is now envy-free for agent %d", i, a)

    adjust(0); adjust(1)
    return π


# ---------------------------------------------------------------------------
# 4.  FairPyx adapters
# ---------------------------------------------------------------------------
def algorithm1_div(builder: AllocationBuilder, round_idx: int = 0, **_):
    utils  = builder.instance._valuations
    bundle = algorithm1(utils)[round_idx]             # new call
    builder.give_bundles({a: list(b) for a, b in bundle.items()})

def algorithm2_div(builder: AllocationBuilder, k: int, round_idx: int = 0, **_):
    utils  = builder.instance._valuations
    bundle = algorithm2(k, utils)[round_idx]      # <── new call
    builder.give_bundles({a: list(b) for a, b in bundle.items()})


# ---------------------------------------------------------------------------
# 5.  Self-test (doctest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest, random
    from pprint import pprint

    doctest.testmod(verbose=True)

    seed = random.randint(0, 10000)
    rnd = random.Random(seed)
    print(f"\nRunning self-test with random seed {seed}…")

    k   = 4                                           # number of rounds to test
    for trial in range(20):
        # random 2×6 utilities in [-5,5]
        utils = {
            0: {i: rnd.randint(-5, 5) for i in range(6)},
            1: {i: rnd.randint(-5, 5) for i in range(6)},
        }
        print(f"\nTrial {trial+1} / 20: k={k}, utilities:")
        pprint(utils)
        print("=== Algorithm 1 (EF1 + PO overall) ===")
        for r in range(2):
            alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
            print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
            assert EF1_holds(alloc, 0, utils), f"EF1 violated in trial {trial} (round {r+1})"
            assert EF1_holds(alloc, 1, utils), f"EF1 violated in trial {trial} (round {r+1})"
        print("=== Algorithm 2 (weak-EF1 + PO overall) ===")
        for r in range(k):
            alloc = divide(algorithm2_div, valuations=utils, round_idx=r, k=k)
            print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
            assert weak_EF1_holds(alloc, 0, utils), f"Weak-EF1 violated in trial {trial} (round {r+1})"
            assert weak_EF1_holds(alloc, 1, utils), f"Weak-EF1 violated in trial {trial} (round {r+1})"
        print ("✓  Trial passed")
    print("\nAll trials passed")
    print("✓  Module self-tests passed")
