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
from   fairpyx.adaptors import AllocationBuilder

Agent = int
Item  = int
Bundle = Dict[Agent, Set[Item]]



# ------------------------------------------------------------------
# 0-ter.  Public fairness predicates (Definition 2 and weak-EF1)
# ------------------------------------------------------------------
def _to_set(x):
    """Ensure we work with a real set."""
    return x if isinstance(x, set) else set(x)


def EF1_holds(bundle: Bundle,
              agent: Agent,
              utilities: Dict[Agent, Dict[Item, float]]) -> bool:
    """
    True  ⇔  allocation `bundle` satisfies EF1 for `agent`
    (Def. 2 in the paper).

    Prints why it holds / fails and – if relevant – which item `o`
    certifies EF1.
    """
    A = _to_set(bundle[agent])
    B = _to_set(bundle[1 - agent])

    u_self  = sum(utilities[agent][o] for o in A)
    u_other = sum(utilities[agent][o] for o in B)

    if u_self >= u_other:                                      # envy-free
        print(f"     Agent {agent}: no envy  (u_self={u_self} ≥ u_other={u_other})")
        return True

    for o in A | B:                                            # Definition 2
        u_self_after  = u_self  - (utilities[agent][o] if o in A else 0)
        u_other_after = u_other - (utilities[agent][o] if o in B else 0)
        if u_self_after >= u_other_after:
            side = "own" if o in A else "opp"
            print(f"     Agent {agent}: EF1 by removing {side} item {o!r}")
            return True

    print(f"     Agent {agent}: **violates EF1**  (u_self={u_self}, u_other={u_other})")
    return False


def weak_EF1_holds(bundle: Bundle,
                   agent: Agent,
                   utilities: Dict[Agent, Dict[Item, float]]) -> bool:
    """
    Returns True  ⇔  the allocation `bundle` is weak-EF1 for `agent`
    according to Definition 6 in the paper.

    It prints the witness that certifies weak-EF1:
      * “no envy”, or
      * the item o with the direction (add / remove) that makes the
        inequality of Definition 6 hold.
    """
    A = _to_set(bundle[agent])          # π_i
    B = _to_set(bundle[1 - agent])      # π_j

    u_self  = sum(utilities[agent][o] for o in A)   # u_i(π_i)
    u_other = sum(utilities[agent][o] for o in B)   # u_i(π_j)

    # 0) envy-free already ----------------------------------------------------
    if u_self >= u_other:
        print(f"     Agent {agent}: no envy  (u_self={u_self} ≥ u_other={u_other})")
        return True

    # 1) look for an item that fixes the envy via Definition 6 ---------------
    #
    #    o ∈ π_i ∪ π_j  such that
    #       u_i(π_i ∪ {o}) ≥ u_i(π_j \ {o})      (case ADD-to-self / REMOVE-from-other)
    #       OR
    #       u_i(π_i \ {o}) ≥ u_i(π_j ∪ {o})      (case REMOVE-from-self / ADD-to-other)
    #
    for o in A | B:
        val = utilities[agent][o]

        if o in B:                   # o currently in opponent's bundle
            # Case (ADD to self, REMOVE from other)
            if u_self + val >= u_other - val:
                print(f"     Agent {agent}: weak-EF1 by taking {o!r} from opp")
                return True
        else:                        # o in own bundle
            # Case (REMOVE from self, ADD to other)
            if u_self - val >= u_other + val:
                print(f"     Agent {agent}: weak-EF1 by giving {o!r} to opp")
                return True

    # 2) nothing works --------------------------------------------------------
    print(f"     Agent {agent}: **violates weak-EF1**  (u_self={u_self}, u_other={u_other})")
    return False


# ---------------------------------------------------------------------------
# 1.  Fractional ILP   (Figure 1 of the paper)
# ---------------------------------------------------------------------------
def solve_fractional_ILP(
        utilities: Dict[Agent, Dict[Item, float]],
        k: int,
        solver: str = "GLPK_MI"
) -> Dict[Tuple[Agent, Item], int]:
    """
    Solve the proportional-and-PO ILP from the article (Fig. 1).

    Returns a dict  (agent,item) ↦ count  with ∑_agents count = k for every item.

    >>> utils  = {0:{0:1, 1:2}, 1:{0:2, 1:1}}
    >>> solve_fractional_ILP(utils, 2)
    {(0, 0): 0, (0, 1): 2, (1, 0): 2, (1, 1): 0}
    """
    agents = list(utilities)
    items  = list(next(iter(utilities.values())))
    n, m   = len(agents), len(items)

    x  = cp.Variable((n, m), integer=True)
    cons = [x >= 0, x <= k]

    # each item allocated k times
    for j in range(m):
        cons.append(cp.sum(x[:, j]) == k)

    # proportionality for every agent
    for i in range(n):
        u_vec = [utilities[agents[i]][it] for it in items]
        total = sum(u_vec)
        cons.append(cp.reshape(x[i, :], (1, m), order='C') @ u_vec >= (k/n) * total)

    # maximise social welfare (→ PO)
    U = [utilities[a][it] for a in agents for it in items]
    obj = cp.Maximize(cp.reshape(x, (n*m,), order='C') @ U)

    cp.Problem(obj, cons).solve(solver=solver)

    return {(agents[i], items[j]): int(round(x.value[i, j]))
            for i in range(n) for j in range(m)}


# ---------------------------------------------------------------------------
# 2.  Algorithm 1  (n = 2 , k = 2)
# ---------------------------------------------------------------------------
def algorithm1(initial_alloc: List[Bundle],
               utilities: Dict[Agent, Dict[Item, float]]) -> List[Bundle]:
    """
    Two-round EF1 sequence (Algorithm 1).
    """
    k = 2
    counts = solve_fractional_ILP(utilities, k)

    # --- split counts into two preliminary rounds --------------------------------
    prelim = [{0:set(), 1:set()} for _ in range(k)]
    for (agent, item), c in counts.items():
        for r in range(c):
            prelim[r][agent].add(item)
    π1, π2 = prelim

    # --- persistent + optional items --------------------------------------------
    I1 = π1[0] & π2[0]          # always agent 0
    I2 = π1[1] & π2[1]          # always agent 1
    all_items = set(utilities[0])
    O  = all_items - (I1 | I2)  # items distributed once each

    O_plus  = {o for o in O if utilities[0][o] >= 0 and utilities[1][o] >= 0}
    O_minus = O - O_plus

    π1 = {0: I1 | O_minus, 1: I2 | O_plus}
    π2 = {0: I1 | O_plus, 1: I2 | O_minus}

    # --- EF1 checker (mixed goods / chores, exact Def. 2) -----------------------
    def EF1(bundle):
        return EF1_holds(bundle, 0, utilities) and EF1_holds(bundle, 1, utilities)


    # --- swap loop --------------------------------------------------------------
    swap_pool = list(O)
    idx = 0
    while (not EF1(π1) or not EF1(π2)) and idx < len(swap_pool):
        o = swap_pool[idx]
        if o in O_minus:                           # chore-swap
            π1[0].discard(o); π2[1].discard(o)
            π1[1].add(o);     π2[0].add(o)
        else:                                     # good-swap
            π1[1].discard(o); π2[0].discard(o)
            π1[0].add(o);     π2[1].add(o)
        idx += 1

    return [π1, π2]


# ---------------------------------------------------------------------------
# 3.  Algorithm 2  (n = 2 , k even)   
# ---------------------------------------------------------------------------
def algorithm2(initial_alloc: List[Bundle],
               utilities: Dict[Agent, Dict[Item, float]]) -> List[Bundle]:
    """Return k-round weak-EF1 sequence (Algorithm 2)."""
    k       = len(initial_alloc)
    counts  = solve_fractional_ILP(utilities, k)

    # ---------- Build initial π via round-robin  -------------------------------
    π = [{0:set(), 1:set()} for _ in range(k)]
    occ: Dict[Item, List[int]] = {o:[] for o in utilities[0]}
    for (ag,it), c in counts.items():
        occ[it].extend([ag]*c)

    rr = 0
    for it, owners in occ.items():
        for ag in owners:
            for off in range(k):
                r = (rr + off) % k
                if it not in π[r][0] and it not in π[r][1]:
                    π[r][ag].add(it)
                    rr = (rr + 1) % k
                    break
            else:
                raise RuntimeError("could not place item")

    # ---------- helper predicates ---------------------------------------------
    def envy_free(r,a):
        uS = sum(utilities[a][o] for o in π[r][a])
        uO = sum(utilities[a][o] for o in π[r][1-a])
        return uS >= uO

    def weak_EF1(r, a):
        return weak_EF1_holds(π[r], a, utilities)
    # ---------- adjustment loop (paper’s pseudo-code) --------------------------
    def adjust(a:int):
        E = {r for r in range(k) if not envy_free(r,a)}
        F = set(range(k)) - E
        while any(not weak_EF1(r,a) for r in E):
            j = next(r for r in E if not weak_EF1(r,a))
            while not weak_EF1(j,a):
                i = min(F)
                # pick o as in Lemma-17
                try:
                    if any(utilities[a][x] > 0 for x in π[j][a] - π[i][a]):
                        o = next(x for x in π[j][a]-π[i][a] if utilities[a][x] > 0)
                        src1,dst1,src2,dst2 = j,i,i,j
                    else:
                        o = next(x for x in π[i][a]-π[j][a] if utilities[a][x] < 0)
                        src1,dst1,src2,dst2 = i,j,j,i
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

                if not envy_free(i,a):
                    F.remove(i); E.add(i)

    adjust(0); adjust(1)
    return π


# ---------------------------------------------------------------------------
# 4.  FairPyx adapters
# ---------------------------------------------------------------------------
def algorithm1_div(builder: AllocationBuilder, round_idx:int=0, **_):
    utils  = builder.instance._valuations
    bundle = algorithm1([{},{}], utils)[round_idx]
    builder.give_bundles({a:list(b) for a,b in bundle.items()})

def algorithm2_div(builder: AllocationBuilder, k:int, round_idx:int=0, **_):
    utils  = builder.instance._valuations
    init   = [{} for _ in range(k)]
    bundle = algorithm2(init, utils)[round_idx]
    builder.give_bundles({a:list(b) for a,b in bundle.items()})


# ---------------------------------------------------------------------------
# 5.  Self-test (doctest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest, random
    from pprint import pprint

    doctest.testmod(verbose=True)

    rnd = random.Random(2025)
    k   = 4                                           # number of rounds to test
    for trial in range(20):
        # random 2×6 utilities in [-5,5]
        utils = {
            0: {i: rnd.randint(-5, 5) for i in range(6)},
            1: {i: rnd.randint(-5, 5) for i in range(6)},
        }

        rounds = algorithm2([{} for _ in range(k)], utils)

        ok = all(
            weak_EF1_holds(rounds[r], a, utils)
            for r in range(k)
            for a in (0, 1)
        )
        assert ok, f"Weak-EF1 violated in trial {trial}\nutilities = {pprint(utils)}"

    print("✓  Module self-tests passed")
