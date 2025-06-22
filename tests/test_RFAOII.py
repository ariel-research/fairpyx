"""	
Tests for the Repeated Fair Allocation of Indivisible Items algorithms.
Programmer: Shaked Shvartz
Since: 2025-05
"""

import pytest
import random

from fairpyx.adaptors import divide
from fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items import (
    solve_fractional_ILP,
    algorithm1_div,
    algorithm2_div,
    EF1_holds,
    weak_EF1_holds,
    OneDayAllocation,
    Agent,
    Item,
    Dict,
    List,
    round_robin_from_counts,
    algorithm1,
    algorithm2,
)


def overall_EF_holds(rounds: List[OneDayAllocation],
                     utilities: Dict[Agent, Dict[Item, float]]) -> bool:
    """
    True iff the union of all k rounds is envy-free overall:
      let B_i = ∪_{r} rounds[r][i]. Then for both i=0,1:
         sum_{o in B_i} u_i(o) >= sum_{o in B_j} u_i(o).
    """
    n = len(utilities)
    # build each agent's union‐bundle
    union = {i:set() for i in utilities}
    for r in rounds:
        for i, b in r.items():
            union[i].update(b)
    ok = True
    for i in utilities:
        u_i = sum(utilities[i][o] for o in union[i])
        u_j = sum(utilities[i][o] for o in union[1-i])
        if u_i < u_j:
            return False
    return True

def overall_PROP_holds(
    rounds: List[OneDayAllocation],
    utilities: Dict[Agent, Dict[Item, float]]
) -> bool:
    """
    True iff the *union* allocation over all rounds is proportional (Def. 3).
    I.e. each agent i gets at least (k/n)*sum_{o∈I} u_i(o).
    """
    n = len(utilities)
    # sum of one copy of every item
    items = list(utilities[next(iter(utilities))].keys())
    total_per_copy = {i: sum(utilities[i][o] for o in items)
                      for i in utilities}
    # sum over all rounds
    k = len(rounds)
    got = {i: 0.0 for i in utilities}
    for r in rounds:
        for i in utilities:
            got[i] += sum(utilities[i][o] for o in r[i])

    for i in utilities:
        if got[i] + 1e-8 < (k / n) * total_per_copy[i]:
            return False
    return True



def overall_checks(rounds: List[OneDayAllocation],
                     utilities: Dict[Agent, Dict[Item, float]],
                     solver: str = "GLPK_MI") -> None:
    """
    Run all overall checks on the k rounds.
    """
    assert overall_EF_holds(rounds, utilities), \
        "Overall EF check failed"
    assert overall_PROP_holds(rounds, utilities), \
        "Overall PROP check failed"

def ef1_round_checks(
    rounds: List[OneDayAllocation],
    utilities: Dict[Agent, Dict[Item, float]]
) -> None:
    """
    Run all EF1 checks on the k rounds.
    """
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert EF1_holds(alloc, a, utilities), \
                f"Round {r} – agent {a} fails EF1"
            print(f"  Agent {a} holds EF1 in round {r}")
    print("All rounds passed EF1 checks.")


def weak_ef1_round_checks(
    rounds: List[OneDayAllocation],
    utilities: Dict[Agent, Dict[Item, float]]
) -> None:
    """
    Run all weak-EF1 checks on the k rounds.
    """
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert weak_EF1_holds(alloc, a, utilities), \
                f"Round {r} – agent {a} fails weak-EF1"
            print(f"  Agent {a} holds weak-EF1 in round {r}")
    print("All rounds passed weak-EF1 checks.")

def get_round_allocs(algo_div, utils, k):
    """
    Run `divide(algo_div, …)` for round_idx = 0..k-1 and return the
    list of allocations.
    """
    return [
        divide(algo_div, valuations=utils, k=k, round_idx=r)
        for r in range(k)
    ]


def check_coverage_and_print(rounds, utils):
    """
    For each round-allocation in `rounds`, verify that every item
    appears exactly once (i.e. coverage), and print the bundles.
    """
    items = set(utils[0])
    for r, alloc in enumerate(rounds, 1):
        print(f"\n Round {r}: 0→{sorted(alloc[0])} | 1→{sorted(alloc[1])}")
        assert set(alloc[0]) | set(alloc[1]) == items
    return rounds
 

 # ---------------------------------------------------------------------------------------------------------------------

from random import randint
SEED = randint(0, 10000)
print(f"Running tests with random seed: {SEED}")

def simple_utilities():
    return {
        0: {0: 1.0, 1: 2.0, 2: 5.0},
        1: {0: 5.0, 1: 1.0, 2: 2.0},
    }

def mixed_utilities():
    return {
        0: {0: 3.0, 1: -2.0, 2: 1.0},
        1: {0: 1.0, 1: -3.0, 2: 2.0},
    }

def chores_only_utilities():
    return {
        0: {0: -1.0, 1: -2.0, 2: -5.0},
        1: {0: -5.0, 1: -1.0, 2: -2.0},
    }

def random_utilities(num_items: int, seed: int):
    """
    Two agents, num_items items, utilities drawn uniformly from -5..5.
    """
    rnd = random.Random(seed)
    return {
        0: {i: float(rnd.randint(-5,5)) for i in range(num_items)},
        1: {i: float(rnd.randint(-5,5)) for i in range(num_items)},
    }


def test_round_robin_from_counts():
    utils = simple_utilities()
    k = 2
    counts = solve_fractional_ILP(utils, k)
    rounds = round_robin_from_counts(counts, k)
    overall_checks(rounds, utils)

def test_round_robin_from_counts_random():
    """Check EF overall for a random 2-agent, k=4 round-robin from ILP counts."""
    utils = random_utilities(num_items=5, seed=SEED)
    k = 4
    counts = solve_fractional_ILP(utils, k)
    rounds = round_robin_from_counts(counts, k)
    overall_checks(rounds, utils)


def test_solve_fractional_ILP_counts_sum_to_k():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    k = 2
    counts = solve_fractional_ILP(utils, k)
    print("\n=== ILP Counts (k=2) ===")
    for o in sorted(utils[0]):
        total = sum(counts[(i, o)] for i in utils)
        print(f" Item {o}: allocated {total} times")
        assert total == k

def test_algorithm1():
    """
    run algorithm1 without divide, just to check the basic functionality.
    """
    utils = simple_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm1 (without divide) ===")
    alloc = algorithm1(utils)
    print ("\nalloc:", alloc)
    assert alloc[0] == {0: {1, 2}, 1: {0}}
    assert alloc[1] == {0: {1, 2}, 1: {0}}


def test_algorithm2():
    """
    run algorithm2 without divide, just to check the basic functionality.
    """
    utils = simple_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm2 (without divide) ===")
    alloc = algorithm2(6, utils)
    print ("\nalloc:", alloc)
    
        

def test_algorithm1_simple_goods_divide():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm1 (simple goods) — Both Rounds ===")
    rounds = get_round_allocs(algorithm1_div, utils, k=2)
    check_coverage_and_print(rounds, utils)
    ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm1_mixed_goods_chores_divide():
    utils = mixed_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm1 (mixed goods & chores) — Both Rounds ===")
    rounds = get_round_allocs(algorithm1_div, utils, k=2)
    check_coverage_and_print(rounds, utils)
    ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm1_chores_only_divide():
    utils = chores_only_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm1 (chores only) — Both Rounds ===")
    rounds = get_round_allocs(algorithm1_div, utils, k=2)
    check_coverage_and_print(rounds, utils)
    ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm1_random_divide():
    utils = random_utilities(num_items=5, seed=SEED)
    print(f"\nUtilities (random seed={SEED}): {utils}")
    rounds = get_round_allocs(algorithm1_div, utils, k=2)
    print("\n=== Algorithm1 (random) — Both Rounds ===")
    check_coverage_and_print(rounds, utils)
    ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm2_simple_goods_divide():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    print("\n=== Algorithm2 (simple goods) — All Rounds ===")
    rounds = get_round_allocs(algorithm2_div, utils, k=4)
    check_coverage_and_print(rounds, utils)
    weak_ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm2_needs_swap_divide():
    print("\n=== Algorithm2 (needs swap) ===")
    """Deliberately craft utilities so that Algorithm 2 must move an item
    between the two rounds to satisfy weak-EF1."""
    utils = {
        0: {0: 8,  1: 8,  2: 1, 3: 1},
        1: {0: 10, 1: 10, 2: 5, 3: 5},
    }
    print(f"\nUtilities: {utils}")
    rounds = get_round_allocs(algorithm2_div, utils, k=2)
    check_coverage_and_print(rounds, utils)
    weak_ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)

def test_algorithm2_random_divide():
    utils = random_utilities(num_items=5, seed=SEED)
    print(f"\nUtilities (random seed={SEED}): {utils}")
    print("\n=== Algorithm2 (random) — All Rounds ===")
    rounds = get_round_allocs(algorithm2_div, utils, k=4)
    check_coverage_and_print(rounds, utils)
    weak_ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)


def test_algorithm1_larger_random_divide():
    utils = random_utilities(num_items=20, seed=SEED+10)
    print(f"\nUtilities (larger random seed={SEED+10}): {utils}")
    print("\n=== Algorithm1 (larger random) — Both Rounds ===")
    rounds = get_round_allocs(algorithm1_div, utils, k=2)
    check_coverage_and_print(rounds, utils)
    ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)


def test_algorithm2_larger_random_divide():
    utils = random_utilities(num_items=20, seed=SEED + 20)
    print(f"\nUtilities (larger random seed={SEED+20}): {utils}")
    print("\n=== Algorithm2 (larger random) — All Rounds ===")
    rounds = get_round_allocs(algorithm2_div, utils, k=8)
    check_coverage_and_print(rounds, utils)
    weak_ef1_round_checks(rounds, utils)
    overall_checks(rounds, utils)





