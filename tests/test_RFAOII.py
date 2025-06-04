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
    weak_EF1_holds
)

# ─── GLOBAL SEED ─────────────────────────────────────────────────────────────────────────
SEED = 4941
### TODO: choose seed at random, and print it for debug
# ──────────────────────────────────────────────────────────────────────────────────────────

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

### TODO: test allocation genetated from the ILP

def test_algorithm1_simple_goods_divide():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    items = set(utils[0])

    print("\n=== Algorithm1 (simple goods) — Both Rounds ===")

    ### TODO: in this function, test algorithm 1 directly.
    for r in (0, 1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        # coverage
        assert set(alloc[0]) | set(alloc[1]) == items

        # EF1 checks
        print("EF1 checks:")
        for a in (0, 1):
            assert EF1_holds(alloc, a, utils)

    ### TODO: move common test code to a separate function
    ### TODO: add check for proportionality / EF overall 

def test_algorithm1_mixed_goods_chores_divide():
    utils = mixed_utilities()
    print("\nUtilities:", utils)
    items = set(utils[0])

    print("\n=== Algorithm1 (mixed goods & chores) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == items

        print("EF1 checks:")
        for a in (0, 1):
            assert EF1_holds(alloc, a, utils)


@pytest.mark.parametrize("utils_fn", [mixed_utilities, chores_only_utilities])
def test_algorithm1_second_round_divide(utils_fn):
    utils = utils_fn()
    print("\nUtilities:", utils)
    alloc2 = divide(algorithm1_div, valuations=utils, round_idx=1)
    print("\n=== Algorithm1 — Direct Round 2 ===")
    print(f" Agent 0 → {alloc2[0]}")
    print(f" Agent 1 → {alloc2[1]}")
    items = set(utils[0])
    assert set(alloc2[0]) | set(alloc2[1]) == items


def test_algorithm1_random():
    utils = random_utilities(num_items=5, seed=SEED)
    print(f"\nUtilities (random seed={SEED}): {utils}")


    print("\n=== Algorithm1 (random) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])

        print("  EF1 checks:")
        for a in (0, 1):
            assert EF1_holds(alloc, a, utils)


def test_algorithm2_simple_goods_divide():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    k = 4
    rounds = []
    

    print("\n=== Algorithm2 (simple goods) — All Rounds ===")
    for r in range(k):
        alloc = divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])
        rounds.append(alloc)

    print("\nWeak-EF1 checks per round:")
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert weak_EF1_holds(alloc, a, utils), \
                f"Round {r} – agent {a} fails weak-EF1"


def test_algorithm2_needs_swap():
    print("\n=== Algorithm2 (needs swap) ===")
    """Deliberately craft utilities so that Algorithm 2 must move an item
    between the two rounds to satisfy weak-EF1."""
    utils = {
        0: {0: 8,  1: 8,  2: 1, 3: 1},
        1: {0: 10, 1: 10, 2: 5, 3: 5},
    }
    k = 2
    rounds = [divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
              for r in range(k)]

    # show what we got
    for r, alloc in enumerate(rounds, 1):
        print(f"Round {r}: A0→{sorted(alloc[0])} | A1→{sorted(alloc[1])}")

    # they must not be identical
    assert rounds[0] != rounds[1], "Algorithm 2 should perform a swap here"

    print("\nWeak-EF1 checks per round:")
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert weak_EF1_holds(alloc, a, utils), \
                f"Round {r} – agent {a} fails weak-EF1"




def test_algorithm2_random():
    utils = random_utilities(num_items=5, seed=SEED)
    print(f"\nUtilities (random seed={SEED}): {utils}")
    k = 4
    rounds = []

    print("\n=== Algorithm2 (random goods/chores) — All Rounds ===")
    for r in range(k):
        alloc = divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])
        rounds.append(alloc)

    print("\nWeak-EF1 checks per round:")
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert weak_EF1_holds(alloc, a, utils), \
                f"Round {r} – agent {a} fails weak-EF1"


def test_algorithm1_larger_random():
    utils = random_utilities(num_items=20, seed=SEED+10)
    print(f"\nUtilities (larger random seed={SEED+10}): {utils}")

    print("\n=== Algorithm1 (larger random) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])

        print("  EF1 checks:")
        for a in (0, 1):
            assert EF1_holds(alloc, a, utils)

def test_algorithm2_larger_random():
    utils = random_utilities(num_items=20, seed=SEED + 20)
    print(f"\nUtilities (larger random seed={SEED+20}): {utils}")
    k = 8
    rounds = []

    print("\n=== Algorithm2 (larger random) — All Rounds ===")
    for r in range(k):
        alloc = divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        # coverage
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])
        rounds.append(alloc)


    print("\nWeak-EF1 checks per round:")
    for r, alloc in enumerate(rounds, 1):
        print(f"\nRound {r}:")
        for a in (0, 1):
            assert weak_EF1_holds(alloc, a, utils), \
                f"Round {r} – agent {a} fails weak-EF1"



