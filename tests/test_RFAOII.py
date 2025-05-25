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
)

# ─── GLOBAL SEED ─────────────────────────────────────────────────────────────────────────
SEED = 42
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


def test_algorithm1_simple_goods_divide():
    utils = simple_utilities()
    print("\nUtilities:", utils)
    items = set(utils[0])

    print("\n=== Algorithm1 (simple goods) — Both Rounds ===")
    for r in (0, 1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        # coverage
        assert set(alloc[0]) | set(alloc[1]) == items

        # EF1 checks
        print("  EF1 checks:")
        for a in (0,1):
            u_self   = sum(utils[a][o] for o in alloc[a])
            u_other  = sum(utils[a][o] for o in alloc[1-a])
            max_item = max(utils[a].values())
            ok       = u_self >= u_other - max_item
            print(f"   Agent {a}: {u_self} ≥ {u_other} - {max_item}? → {ok}")
            assert ok


def test_algorithm1_mixed_goods_chores_divide():
    utils = mixed_utilities()
    print("\nUtilities:", utils)
    items = set(utils[0])

    print("\n=== Algorithm1 (mixed goods & chores) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == items

        # EF1 (remove worst) checks
        print("  EF1 (remove worst) checks:")
        for a in (0,1):
            u_self  = sum(utils[a][o] for o in alloc[a])
            u_other = sum(utils[a][o] for o in alloc[1-a])
            worst   = min(utils[a][o] for o in alloc[1-a])
            ok      = u_self >= u_other - worst
            print(f"   Agent {a}: {u_self} ≥ {u_other} - {worst}? → {ok}")
            assert ok


@pytest.mark.parametrize("utils_fn", [mixed_utilities, chores_only_utilities])
def test_algorithm1_second_round_divide(utils_fn):
    # sanity check you can directly ask for round 2
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

    # Skip if pure chores-only (EF1 not guaranteed there)
    if all(v <= 0 for v in utils[0].values()) and all(v <= 0 for v in utils[1].values()):
        pytest.skip("pure chores-only; skipping Algorithm1 EF1 test")

    print("\n=== Algorithm1 (random) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])

        print("  EF1 checks:")
        for a in (0,1):
            u_self   = sum(utils[a][o] for o in alloc[a])
            u_other  = sum(utils[a][o] for o in alloc[1-a])
            max_item = max(utils[a].values())
            ok       = u_self >= u_other - max_item
            print(f"   Agent {a}: {u_self} ≥ {u_other} - {max_item}? → {ok}")
            assert ok


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
    for r, alloc in enumerate(rounds,1):
        print(f"\n Round {r}:")
        for a in (0,1):
            u_self  = sum(utils[a][o] for o in alloc[a])
            u_other = sum(utils[a][o] for o in alloc[1-a])
            if u_self >= u_other:
                print(f"  Agent {a}: no envy ({u_self} ≥ {u_other})")
            else:
                print(f"  Agent {a}: envies ({u_self} < {u_other})")
                fixed = False
                for o in alloc[1-a]:
                    ok = u_self >= u_other - utils[a][o]
                    print(f"   Removing item {o}: {ok}")
                    if ok:
                        fixed = True
                for o in alloc[a]:
                    ok = u_self + utils[a][o] >= u_other
                    print(f"   Adding item {o}: {ok}")
                    if ok:
                        fixed = True
                assert fixed


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
    for r, alloc in enumerate(rounds,1):
        print(f"\n Round {r}:")
        for a in (0,1):
            u_self  = sum(utils[a][o] for o in alloc[a])
            u_other = sum(utils[a][o] for o in alloc[1-a])
            if u_self >= u_other:
                print(f"  Agent {a}: no envy ({u_self} ≥ {u_other})")
            else:
                print(f"  Agent {a}: envies ({u_self} < {u_other})")
                fixed = False
                for o in alloc[1-a]:
                    ok = u_self >= u_other - utils[a][o]
                    print(f"   Removing item {o}: {ok}")
                    if ok:
                        fixed = True
                for o in alloc[a]:
                    ok = u_self + utils[a][o] >= u_other
                    print(f"   Adding item {o}: {ok}")
                    if ok:
                        fixed = True
                assert fixed




def test_algorithm2_random():
    utils = random_utilities(num_items=5, seed=SEED)
    print(f"\nUtilities (random seed={SEED}): {utils}")
    k = 4

    print("\n=== Algorithm2 (random goods/chores) — All Rounds ===")
    for r in range(k):
        alloc = divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])

        u0 = sum(utils[0][o] for o in alloc[0])
        u1 = sum(utils[1][o] for o in alloc[1])
        if u0 >= u1:
            print(f"  Agent 0: no envy ({u0} ≥ {u1})")
        else:
            print(f"  Agent 0: envies ({u0} < {u1})")
            fixed = False
            for o in alloc[1]:
                ok = u0 >= u1 - utils[0][o]
                print(f"   Removing item {o}: {ok}")
                if ok:
                    fixed = True
            for o in alloc[0]:
                ok = u0 + utils[0][o] >= u1
                print(f"   Adding item {o}: {ok}")
                if ok:
                    fixed = True
            assert fixed


def test_algorithm1_larger_random():
    utils = random_utilities(num_items=20, seed=SEED+10)
    print(f"\nUtilities (larger random seed={SEED+10}): {utils}")
    # skip pure-chores
    if all(v <= 0 for v in utils[0].values()) and all(v <= 0 for v in utils[1].values()):
        pytest.skip("pure chores-only; skipping Algorithm1 EF1 test")

    print("\n=== Algorithm1 (larger random) — Both Rounds ===")
    for r in (0,1):
        alloc = divide(algorithm1_div, valuations=utils, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])

        print("  EF1 checks:")
        for a in (0,1):
            u_self   = sum(utils[a][o] for o in alloc[a])
            u_other  = sum(utils[a][o] for o in alloc[1-a])
            max_item = max(utils[a].values())
            ok       = u_self >= u_other - max_item
            print(f"   Agent {a}: {u_self} ≥ {u_other} - {max_item}? → {ok}")
            assert ok


def test_algorithm2_larger_random():
    utils = random_utilities(num_items=20, seed=SEED+20)
    print(f"\nUtilities (larger random seed={SEED+20}): {utils}")
    k = 8
    rounds = []

    print("\n=== Algorithm2 (larger random) — All Rounds ===")
    for r in range(k):
        alloc = divide(algorithm2_div, valuations=utils, k=k, round_idx=r)
        print(f"\n Round {r+1}: 0→{alloc[0]} | 1→{alloc[1]}")
        assert set(alloc[0]) | set(alloc[1]) == set(utils[0])
        rounds.append(alloc)

    print("\nWeak-EF1 checks per round:")
    for r, alloc in enumerate(rounds,1):
        print(f"\n Round {r}:")
        for a in (0,1):
            u_self  = sum(utils[a][o] for o in alloc[a])
            u_other = sum(utils[a][o] for o in alloc[1-a])
            if u_self >= u_other:
                print(f"  Agent {a}: no envy ({u_self} ≥ {u_other})")
            else:
                print(f"  Agent {a}: envies ({u_self} < {u_other})")
                fixed = False
                for o in alloc[1-a]:
                    ok = u_self >= u_other - utils[a][o]
                    print(f"   Removing item {o}: {ok}")
                    if ok:
                        fixed = True
                for o in alloc[a]:
                    ok = u_self + utils[a][o] >= u_other
                    print(f"   Adding item {o}: {ok}")
                    if ok:
                        fixed = True
                assert fixed
