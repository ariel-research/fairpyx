"""
Test the Fair Division Under Cardinality Constraints algorithm.

Programmer: Sapir Dahan
Since:  2026-04
"""

import math
import pytest
import numpy as np
import networkz as nz
from fairpyx import Instance, AllocationBuilder, divide
from fairpyx.algorithms.fair_division_under_cardinality_constraints import (
    fair_division_under_cardinality_constraints,
    greedy_round_robin,
    eliminate_envy_cycles,
    validate_fair_division_inputs,
)

NUM_OF_RANDOM_INSTANCES = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_ef1(result, valuations):
    """
    Return True iff the allocation satisfies EF1 (Envy-Free up to one good).

    EF1 definition: for every pair of agents (i, j), there EXISTS some good g in j's
    bundle such that when g is removed, i no longer envies j:
        v_i(result[i]) >= v_i(result[j] \\ {g})

    To check whether such a g exists, we look at the good in j's bundle that i values
    MOST. Removing the highest-valued item causes the largest drop in v_i(result[j]),
    giving us the best possible chance of eliminating the envy. If even removing that
    item is not enough, then no single removal can fix the envy and EF1 fails.

    Formally: EF1 holds for pair (i, j) iff
        v_i(result[i]) >= v_i(result[j]) - max_{g in result[j]} v_i(g)
    """
    agents = list(result.keys())
    for i in agents:
        # How much agent i values their own bundle
        val_i_own = sum(valuations[i].get(g, 0) for g in result[i])
        for j in agents:
            if i == j:
                continue
            # How much agent i values j's bundle
            val_i_other = sum(valuations[i].get(g, 0) for g in result[j])
            if val_i_other <= val_i_own:
                continue  # i does not envy j at all — EF1 trivially holds for this pair

            # i envies j, check EF1: can removing one good from j's bundle fix it?
            if not result[j]:
                # j has an empty bundle but i envies them — logically impossible, skip
                continue

            # The best single removal: take away the item i values most in j's bundle.
            # If val_i_own >= val_i_other - max_val, envy disappears → EF1 holds for (i, j).
            # If not, no single removal helps → EF1 is violated.
            max_val = max(valuations[i].get(g, 0) for g in result[j])
            if val_i_own < val_i_other - max_val:
                return False  # EF1 violated: even removing j's best item is not enough

    return True


def check_cardinality_constraints(result, item_categories, category_capacities):
    """
    Return True iff every agent's bundle respects the per-category threshold k_h.

    For each agent and each category h, count how many goods from that category the
    agent received. This count must not exceed k_h (the shared capacity for category h).
    """
    for agent, bundle in result.items():
        for category, items in item_categories.items():
            # Count how many goods from this category the agent received
            count = sum(1 for g in bundle if g in items)
            # k_h is the maximum allowed goods per agent from this category
            if count > category_capacities[category]:
                return False  # agent received too many goods from this category
    return True


def random_valid_instance(num_agents, num_items, num_categories, seed):
    """
    Generate a valid random instance for fair_division_under_cardinality_constraints.
    Returns (valuations, item_categories, category_capacities, agent_order).

    All generated instances satisfy the algorithm's preconditions:
      - goods are partitioned into exactly num_categories non-empty categories
      - each threshold k_h >= ceil(|C_h| / n)  (feasibility condition from the paper)
      - all valuations are non-negative integers
    """
    rng = np.random.default_rng(seed)

    # Build agent and good name lists.
    # Example: num_agents=3, num_items=6 → agents=['a1','a2','a3'], goods=['g1',...,'g6']
    agents = [f"a{i}" for i in range(1, num_agents + 1)]
    goods = [f"g{i}" for i in range(1, num_items + 1)]

    # --- Partition goods into num_categories non-empty categories ---
    # We want each category to contain a random subset of goods, not just a prefix.
    # Strategy: shuffle the goods list first, then cut it at (num_categories-1) random
    # positions to produce num_categories non-empty slices.
    #
    # Example with 6 goods and 3 categories:
    #   shuffled  = ['g4', 'g1', 'g6', 'g2', 'g5', 'g3']
    #   cut points chosen from {1,2,3,4,5}: say [2, 4]
    #   → full cut_points = [0, 2, 4, 6]
    #   → c1 = shuffled[0:2] = ['g4','g1']
    #     c2 = shuffled[2:4] = ['g6','g2']
    #     c3 = shuffled[4:6] = ['g5','g3']
    shuffled = list(goods)
    rng.shuffle(shuffled)

    # replace=False guarantees all cut points are distinct integers from {1,...,num_items-1},
    # so after sorting: each consecutive pair differs by >= 1 → every slice is non-empty.
    cut_points = sorted(rng.choice(range(1, num_items), num_categories - 1, replace=False).tolist())
    cut_points = [0] + cut_points + [num_items]
    item_categories = {
        f"c{h + 1}": shuffled[cut_points[h]:cut_points[h + 1]]
        for h in range(num_categories)
    }

    # --- Set thresholds: k_h >= ceil(|C_h| / n) ---
    # The paper requires n * k_h >= |C_h| so that all goods in category h can be
    # distributed (n agents, each taking at most k_h goods from h).
    # Rearranged: k_h >= ceil(|C_h| / n).
    #
    # Example: |C_h| = 5 goods, n = 3 agents → k_h >= ceil(5/3) = 2.
    #   With k_h=2 every agent takes at most 2 goods from c_h; 3*2=6 >= 5, so all goods fit.
    #   With k_h=1 only 3*1=3 goods could be allocated, leaving 2 goods unallocated — invalid.
    #
    # We add a small random slack of 0 or 1 so we also test non-tight (loose) thresholds.
    category_capacities = {
        cat: int(math.ceil(len(items) / num_agents)) + int(rng.integers(0, 2))
        for cat, items in item_categories.items()
    }

    # --- Generate non-negative integer valuations ---
    # Each agent independently assigns a random value in [0, 19] to every good.
    # Valuations are additive: an agent's value for a bundle is the sum over its goods.
    #
    # Example with 2 agents and 3 goods:
    #   valuations = {
    #       'a1': {'g1': 12, 'g2': 0, 'g3': 7},
    #       'a2': {'g1': 3,  'g2': 15, 'g3': 9},
    #   }
    valuations = {
        a: {g: int(rng.integers(0, 20)) for g in goods}
        for a in agents
    }

    # --- Shuffle agent order to avoid bias toward any fixed picking sequence ---
    # The initial_agent_order determines who picks first in the first category.
    # Shuffling here ensures tests cover many different starting orders, not always a1 first.
    agent_order = list(agents)
    rng.shuffle(agent_order)

    return valuations, item_categories, category_capacities, agent_order


# ---------------------------------------------------------------------------
# Section 1: fair_division_under_cardinality_constraints (Algorithm 1)
# ---------------------------------------------------------------------------

def test_algorithm1_all_goods_allocated():
    """Every good appears in exactly one bundle (no omissions, no duplicates)."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        # Generate a fresh random instance for each iteration using a deterministic seed,
        # so failures are reproducible.
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents=4, num_items=20, num_categories=3, seed=i * 7
        )
        # Collect the ground-truth list of all goods across all categories.
        all_goods = [g for items in item_categories.values() for g in items]
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # Flatten all bundles into one list and sort both sides before comparing,
        # so the check is order-independent.
        allocated = sorted([g for bundle in result.values() for g in bundle])
        assert allocated == sorted(all_goods), (
            f"Seed {i * 7}: allocated goods {allocated} != all goods {sorted(all_goods)}"
        )


def test_algorithm1_default_order():
    """Passing initial_agent_order=None defaults to sorted(agents), giving the same result."""
    valuations = {
        'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
        'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    }
    instance = Instance(valuations=valuations)
    item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    category_capacities = {'c1': 1, 'c2': 1}

    # Run once with None — the algorithm should default to sorted(agents) internally.
    result_none = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories=item_categories,
        category_capacities=category_capacities,
        initial_agent_order=None,
    )
    # Run again with the sorted order passed explicitly.
    result_sorted = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories=item_categories,
        category_capacities=category_capacities,
        initial_agent_order=sorted(valuations.keys()),
    )

    # Sanity check: the allocation must be non-trivial (goods were actually distributed).
    assert any(result_none[a] for a in result_none), "allocation is empty — algorithm did not run"

    # Core check: both calls must produce identical results, proving None == sorted order.
    assert result_none == result_sorted


def test_algorithm1_topo_order_influences_next_category():
    """
    Verify that the topological sort of the envy graph after C1 actually changes
    who picks first in C2, and that this change affects the outcome.

    Setup:
      C1 round-robin σ=['a1','a2'], k_h=2 (3 goods):
        a1 picks g1 (value 9), a2 picks g2 (value 8), a1 picks g3 (only remaining).
      After C1: a1=[g1,g3], a2=[g2].

      Envy check:
        v_a2(a1's bundle) = 7+2 = 9  >  v_a2(a2's bundle) = 8  → a2 envies a1. Edge: a2→a1.
      No cycle → topo sort gives σ = ['a2','a1'] for C2.

      C2 round-robin σ=['a2','a1'], k_h=2 (3 goods):
        a2 picks g4 (its top C2 good, value 9).
        a1 picks g5 (best remaining for a1: value 3).
        a2 picks g6 (only remaining, value 1).

    The key: BOTH agents rank g4 as their top C2 good (a1 values it 10, a2 values it 9).
    If a1 had gone first (original order), a1 would have taken g4.
    Because a2 envied a1, topo sort put a2 first → a2 gets g4 instead.
    This proves the order change had a concrete effect on the allocation.
    """
    valuations = {
        'a1': {'g1': 9, 'g2': 6, 'g3': 1, 'g4': 10, 'g5': 3, 'g6': 2},
        'a2': {'g1': 7, 'g2': 8, 'g3': 2, 'g4': 9,  'g5': 4, 'g6': 1},
    }
    instance = Instance(valuations=valuations)
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories={'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']},
        category_capacities={'C1': 2, 'C2': 2},
        initial_agent_order=['a1', 'a2'],
    )
    # a2 picked first in C2 (topo sort) → a2 must have g4 (top C2 good for both agents).
    # If a1 had gone first, a1 would have taken g4 (a1 values it at 10, its personal best in C2).
    assert 'g4' in result['a2'], (
        f"Expected a2 to get g4 (top C2 good for both agents) since a2 envies a1 after C1 "
        f"→ topo sort puts a2 first in C2. Got: a2={result['a2']}"
    )
    # a1 was pushed to second pick in C2 → a1 gets g5 (best remaining after g4 is taken).
    assert 'g4' not in result['a1'], (
        f"a1 should NOT have g4: topo sort gave a2 the first pick. Got: a1={result['a1']}"
    )


# ---------------------------------------------------------------------------
# Section 2: greedy_round_robin (Algorithm 2)
# ---------------------------------------------------------------------------

def test_greedy_picks_top_good():
    """Each agent greedily picks their personal highest-valued good."""
    valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 8}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['m1', 'm2'], ['Alice', 'Bob'])
    assert alloc.sorted() == {'Alice': ['m1'], 'Bob': ['m2']}


def test_greedy_multiple_rounds():
    """k_h=2: each agent makes 2 picks across 2 rounds."""
    valuations = {
        'Alice': {'m1': 10, 'm2': 8, 'm3': 5, 'm4': 3},
        'Bob':   {'m1': 3,  'm2': 5, 'm3': 8, 'm4': 10},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['m1', 'm2', 'm3', 'm4'], ['Alice', 'Bob'])
    assert alloc.sorted() == {'Alice': ['m1', 'm2'], 'Bob': ['m3', 'm4']}


def test_greedy_partial_rounds():
    """More agents than goods: only the first agents in order receive goods."""
    valuations = {
        'A': {'g1': 5, 'g2': 3},
        'B': {'g1': 3, 'g2': 5},
        'C': {'g1': 4, 'g2': 4},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Only 2 goods, 3 agents, k_h=1 → only first 2 agents in order get a good
    greedy_round_robin(alloc, ['g1', 'g2'], ['A', 'B', 'C'])
    sorted_result = alloc.sorted()
    # Both g1 and g2 must be allocated
    allocated = [g for bundle in sorted_result.values() for g in bundle]
    assert sorted(allocated) == ['g1', 'g2']
    # C (third in order) should get nothing
    assert sorted_result['C'] == []


def test_greedy_all_goods_allocated():
    """Every good in the category ends up in exactly one agent's bundle."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 5)
        n, m = 3, 9
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(1, 15)) for g in goods} for a in agents}
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        greedy_round_robin(alloc, goods, agents)
        allocated = sorted([g for bundle in alloc.bundles.values() for g in bundle])
        assert allocated == sorted(goods), f"Seed {i * 5}: not all goods allocated"


def test_greedy_respects_order():
    """
    The agent first in agent_order gets the globally most-valued good (if they value it highest).
    Reversing the order changes who gets the top good.
    """
    valuations = {'A': {'x': 10, 'y': 1}, 'B': {'x': 10, 'y': 1}}
    # Both value x=10 equally; whoever is first gets x.
    instance = Instance(valuations=valuations)

    alloc1 = AllocationBuilder(instance)
    greedy_round_robin(alloc1, ['x', 'y'], ['A', 'B'])
    assert 'x' in alloc1.bundles['A']  # A is first, gets x

    alloc2 = AllocationBuilder(instance)
    greedy_round_robin(alloc2, ['x', 'y'], ['B', 'A'])
    assert 'x' in alloc2.bundles['B']  # B is first, gets x


def test_greedy_tied_valuations():
    """Tied valuations must not crash; both goods must be allocated."""
    valuations = {'A': {'g1': 5, 'g2': 5}, 'B': {'g1': 5, 'g2': 5}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['g1', 'g2'], ['A', 'B'])
    allocated = sorted([g for bundle in alloc.bundles.values() for g in bundle])
    assert allocated == ['g1', 'g2']


# ---------------------------------------------------------------------------
# Section 3: eliminate_envy_cycles
# ---------------------------------------------------------------------------

def test_eliminate_no_cycle():
    """No envy: graph is already a DAG, bundles are unchanged."""
    valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 9}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Alice has m1, Bob has m2 — each holds the good they value most.
    alloc.give('Alice', 'm1')
    alloc.give('Bob', 'm2')
    # Envy check:
    #   Alice values own bundle=9, Bob's bundle=3 → Alice does NOT envy Bob.
    #   Bob   values own bundle=9, Alice's bundle=3 → Bob   does NOT envy Alice.
    # No edges → graph is already a DAG. No cycle to eliminate.
    G = eliminate_envy_cycles(alloc)
    # Graph must contain a node for every agent.
    assert set(G.nodes()) == {'Alice', 'Bob'}
    assert list(nz.simple_cycles(G)) == []
    # Bundles must be untouched — no rotation happened.
    assert sorted(alloc.bundles['Alice']) == ['m1']
    assert sorted(alloc.bundles['Bob']) == ['m2']


def test_eliminate_2agent_cycle():
    """2-agent mutual envy cycle: bundles are swapped."""
    valuations = {'Alice': {'m1': 3, 'm2': 7}, 'Bob': {'m1': 7, 'm2': 3}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Alice has m1 (she values at 3), Bob has m2 (he values at 3).
    alloc.give('Alice', 'm1')
    alloc.give('Bob', 'm2')
    # Envy check:
    #   Alice values own bundle=3, Bob's bundle=7  → Alice ENVIES Bob. Edge: Alice→Bob.
    #   Bob   values own bundle=3, Alice's bundle=7 → Bob   ENVIES Alice. Edge: Bob→Alice.
    # Cycle detected: Alice→Bob→Alice.
    # Rotation: Alice receives Bob's bundle, Bob receives Alice's bundle (swap).
    # After swap: Alice has m2 (value 7), Bob has m1 (value 7) — no more envy.
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    assert sorted(alloc.bundles['Alice']) == ['m2']
    assert sorted(alloc.bundles['Bob']) == ['m1']


def test_eliminate_3agent_cycle():
    """3-agent cycle: bundle rotation resolves all envy."""
    valuations = {
        'A': {'m1': 3, 'm2': 5, 'm3': 1},
        'B': {'m1': 1, 'm2': 3, 'm3': 5},
        'C': {'m1': 5, 'm2': 1, 'm3': 3},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Initial assignment: A has m1, B has m2, C has m3.
    alloc.give('A', 'm1')
    alloc.give('B', 'm2')
    alloc.give('C', 'm3')
    # Envy check:
    #   A values own=3, B's=5, C's=1 → A ENVIES B. Edge: A→B.
    #   B values own=3, A's=1, C's=5 → B ENVIES C. Edge: B→C.
    #   C values own=3, A's=5, B's=1 → C ENVIES A. Edge: C→A.
    # Cycle detected: A→B→C→A.
    # Rotation along the cycle: A←B's bundle, B←C's bundle, C←A's bundle.
    # After rotation: A has m2 (value 5), B has m3 (value 5), C has m1 (value 5) — no envy.
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    assert sorted(alloc.bundles['A']) == ['m2']
    assert sorted(alloc.bundles['B']) == ['m3']
    assert sorted(alloc.bundles['C']) == ['m1']


def test_eliminate_result_is_dag():
    """After eliminate_envy_cycles, the returned graph always has no directed cycles."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 3)
        n, m = 10, 30
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(0, 15)) for g in goods} for a in agents}
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        # Distribute goods round-robin style (idx % n) to create arbitrary, potentially
        # envy-heavy allocations — this maximises the chance of generating cycles.
        shuffled = list(goods)
        rng.shuffle(shuffled)
        for idx, good in enumerate(shuffled):
            alloc.give(agents[idx % n], good)
        # The paper guarantees eliminate_envy_cycles always produces a DAG.
        # We verify this holds for every random allocation.
        G = eliminate_envy_cycles(alloc)
        assert set(G.nodes()) == set(agents), f"Seed {i * 3}: graph missing agent nodes"
        assert list(nz.simple_cycles(G)) == [], f"Seed {i * 3}: cycle found in result graph"


def test_eliminate_no_value_decrease():
    """Lemma 1: after cycle elimination, no agent's total valuation decreases."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 11)
        n, m = 10, 30
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(0, 15)) for g in goods} for a in agents}
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        # Same arbitrary distribution as test_eliminate_result_is_dag.
        shuffled = list(goods)
        rng.shuffle(shuffled)
        for idx, good in enumerate(shuffled):
            alloc.give(agents[idx % n], good)
        # Snapshot each agent's total value before any cycle elimination.
        value_before = {
            a: sum(instance.agent_item_value(a, g) for g in alloc.bundles[a])
            for a in agents
        }
        G = eliminate_envy_cycles(alloc)
        assert set(G.nodes()) == set(agents), f"Seed {i * 11}: graph missing agent nodes"
        # Snapshot each agent's total value after cycle elimination.
        value_after = {
            a: sum(instance.agent_item_value(a, g) for g in alloc.bundles[a])
            for a in agents
        }
        # Lemma 1: rotating bundles along an envy cycle never
        # reduces any agent's utility — each agent in the cycle receives a bundle it
        # envied, so its value can only stay the same or increase.
        for a in agents:
            assert value_after[a] >= value_before[a], (
                f"Seed {i * 11}: agent {a} value decreased from {value_before[a]} to {value_after[a]}"
            )

# ---------------------------------------------------------------------------
# Section 4: validate_fair_division_inputs
# ---------------------------------------------------------------------------

def _make_alloc(valuations):
    """Helper: construct an AllocationBuilder from a valuations dict."""
    return AllocationBuilder(Instance(valuations=valuations))


def test_validate_check1_agent_order_type():
    """Check 1: initial_agent_order must be None or a list — any other type raises ValueError."""
    alloc = _make_alloc({'X': {'g1': 5, 'g2': 3}, 'Y': {'g1': 3, 'g2': 7}})
    item_categories = {'c1': ['g1', 'g2']}
    category_capacities = {'c1': 1}

    # Valid: None is explicitly allowed (defaults to sorted agents)
    validate_fair_division_inputs(alloc, item_categories, category_capacities, None)  # must not raise
    # Valid: a list is the correct type
    validate_fair_division_inputs(alloc, item_categories, category_capacities, ['X', 'Y'])  # must not raise

    # Invalid: tuple looks like a list but is a different type
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, ('X', 'Y'))
    # Invalid: set has no defined order and is not a list
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, {'X', 'Y'})
    # Invalid: integer is clearly wrong
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, 42)


def test_validate_check2_item_categories_type():
    """Check 2: item_categories must be a dict — any other type raises ValueError."""
    alloc = _make_alloc({'A': {'g1': 4, 'g2': 6}, 'B': {'g1': 6, 'g2': 4}})
    category_capacities = {'c1': 1}

    # Invalid: list of tuples is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, [('c1', ['g1', 'g2'])], category_capacities, ['A', 'B'])
    # Invalid: None is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, None, category_capacities, ['A', 'B'])
    # Invalid: a plain string is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, 'c1', category_capacities, ['A', 'B'])


def test_validate_check3_category_capacities_type():
    """Check 3: category_capacities must be a dict — any other type raises ValueError."""
    alloc = _make_alloc({'P': {'x': 2, 'y': 8}, 'Q': {'x': 8, 'y': 2}})
    item_categories = {'c1': ['x', 'y']}

    # Invalid: list of tuples is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, [('c1', 1)], ['P', 'Q'])
    # Invalid: None is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, None, ['P', 'Q'])
    # Invalid: integer is not a dict
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, 1, ['P', 'Q'])


def test_validate_check4_category_values_are_lists():
    """Check 4: every value in item_categories must be a list — tuple or set raises ValueError."""
    alloc = _make_alloc({'A': {'m1': 3, 'm2': 7}, 'B': {'m1': 7, 'm2': 3}})
    category_capacities = {'c1': 1}

    # Invalid: tuple of items instead of list
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ('m1', 'm2')}, category_capacities, ['A', 'B'])
    # Invalid: set of items instead of list
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': {'m1', 'm2'}}, category_capacities, ['A', 'B'])


def test_validate_check5_at_least_one_agent():
    """Check 5: initial_agent_order must not be empty — empty list raises ValueError.
    Empty valuations are rejected by the fairpyx framework before reaching this function."""
    alloc = _make_alloc({'Alice': {'g1': 5, 'g2': 3}, 'Bob': {'g1': 3, 'g2': 5}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['g1', 'g2']}, {'c1': 1}, [])


def test_validate_check6_at_least_one_category():
    """Check 6: item_categories must not be empty — no categories at all raises ValueError."""
    alloc = _make_alloc({'A': {'g1': 5}, 'B': {'g1': 3}})

    # Invalid: empty dict means no categories exist
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {}, {}, ['A', 'B'])


def test_validate_check7_no_empty_category():
    """Check 7: every category must contain at least one item — empty list raises ValueError."""
    alloc = _make_alloc({'A': {'m1': 5, 'm2': 3}, 'B': {'m1': 3, 'm2': 7}})

    # Invalid: the only category has an empty item list
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': []}, {'c1': 1}, ['A', 'B'])
    # Invalid: one of two categories is empty
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2'], 'c2': []},
            {'c1': 1, 'c2': 1},
            ['A', 'B'],
        )


def test_validate_check8_agent_order_is_valid_permutation():
    """Check 8: if given, initial_agent_order must be an exact permutation of all agents."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    item_categories = {'c1': ['m1', 'm2']}
    category_capacities = {'c1': 1}

    # Valid: None skips this check entirely
    validate_fair_division_inputs(alloc, item_categories, category_capacities, None)  # must not raise
    # Valid: correct permutation
    validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Bob', 'Alice'])  # must not raise

    # Invalid: one agent missing from the order
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice'])
    # Invalid: duplicate agent (Bob appears twice, Alice missing)
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Alice'])
    # Invalid: unknown agent 'Charlie' not in instance
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Charlie'])
    # Invalid: correct agents plus an extra unknown agent
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Bob', 'Eve'])


def test_validate_check9_capacities_keys_match_categories():
    """Check 9: category_capacities keys must match item_categories keys exactly."""
    alloc = _make_alloc({'A': {'g1': 5, 'g2': 3, 'g3': 7}, 'B': {'g1': 7, 'g2': 5, 'g3': 3}})

    # Valid: keys match exactly
    validate_fair_division_inputs(
        alloc,
        {'c1': ['g1', 'g2'], 'c2': ['g3']},
        {'c1': 1, 'c2': 1},
        ['A', 'B'],
    )  # must not raise

    # Invalid: c2 exists in item_categories but has no threshold
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['g1', 'g2'], 'c2': ['g3']},
            {'c1': 1},
            ['A', 'B'],
        )
    # Invalid: c99 appears in category_capacities but not in item_categories
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['g1', 'g2', 'g3']},
            {'c1': 1, 'c99': 2},
            ['A', 'B'],
        )


def test_validate_check10_items_in_categories_exist_in_instance():
    """Check 10: every item listed in item_categories must exist in the instance valuations."""
    alloc = _make_alloc({'A': {'g1': 5, 'g2': 3}, 'B': {'g1': 3, 'g2': 7}})

    # Invalid: 'ghost' is listed in a category but has no valuation in the instance
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['g1', 'ghost']}, {'c1': 1}, ['A', 'B'])
    # Invalid: entirely unknown item
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['g1'], 'c2': ['g2', 'g99']}, {'c1': 1, 'c2': 1}, ['A', 'B'])


def test_validate_check11_all_instance_items_are_categorised():
    """Check 11: every item in the instance valuations must appear in at least one category."""
    alloc = _make_alloc({'A': {'g1': 5, 'g2': 3, 'g3': 8}, 'B': {'g1': 3, 'g2': 7, 'g3': 2}})

    # Invalid: g3 is in the instance but not listed in any category
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['g1', 'g2']},
            {'c1': 1},
            ['A', 'B'],
        )
    # Invalid: g2 and g3 both uncategorised
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['g1']}, {'c1': 1}, ['A', 'B'])


def test_validate_check12_no_item_in_two_categories():
    """Check 12: no item may appear in more than one category."""
    alloc = _make_alloc({'A': {'m1': 5, 'm2': 3}, 'B': {'m1': 3, 'm2': 7}})

    # Invalid: m1 appears in both c1 and c2
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2'], 'c2': ['m1']},
            {'c1': 1, 'c2': 1},
            ['A', 'B'],
        )
    # Invalid: m2 duplicated across two categories
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2'], 'c2': ['m2']},
            {'c1': 1, 'c2': 1},
            ['A', 'B'],
        )


def test_validate_check13_non_negative_valuations():
    """Check 13: all valuation values must be >= 0 — negative values raise ValueError."""
    # Valid: zero valuation is allowed
    alloc_zero = _make_alloc({'A': {'g1': 0, 'g2': 5}, 'B': {'g1': 5, 'g2': 0}})
    validate_fair_division_inputs(alloc_zero, {'c1': ['g1', 'g2']}, {'c1': 1}, ['A', 'B'])  # must not raise

    # Invalid: one agent has a negative value for a good
    alloc_neg = _make_alloc({'A': {'g1': -1, 'g2': 5}, 'B': {'g1': 5, 'g2': 3}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc_neg, {'c1': ['g1', 'g2']}, {'c1': 1}, ['A', 'B'])
    # Invalid: negative value hidden in a multi-category instance
    alloc_neg2 = _make_alloc({'A': {'g1': 3, 'g2': -2, 'g3': 5}, 'B': {'g1': 5, 'g2': 4, 'g3': -1}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc_neg2,
            {'c1': ['g1', 'g2'], 'c2': ['g3']},
            {'c1': 1, 'c2': 1},
            ['A', 'B'],
        )


def test_validate_check14_k_h_is_positive_integer():
    """Check 14: each k_h must be a positive integer — zero, negative, or non-integer raises ValueError."""
    alloc = _make_alloc({'A': {'g1': 5, 'g2': 3}, 'B': {'g1': 3, 'g2': 7}})
    item_categories = {'c1': ['g1', 'g2']}

    # Invalid: zero is not positive
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, {'c1': 0}, ['A', 'B'])
    # Invalid: negative integer
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, {'c1': -1}, ['A', 'B'])
    # Invalid: float, even if it looks like a whole number
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, {'c1': 1.5}, ['A', 'B'])
    # Invalid: string representation of a number
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, item_categories, {'c1': '1'}, ['A', 'B'])


def test_validate_check15_k_h_feasibility():
    """Check 15: k_h >= ceil(|C_h| / n) — ensures all goods in the category can be distributed."""
    # 3 goods, 2 agents → minimum k_h = ceil(3/2) = 2
    alloc = _make_alloc({
        'A': {'g1': 9, 'g2': 5, 'g3': 1},
        'B': {'g1': 1, 'g2': 5, 'g3': 9},
    })

    # Valid: k_h=2 is exactly the minimum (boundary case)
    validate_fair_division_inputs(alloc, {'c1': ['g1', 'g2', 'g3']}, {'c1': 2}, ['A', 'B'])  # must not raise
    # Valid: k_h=3 is above the minimum
    validate_fair_division_inputs(alloc, {'c1': ['g1', 'g2', 'g3']}, {'c1': 3}, ['A', 'B'])  # must not raise

    # Invalid: k_h=1 → only 2*1=2 goods can be allocated but there are 3
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['g1', 'g2', 'g3']}, {'c1': 1}, ['A', 'B'])

    # Multi-category case: c1 has 4 goods, 3 agents → k_h >= ceil(4/3) = 2
    alloc2 = _make_alloc({
        'X': {'g1': 9, 'g2': 7, 'g3': 4, 'g4': 1},
        'Y': {'g1': 1, 'g2': 4, 'g3': 7, 'g4': 9},
        'Z': {'g1': 5, 'g2': 5, 'g3': 5, 'g4': 5},
    })
    # Valid: k_h=2 == ceil(4/3)
    validate_fair_division_inputs(alloc2, {'c1': ['g1', 'g2', 'g3', 'g4']}, {'c1': 2}, ['X', 'Y', 'Z'])  # must not raise
    # Invalid: k_h=1 → only 3*1=3 goods can be allocated but there are 4
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc2, {'c1': ['g1', 'g2', 'g3', 'g4']}, {'c1': 1}, ['X', 'Y', 'Z'])


# ---------------------------------------------------------------------------
# Large random integration tests (Algorithm 1, varying sizes)
# ---------------------------------------------------------------------------

def test_random_integration_small():
    """3-5 agents, 10-20 goods, 2-4 categories: EF1 and cardinality must hold."""
    # Each tuple is (num_agents, num_items, num_categories, seed).
    # The range of sizes covers the most common real-world scenarios:
    # few agents, moderate number of goods, 2-4 categories.
    # Fixed seeds make failures fully reproducible.
    configs = [
        (3, 12, 2, 101),
        (4, 16, 3, 202),
        (5, 20, 4, 303),
        (3, 10, 2, 404),
        (4, 15, 3, 505),
    ]
    for num_agents, num_items, num_categories, seed in configs:
        # Generate a fresh valid random instance for this configuration.
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents, num_items, num_categories, seed
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # Sanity: every good must appear in exactly one bundle.
        all_goods = sorted(g for items in item_categories.values() for g in items)
        allocated = sorted(g for bundle in result.values() for g in bundle)
        assert allocated == all_goods, (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): not all goods allocated"
        )
        # Guarantee 1: no agent receives more than k_h goods from any single category.
        assert check_cardinality_constraints(result, item_categories, category_capacities), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): cardinality violated"
        )
        # Guarantee 2: EF1 holds — for every pair (i,j), removing j's most valued good
        # by i eliminates i's envy (Biswas & Barman 2018, Theorem 1).
        assert check_ef1(result, valuations), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): EF1 violated"
        )


def test_random_integration_large():
    """6-8 agents, 30-50 goods, 4-6 categories: EF1 and cardinality must hold."""
    # Larger instances stress-test the algorithm's correctness at scale.
    # More agents and categories means more envy cycles and more topo-sort reorderings,
    # so bugs that only surface in complex interactions are more likely to appear here.
    configs = [
        (6, 30, 4, 1001),
        (7, 35, 5, 2002),
        (8, 50, 6, 3003),
    ]
    for num_agents, num_items, num_categories, seed in configs:
        # Generate a fresh valid random instance for this configuration.
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents, num_items, num_categories, seed
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # Sanity: every good must appear in exactly one bundle.
        all_goods = sorted(g for items in item_categories.values() for g in items)
        allocated = sorted(g for bundle in result.values() for g in bundle)
        assert allocated == all_goods, (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): not all goods allocated"
        )
        # Guarantee 1: no agent receives more than k_h goods from any single category.
        assert check_cardinality_constraints(result, item_categories, category_capacities), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): cardinality violated"
        )
        # Guarantee 2: EF1 holds — for every pair (i,j), removing j's most valued good
        # by i eliminates i's envy (Biswas & Barman 2018, Theorem 1).
        assert check_ef1(result, valuations), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): EF1 violated"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])