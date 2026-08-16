"""
An implementation of the algorithm in:
"Fair Division Under Cardinality Constraints", by A. Biswas, S. Barman (2018), https://arxiv.org/abs/1804.09521
Programmer: Sapir Dahan
Date : 2026-05
"""

import math
import logging
import networkz as nz
from fairpyx import Instance, AllocationBuilder

logger = logging.getLogger(__name__)



def fair_division_under_cardinality_constraints(
    alloc: AllocationBuilder,
    item_categories: dict,
    category_capacities: dict,
    initial_agent_order: list = None,
):
    """
    Compute a feasible EF1 allocation under cardinality constraints (Algorithm 1).

    All agents share the same per-category threshold k_h. Valuations are additive and
    may differ across agents. The algorithm guarantees EF1: for every pair of agents
    (i, j) there exists some good g in j's bundle such that v_i(A_i) >= v_i(A_j \\ {g}).

    Procedure:
      1. Validate all inputs via validate_fair_division_inputs.
      2. For each category h in item_categories (in iteration order):
         a. Call greedy_round_robin to allocate the goods in h using the current agent order.
         b. Call eliminate_envy_cycles to remove all directed cycles from the envy graph and
            obtain an acyclic envy graph G_h.
         c. Derive the agent order for the next category as the topological sort of G_h
            (agents with no incoming envy edges pick first in the next round).

    :param alloc: an allocation builder, which tracks the allocation and the remaining
        capacity for items and agents.
    :param item_categories: a dictionary mapping each category name (str) to a list of
        item names. Every item in the instance must appear in exactly one category.
        Example: {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4', 'm5']}.
    :param category_capacities: a dictionary mapping each category name (str) to a positive
        integer threshold k_h — the maximum number of goods each agent may receive from that
        category. All agents share the same k_h (required by the paper).
        Example: {'c1': 1, 'c2': 2}.
    :param initial_agent_order: a list of agent names specifying the initial picking order
        for the first category. Must contain each agent exactly once. If None (default),
        agents are sorted lexicographically, giving a fully deterministic result.

    >>> # Example 1: basic — 2 agents, 2 categories, 1 good each, k_h=1
    >>> # Agent1 prefers m1 and m3; Agent2 prefers m2 and m4.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
    ...     'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

    >>> # Example 1b: same setup, no initial_agent_order — defaults to sorted(agents) = ['Agent1', 'Agent2']
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities)
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

    >>> # Example 2: 3 agents, 2 categories of 3 goods each, k_h=1
    >>> # Each agent's top good in each category is unique, so the output is deterministic.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 9, 'm2': 6, 'm3': 3, 'm4': 8, 'm5': 5, 'm6': 2},
    ...     'Agent2': {'m1': 3, 'm2': 9, 'm3': 6, 'm4': 2, 'm5': 8, 'm6': 5},
    ...     'Agent3': {'m1': 6, 'm2': 3, 'm3': 9, 'm4': 5, 'm5': 2, 'm6': 8},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2', 'm3'], 'c2': ['m4', 'm5', 'm6']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2', 'Agent3'])
    {'Agent1': ['m1', 'm4'], 'Agent2': ['m2', 'm5'], 'Agent3': ['m3', 'm6']}

    >>> # Example 3: single agent — trivially receives all goods (up to its capacity)
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Alice': {'m1': 10, 'm2': 7, 'm3': 4}}
    >>> item_categories = {'c1': ['m1', 'm2', 'm3']}
    >>> category_capacities = {'c1': 3}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Alice'])
    {'Alice': ['m1', 'm2', 'm3']}

    >>> # Example 4: single category — degenerates to a single greedy_round_robin call
    >>> from fairpyx import Instance, divide
    >>> valuations = {'A': {'x': 10, 'y': 5, 'z': 1}, 'B': {'x': 1, 'y': 5, 'z': 10}}
    >>> item_categories = {'c1': ['x', 'y', 'z']}
    >>> category_capacities = {'c1': 2}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['A', 'B'])
    {'A': ['x', 'y'], 'B': ['z']}

    >>> # Example 5: single good per category — each agent can receive at most one good per category
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 8, 'm2': 5}, 'Agent2': {'m1': 5, 'm2': 8}}
    >>> item_categories = {'c1': ['m1'], 'c2': ['m2']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1'], 'Agent2': ['m2']}

    >>> # Example 6: asymmetric category sizes — c1 has 4 goods (k_h=2), c2 has 1 good (k_h=1)
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 10, 'm2': 8, 'm3': 6, 'm4': 4, 'm5': 9},
    ...     'Agent2': {'m1': 4,  'm2': 6, 'm3': 8, 'm4': 10, 'm5': 7},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2', 'm3', 'm4'], 'c2': ['m5']}
    >>> category_capacities = {'c1': 2, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1', 'm2', 'm5'], 'Agent2': ['m3', 'm4']}

    >>> # Example 7: minimal base case — 1 agent, 1 good, 1 category.
    >>> # σ = ['a1'].  C1 round-robin: a1 picks g1 (only good). No envy possible.
    >>> # Final allocation: {a1: [g1]}.
    >>> from fairpyx import Instance, divide
    >>> valuations = {'a1': {'g1': 2}}
    >>> item_categories = {'C1': ['g1']}
    >>> category_capacities = {'C1': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1'])
    {'a1': ['g1']}

    >>> # Example 8: 2 agents, 2 goods, 1 category, k_h=1 - full example explanation
    >>> # σ = ['a1','a2'].
    >>> # C1 round-robin:
    >>> #   a1 picks best of {g1,g2}: values 2,3 → picks g2 (value 3).
    >>> #   a2 picks best of {g1}:    value  1   → picks g1 (only remaining).
    >>> # After C1: {a1:[g2], a2:[g1]}.
    >>> #
    >>> # Envy table (row = whose eyes, col = whose bundle):
    >>> #              a1's bundle   a2's bundle
    >>> #   a1 values:    3             2        → a1 does NOT envy a2 (3 >= 2).
    >>> #   a2 values:    4             1        → a2 ENVIES a1 (4 > 1). Edge: a2→a1.
    >>> #
    >>> # Envy graph: a2→a1. No cycle. Topo sort: ['a2','a1'].
    >>> # (Only one category, so updated σ is never used.)
    >>> # Final allocation: {a1:[g2], a2:[g1]}.
    >>> from fairpyx import Instance, divide
    >>> valuations = {'a1': {'g1': 2, 'g2': 3}, 'a2': {'g1': 1, 'g2': 4}}
    >>> item_categories = {'C1': ['g1', 'g2']}
    >>> category_capacities = {'C1': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2'])
    {'a1': ['g2'], 'a2': ['g1']}

    >>> # Example 9: 2 agents, 6 goods, 2 categories, k_h=2 - full example explanation
    >>> # Key feature: a2 envies a1 after C1, so the agent order REVERSES for C2,
    >>> # giving a2 the first pick in the second category.
    >>> #
    >>> # σ = ['a1','a2'].
    >>> # C1 round-robin (each agent picks at most k_h=2 goods):
    >>> #   a1 picks best of {g1,g2,g3}: values 9,6,3 → g1 (value 9).
    >>> #   a2 picks best of {g2,g3}:    values 7,8   → g3 (value 8).
    >>> #   a1 picks best of {g2}:        value  6    → g2 (only remaining).
    >>> # After C1: {a1:[g1,g2], a2:[g3]}.
    >>> #
    >>> # Envy table after C1:
    >>> #              a1's bundle {g1,g2}   a2's bundle {g3}
    >>> #   a1 values:     9+6=15                3         → a1 does NOT envy a2.
    >>> #   a2 values:     4+7=11                8         → a2 ENVIES a1 (11>8). Edge: a2→a1.
    >>> #
    >>> # Envy graph: a2→a1. No cycle. Topo sort: a2 has no incoming edge → first.
    >>> # σ updated to ['a2','a1'] for C2.
    >>> #
    >>> # C2 round-robin with σ=['a2','a1']:
    >>> #   a2 picks best of {g4,g5,g6}: values 3,9,6 → g5 (value 9).
    >>> #   a1 picks best of {g4,g6}:    values 8,2   → g4 (value 8).
    >>> #   a2 picks best of {g6}:        value  6    → g6 (only remaining).
    >>> # After C2: {a1:[g1,g2,g4], a2:[g3,g5,g6]}.
    >>> #
    >>> # Final envy table:
    >>> #              a1's bundle {g1,g2,g4}   a2's bundle {g3,g5,g6}
    >>> #   a1 values:    9+6+8=23                  3+5+2=10   → a1 does NOT envy a2.
    >>> #   a2 values:    4+7+3=14                  8+9+6=23   → a2 does NOT envy a1.
    >>> # No envy. Final allocation below.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'a1': {'g1': 9, 'g2': 6, 'g3': 3, 'g4': 8, 'g5': 5, 'g6': 2},
    ...     'a2': {'g1': 4, 'g2': 7, 'g3': 8, 'g4': 3, 'g5': 9, 'g6': 6},
    ... }
    >>> item_categories = {'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']}
    >>> category_capacities = {'C1': 2, 'C2': 2}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2'])
    {'a1': ['g1', 'g2', 'g4'], 'a2': ['g3', 'g5', 'g6']}

    >>> # Example 10: 3 agents, 6 goods, 2 categories, k_h=1 - full example explanation
    >>> # Key feature: after C2, the envy graph contains MULTIPLE OVERLAPPING CYCLES,
    >>> # requiring eliminate_envy_cycles to resolve them iteratively.
    >>> #
    >>> # σ = ['a1','a2','a3'].
    >>> # C1 round-robin:
    >>> #   a1 picks best of {g1,g2,g3}: values 9,2,1 → g1.
    >>> #   a2 picks best of {g2,g3}:    values 8,1   → g2.
    >>> #   a3 picks best of {g3}:        value  5    → g3.
    >>> # After C1: {a1:[g1], a2:[g2], a3:[g3]}.
    >>> #
    >>> # Envy table after C1:
    >>> #              a1 {g1}   a2 {g2}   a3 {g3}
    >>> #   a1 values:    9         2         1      → a1 does NOT envy anyone.
    >>> #   a2 values:   10         8         1      → a2 ENVIES a1 (10>8). Edge: a2→a1.
    >>> #   a3 values:   10         9         5      → a3 ENVIES a1 and a2. Edges: a3→a1, a3→a2.
    >>> #
    >>> # Envy graph: a2→a1, a3→a1, a3→a2. No cycle (DAG). Topo sort: ['a3','a2','a1'].
    >>> # σ updated to ['a3','a2','a1'] for C2.
    >>> #
    >>> # C2 round-robin with σ=['a3','a2','a1']:
    >>> #   a3 picks best of {g4,g5,g6}: values 8,1,7 → g4.
    >>> #   a2 picks best of {g5,g6}:    values 1,0   → g5.
    >>> #   a1 picks best of {g6}:        value  0    → g6.
    >>> # After C2: {a1:[g1,g6], a2:[g2,g5], a3:[g3,g4]}.
    >>> #
    >>> # Envy table after C2:
    >>> #              a1 {g1,g6}   a2 {g2,g5}   a3 {g3,g4}
    >>> #   a1 values:  9+0=9       2+10=12       1+1=2      → a1 ENVIES a2. Edge: a1→a2.
    >>> #   a2 values: 10+0=10      8+1=9        1+10=11     → a2 ENVIES a1 and a3. Edges: a2→a1, a2→a3.
    >>> #   a3 values: 10+7=17      9+1=10        5+8=13     → a3 ENVIES a1. Edge: a3→a1.
    >>> #
    >>> # Envy graph edges: a1→a2, a2→a1, a2→a3, a3→a1.
    >>> # This graph contains TWO overlapping simple cycles:
    >>> #   - 2-cycle: a1↔a2        (edges a1→a2 and a2→a1)
    >>> #   - 3-cycle: a1→a2→a3→a1 (edges a1→a2, a2→a3, a3→a1)
    >>> #
    >>> # Both elimination orderings are valid and both lead to the SAME final allocation
    >>> # (verified by tracing both paths):
    >>> #
    >>> #   PATH A — eliminate 3-cycle first:
    >>> #     Rotate: a1 ← a2's bundle, a2 ← a3's bundle, a3 ← a1's bundle.
    >>> #     After rotation: {a1:[g2,g5], a2:[g3,g4], a3:[g1,g6]}.
    >>> #     Verify: a1 own=12, a2's=9, a3's=9 → no envy.
    >>> #             a2 own=11, a1's=9, a3's=10 → no envy.
    >>> #             a3 own=17, a1's=10, a2's=13 → no envy. Done.
    >>> #
    >>> #   PATH B — eliminate 2-cycle first, then the resulting 2-cycle:
    >>> #     Step 1: swap a1↔a2 → {a1:[g2,g5], a2:[g1,g6], a3:[g3,g4]}.
    >>> #       New envy: a2 values a3's bundle at 11 > own 10, and a3 values a2's at 17 > own 13.
    >>> #       New 2-cycle: a2↔a3.
    >>> #     Step 2: swap a2↔a3 → {a1:[g2,g5], a2:[g3,g4], a3:[g1,g6]}.
    >>> #       No remaining envy. Done.
    >>> #
    >>> # Both paths produce the SAME EF1 allocation for this specific instance — verified above.
    >>> # Note: in general, different cycle orderings CAN lead to different valid EF1 allocations
    >>> # (the paper guarantees EF1 for any order, but not uniqueness). For instances where the
    >>> # two orderings diverge, use: assert result in [answer_path_A, answer_path_B].
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'a1': {'g1': 9, 'g2': 2, 'g3': 1, 'g4': 1, 'g5': 10, 'g6': 0},
    ...     'a2': {'g1': 10, 'g2': 8, 'g3': 1, 'g4': 10, 'g5': 1, 'g6': 0},
    ...     'a3': {'g1': 10, 'g2': 9, 'g3': 5, 'g4': 8, 'g5': 1, 'g6': 7},
    ... }
    >>> item_categories = {'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']}
    >>> category_capacities = {'C1': 1, 'C2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2', 'a3'])
    {'a1': ['g2', 'g5'], 'a2': ['g3', 'g4'], 'a3': ['g1', 'g6']}

    >>> # Example 11: 4 agents, 26 goods, 4 categories (5/6/7/8 goods), k_h=2 - large example
    >>> # This is the largest and most complex run example. Two order changes occur:
    >>> #   - After C1: a3 envies a1 → σ becomes ['a3','a1','a2','a4'].
    >>> #   - After C2: a2 envies a3 → σ becomes ['a2','a1','a3','a4'].
    >>> #   - After C3: no envy     → σ remains  ['a1','a2','a3','a4'].
    >>> #
    >>> # C1=[g1..g5], C2=[g6..g11], C3=[g12..g18], C4=[g19..g26]; k_h=2 for all.
    >>> #
    >>> # C1 round-robin σ=['a1','a2','a3','a4'] (5 goods, 2 picks each — only 4 agents × 2=8 slots but only 5 goods):
    >>> #   a1 picks g1 (9), a2 picks g2 (10), a3 picks g4 (9), a4 picks g3 (10), a1 picks g5 (6).
    >>> # After C1: {a1:[g1,g5], a2:[g2], a3:[g4], a4:[g3]}.
    >>> #
    >>> # Envy table after C1 (values shown as agent_i sees bundle_j):
    >>> #       a1{g1,g5}  a2{g2}  a3{g4}  a4{g3}
    >>> #   a1:    15        4       1       8     → a1 does not envy anyone (15 is max).
    >>> #   a2:     8       10       7       2     → a2 does not envy anyone (10 is max).
    >>> #   a3:    10        1       9       5     → a3 ENVIES a1 (10>9). Edge: a3→a1.
    >>> #   a4:     9        6       3      10     → a4 does not envy anyone (10 is max).
    >>> # No cycle. Topo sort: a3 first. σ = ['a3','a1','a2','a4'].
    >>> #
    >>> # C2 round-robin σ=['a3','a1','a2','a4'] (6 goods):
    >>> #   a3 picks g8 (10), a1 picks g7 (10), a2 picks g6 (9), a4 picks g9 (10),
    >>> #   a3 picks g10 (7), a1 picks g11 (7).
    >>> # After C2: {a1:[g1,g5,g7,g11], a2:[g2,g6], a3:[g4,g8,g10], a4:[g3,g9]}.
    >>> #
    >>> # Envy table after C2:
    >>> #       a1{..}  a2{g2,g6}  a3{g4,g8,g10}  a4{g3,g9}
    >>> #   a1:   32       6          5              13      → a1 does not envy (32 is max).
    >>> #   a2:   11      19         21               6      → a2 ENVIES a3 (21>19). Edge: a2→a3.
    >>> #   a3:   19       5         26              12      → a3 does not envy (26 is max).
    >>> #   a4:   15      11          6              20      → a4 does not envy (20 is max).
    >>> # No cycle. Topo sort: a2 first (only outgoing edge). σ = ['a2','a1','a3','a4'].
    >>> #
    >>> # C3 round-robin σ=['a2','a1','a3','a4'] (7 goods):
    >>> #   a2 picks g15 (10), a1 picks g14 (9), a3 picks g17 (10), a4 picks g18 (10),
    >>> #   a2 picks g13 (7), a1 picks g16 (8), a3 picks g12 (9).
    >>> # After C3: {a1:[g1,g5,g7,g11,g14,g16], a2:[g2,g6,g15,g13], a3:[g4,g8,g10,g17,g12], a4:[g3,g9,g18]}.
    >>> #
    >>> # Envy table after C3: no agent envies another. σ remains ['a1','a2','a3','a4'].
    >>> #
    >>> # C4 round-robin σ=['a1','a2','a3','a4'] (8 goods):
    >>> #   a1 picks g22 (10), a2 picks g25 (10), a3 picks g23 (9), a4 picks g24 (10),
    >>> #   a1 picks g26 (8), a2 picks g21 (9), a3 picks g19 (8), a4 picks g20 (8).
    >>> # After C4: {a1:[g1,g5,g7,g11,g14,g16,g22,g26], a2:[g2,g6,g15,g13,g25,g21],
    >>> #            a3:[g4,g8,g10,g17,g12,g23,g19], a4:[g3,g9,g18,g24,g20]}.
    >>> #
    >>> # Final envy table: no envy among any agents. EF1 holds.
    >>> # NOTE: marked SKIP — the exact output depends on lex tie-breaking across 26 goods.
    >>> from fairpyx import Instance, divide  # doctest: +SKIP
    >>> all_goods = [f'g{i}' for i in range(1, 27)]  # doctest: +SKIP
    >>> valuations = {  # doctest: +SKIP
    ...     'a1': {'g1':9,'g2':4,'g3':8,'g4':1,'g5':6,'g6':2,'g7':10,'g8':3,'g9':5,'g10':1,'g11':7,
    ...            'g12':6,'g13':2,'g14':9,'g15':4,'g16':8,'g17':1,'g18':3,
    ...            'g19':5,'g20':7,'g21':2,'g22':10,'g23':4,'g24':6,'g25':1,'g26':8},
    ...     'a2': {'g1':3,'g2':10,'g3':2,'g4':7,'g5':5,'g6':9,'g7':1,'g8':8,'g9':4,'g10':6,'g11':2,
    ...            'g12':1,'g13':7,'g14':3,'g15':10,'g16':5,'g17':8,'g18':4,
    ...            'g19':6,'g20':2,'g21':9,'g22':1,'g23':7,'g24':3,'g25':10,'g26':5},
    ...     'a3': {'g1':8,'g2':1,'g3':5,'g4':9,'g5':2,'g6':4,'g7':6,'g8':10,'g9':1,'g10':7,'g11':3,
    ...            'g12':9,'g13':5,'g14':2,'g15':6,'g16':1,'g17':10,'g18':4,
    ...            'g19':8,'g20':3,'g21':6,'g22':2,'g23':9,'g24':1,'g25':5,'g26':7},
    ...     'a4': {'g1':2,'g2':6,'g3':10,'g4':3,'g5':7,'g6':5,'g7':2,'g8':1,'g9':10,'g10':8,'g11':4,
    ...            'g12':3,'g13':9,'g14':6,'g15':2,'g16':7,'g17':5,'g18':10,
    ...            'g19':1,'g20':8,'g21':4,'g22':6,'g23':2,'g24':10,'g25':3,'g26':9},
    ... }
    >>> item_categories = {  # doctest: +SKIP
    ...     'C1': ['g1','g2','g3','g4','g5'],
    ...     'C2': ['g6','g7','g8','g9','g10','g11'],
    ...     'C3': ['g12','g13','g14','g15','g16','g17','g18'],
    ...     'C4': ['g19','g20','g21','g22','g23','g24','g25','g26'],
    ... }
    >>> category_capacities = {'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2}  # doctest: +SKIP
    >>> instance = Instance(valuations=valuations)  # doctest: +SKIP
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,  # doctest: +SKIP
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2', 'a3', 'a4'])
    {'a1': ['g1', 'g11', 'g14', 'g16', 'g22', 'g26', 'g5', 'g7'], 'a2': ['g13', 'g15', 'g2', 'g21', 'g25', 'g6'], 'a3': ['g10', 'g12', 'g17', 'g19', 'g23', 'g4', 'g8'], 'a4': ['g18', 'g20', 'g24', 'g3', 'g9']}

    # Invalid-input tests (negative valuations, k_h too small, duplicate goods across
    # categories, non-positive k_h, empty/duplicate initial_agent_order, uncategorised
    # goods) are covered in full by the doctests of validate_fair_division_inputs.
    """

    # Reject bad inputs early so the rest of the function can assume they are correct
    validate_fair_division_inputs(alloc, item_categories, category_capacities, initial_agent_order)

    # Collect all agent names from the instance
    agents = list(alloc.instance.agents)

    # Use the caller-supplied order, or fall back to alphabetical — "fix an ordering of the agents σ"
    agent_order = initial_agent_order if initial_agent_order is not None else sorted(agents)

    # Count total items so the log gives a useful overview before the loop starts
    total_items = sum(len(v) for v in item_categories.values())
    logger.info(
        "Starting: %d agents, %d items total across %d categories %s",
        len(agent_order), total_items, len(item_categories),
        {cat: len(itms) for cat, itms in item_categories.items()},
    )

    # Convert to a list so we can check whether we are on the last category
    categories = list(item_categories.items())

    # "for h = 1 to ℓ"
    for idx, (category, items) in enumerate(categories):

        # Log the category header so each iteration is easy to spot in the output
        logger.info(
            "--- Category %s (%d goods) | picking order: %s ---",
            category, len(items), agent_order,
        )

        # Log each agent's personal ranking of the goods in this category (most to least valued)
        for agent in agent_order:
            ranked = sorted(items, key=lambda g: alloc.instance.agent_item_value(agent, g), reverse=True)
            logger.debug(
                "  %s's preferences: %s",
                agent,
                [(g, int(alloc.instance.agent_item_value(agent, g))) for g in ranked],
            )

        # Distribute this category's goods using greedy round-robin — "B^h ← Greedy-Round-Robin(C_h, [n], (v_i)_i, σ)"
        greedy_round_robin(alloc, items, agent_order)

        # Remove any envy cycles from the current allocation by rotating bundles — "update A^h to obtain an acyclic envy graph G(A^h)"
        G = eliminate_envy_cycles(alloc)

        # Derive the picking order for the next category from the acyclic envy graph — "update σ to be a topological ordering of G(A^h)"
        agent_order = list(nz.topological_sort(G))

        # Only show the new order when there is a next category that will actually use it
        if idx < len(categories) - 1:
            logger.info("Picking order for next category: %s", agent_order)

            # Explain each agent's position: agents nobody envies pick first (their bundle is
            # the least desirable, so they get priority to compensate); agents many others envy
            # pick last (they already hold the most valuable goods)
            for rank, agent in enumerate(agent_order, 1):

                # Find all agents who envy this agent (i.e. who have a directed edge pointing at them)
                enviers = [i for i in G.nodes() if G.has_edge(i, agent)]

                if enviers:
                    # This agent's bundle is considered valuable by others, so they pick later as a penalty
                    logger.info(
                        "  pick %d: %s — envied by %s (their bundle is desirable → picks later)",
                        rank, agent, enviers,
                    )

                else:
                    # Nobody envies this agent's bundle, so they get an earlier pick as compensation.
                    # Note: multiple agents can have no enviers — rank shows their exact position among them
                    logger.info(
                        "  pick %d: %s — nobody envies their bundle → gets an earlier pick (priority)",
                        rank, agent,
                    )

    # Log the completed allocation once all categories have been processed
    logger.info("Final allocation: %s", {a: sorted(alloc.bundles[a]) for a in alloc.instance.agents})


def greedy_round_robin(
    alloc: AllocationBuilder,
    items_in_category: list,
    agent_order: list,
) -> None:
    """
    Allocate all goods from a single category using Greedy Round-Robin (Algorithm 2).

    Agents pick in round-robin order, cycling repeatedly. On each turn an agent greedily
    selects the remaining good in the category it values most. The procedure terminates
    when all goods are allocated.

    No explicit capacity parameter is needed: validate_fair_division_inputs guarantees
    k_h >= ceil(|C_h| / n) before this function is called, so the natural round-robin
    cycling already ensures no agent receives more than k_h goods.

    :param alloc: an allocation builder, which tracks the allocation and the remaining
        capacity for items and agents.
    :param items_in_category: the list of items belonging to the category being processed
        in this round.
    :param agent_order: the ordered list of agents specifying the picking sequence for this
        category (either initial_agent_order or the topological sort from the previous category).

    >>> # Example 1: 2 agents, 2 goods — each agent picks their top good
    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 8}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2'], ['Alice', 'Bob'])
    >>> alloc.sorted()
    {'Alice': ['m1'], 'Bob': ['m2']}

    >>> # Example 2: 3 agents, 3 goods — each picks their unique top good in order
    >>> valuations = {'A': {'m1': 9, 'm2': 5, 'm3': 1},
    ...               'B': {'m1': 3, 'm2': 9, 'm3': 2},
    ...               'C': {'m1': 2, 'm2': 4, 'm3': 8}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3'], ['A', 'B', 'C'])
    >>> alloc.sorted()
    {'A': ['m1'], 'B': ['m2'], 'C': ['m3']}

    >>> # Example 3: 2 agents, 4 goods — 2 rounds, each agent picks 2 goods
    >>> valuations = {'Alice': {'m1': 10, 'm2': 8, 'm3': 5, 'm4': 3},
    ...               'Bob':   {'m1': 3,  'm2': 5, 'm3': 8, 'm4': 10}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3', 'm4'], ['Alice', 'Bob'])
    >>> alloc.sorted()
    {'Alice': ['m1', 'm2'], 'Bob': ['m3', 'm4']}

    >>> # Example 4: competition — both agents want g1 most, but A picks first and takes it
    >>> # Round 1: A takes g1 (value 10, best for A). B wants g1 too (value 9) but it is gone;
    >>> #          B takes g2 instead (value 8, best remaining).
    >>> # Round 2: A takes g3 (only remaining, value 1).
    >>> # Result: A=[g1,g3], B=[g2]. B had to settle for its second choice.
    >>> valuations = {'A': {'g1': 10, 'g2': 3, 'g3': 1},
    ...               'B': {'g1': 9,  'g2': 8, 'g3': 2}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['g1', 'g2', 'g3'], ['A', 'B'])
    >>> alloc.sorted()
    {'A': ['g1', 'g3'], 'B': ['g2']}
    """

    # M is the pool of items still waiting to be distributed in this category — "initialise M ← C"
    # Using a local copy so the caller's list is never modified
    M = list(items_in_category)

    # Keep going until every item in this category has been assigned — "while M ≠ ∅"
    while M:

        # One full pass through all agents in their current picking order — "for i = 1 to n"
        for agent in agent_order:

            # All items in this category have been distributed — stop the round-robin even if
            # some agents in the current pass have not picked yet
            if not M:
                break

            # Each agent greedily picks the item they value most among what is still available — "argmax_{g ∈ M} v_{σ(i)}(g)"
            best_item = max(M, key=lambda g: alloc.instance.agent_item_value(agent, g))

            # Log which item was chosen and its value so picks are traceable
            logger.debug(
                "  Agent %s picks %s (value %g)",
                agent, best_item, alloc.instance.agent_item_value(agent, best_item),
            )

            # Record the pick in the allocation and remove the item from the available pool — "B_{σ(i)} ← B_{σ(i)} ∪ {g}; M ← M \ {g}"
            alloc.give(agent, best_item)
            M.remove(best_item)


def eliminate_envy_cycles(alloc: AllocationBuilder) -> nz.DiGraph:
    """
    Build the envy graph for the current (partial) allocation, eliminate all directed cycles
    by rotating bundles along each cycle, and return the resulting acyclic envy graph (Lemma 1).

    Envy relation: agent i envies agent j if
        sum_{g in A_j} v_i(g) > sum_{g in A_i} v_i(g).
    A directed edge i -> j is added to the graph when i envies j.

    Cycle elimination: for a detected cycle (a_1, a_2, ..., a_r), rotate bundles so that
    a_1 receives a_2's bundle, a_2 receives a_3's bundle, ..., a_r receives a_1's bundle.
    The paper proves that no agent's value decreases under such a rotation. This is repeated
    until the graph contains no directed cycles (i.e., it is a DAG).

    The returned DAG is passed to nz.topological_sort to derive the agent order for the next
    category: agents with no incoming envy edges (nobody envies them) pick first.

    :param alloc: an allocation builder whose alloc.bundles reflect the current partial
        allocation after the most recent greedy_round_robin call.
    :return: a networkz DiGraph representing the acyclic envy graph after all cycles have
        been eliminated.

    >>> # Example 1: no envy — graph is already a DAG, bundles are unchanged
    >>> from fairpyx import Instance, AllocationBuilder
    >>> import networkz as nz
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 9}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['Alice'])
    ['m1']
    >>> sorted(alloc.bundles['Bob'])
    ['m2']

    >>> # Example 2: 2-agent envy cycle — bundles are swapped to eliminate the cycle
    >>> # Alice has m1 (she values at 3), Bob has m2 (he values at 3).
    >>> # Alice envies Bob (values m2=7 > m1=3) and Bob envies Alice (values m1=7 > m2=3).
    >>> valuations = {'Alice': {'m1': 3, 'm2': 7}, 'Bob': {'m1': 7, 'm2': 3}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['Alice'])
    ['m2']
    >>> sorted(alloc.bundles['Bob'])
    ['m1']

    >>> # Example 3: 3-agent cycle — bundle rotation resolves all envy
    >>> # A has m1, B has m2, C has m3.
    >>> # A envies B (5>3), B envies C (5>3), C envies A (5>3) — a 3-cycle.
    >>> # After rotation: A gets m2, B gets m3, C gets m1 — no agent envies another.
    >>> valuations = {'A': {'m1': 3, 'm2': 5, 'm3': 1},
    ...               'B': {'m1': 1, 'm2': 3, 'm3': 5},
    ...               'C': {'m1': 5, 'm2': 1, 'm3': 3}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('A', 'm1')
    >>> alloc.give('B', 'm2')
    >>> alloc.give('C', 'm3')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['A'])
    ['m2']
    >>> sorted(alloc.bundles['B'])
    ['m3']
    >>> sorted(alloc.bundles['C'])
    ['m1']
    """

    # Collect all agent names once, the list stays fixed even as bundles are rotated below
    agents = list(alloc.instance.agents)

    # Repeat until the envy graph has no more cycles — "update A^h to obtain an acyclic envy graph G(A^h)"
    # Each iteration either eliminates one cycle (and restarts) or confirms the graph is a DAG (and exits)
    while True:

        # Build a fresh directed graph from scratch — bundles may have changed in the previous iteration
        G = nz.DiGraph()

        # Every agent must be a node even if nobody envies them, so topological sort sees all agents later
        G.add_nodes_from(agents)

        # Check every ordered pair (i, j) to decide whether agent i envies agent j
        for i in agents:

            # How much agent i values their own current bundle (sum of item values)
            val_i_own = alloc.instance.agent_bundle_value(i, alloc.bundles[i])

            for j in agents:

                # An agent cannot envy themselves, so skip the same-agent pair
                if i == j:
                    continue

                # How much agent i would value j's bundle if they had it instead
                val_i_other = alloc.instance.agent_bundle_value(i, alloc.bundles[j])

                if val_i_other > val_i_own:

                    # i strictly prefer j's bundle — this is the definition of envy, so add edge i → j
                    G.add_edge(i, j)
                    logger.debug(
                        "  %s envies %s  (%s values own bundle=%g < %s's bundle=%g)",
                        i, j, i, val_i_own, j, val_i_other,
                    )

        # Build a readable summary of who envies whom and log it at INFO level
        if G.edges():
            envy_desc = ",  ".join(f"{i} envies {j}" for i, j in G.edges())
            logger.info("Envy: %s", envy_desc)

        else:
            # No edges means no agent envies any other — allocation is already envy-free
            logger.info("No envy — all agents prefer their own bundle")

        # Try to find one directed cycle in the envy graph; we only need one per iteration
        # nz.simple_cycles returns a generator — next() takes the first cycle or None if none exist
        cycle = next(nz.simple_cycles(G), None)

        if cycle is None:

            # No cycle found — the graph is a DAG, so we are done
            logger.info("Envy graph is a DAG — no cycles, moving on")
            break

        # Format the cycle as "a → b → c → a" to make the direction clear in the log
        cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
        logger.info("Cycle detected: %s  — rotating bundles along cycle", cycle_str)

        # Snapshot every bundle in the cycle BEFORE any assignment, because we are about to
        # overwrite alloc.bundles entries and would otherwise lose the original values mid-rotation
        saved = {agent: alloc.bundles[agent] for agent in cycle}

        # Rotate: each agent at position idx receives the bundle of the agent at position idx+1
        # The modulo wraps the last agent back to position 0, giving the last agent the first agent's bundle
        # a_1 ← a_2's bundle, a_2 ← a_3's bundle, ..., a_r ← a_1's bundle
        for idx in range(len(cycle)):
            alloc.bundles[cycle[idx]] = saved[cycle[(idx + 1) % len(cycle)]]

        # Log the full allocation after the rotation so the effect on every agent is visible
        logger.info(
            "After rotation: %s",
            {a: sorted(alloc.bundles[a]) for a in alloc.bundles},
        )

    # Return the final acyclic envy graph so the caller can run topological sort to get the next picking order
    return G



def validate_fair_division_inputs(
    alloc: AllocationBuilder,
    item_categories: dict,
    category_capacities: dict,
    initial_agent_order: list,
):
    """
    Validate all inputs for fair_division_under_cardinality_constraints before the algorithm runs.

    Checks performed (in order):

    Type checks (correct input format):
    1. initial_agent_order is either None (use sorted agents as default) or a list.
    2. item_categories is a dict (correct type).
    3. category_capacities is a dict (correct type).
    4. item_categories values are all lists (correct type).

    Existence/non-empty checks:
    5. At least one agent exists in the instance.
    6. At least one category exists in item_categories.
    7. Every category in item_categories has at least one item (no empty category lists).

    Consistency checks (inputs match each other):
    8. If initial_agent_order is not None, it must be a valid permutation of the agents —
       contains each agent exactly once (no duplicates, no missing agents, no extra agents
       not in the instance). If None, this check is skipped.
    9. category_capacities keys match item_categories keys exactly (every category has a
       threshold, no threshold for a non-existent category).
    10. Every item listed in item_categories appears in the instance valuations.
    11. Every item in the instance valuations appears in at least one category (no uncategorised goods).
    12. No item appears in more than one category.

    Mathematical/feasibility checks:
    13. All valuation values are non-negative (>= 0).
    14. All thresholds k_h in category_capacities are positive integers.
    15. Each k_h satisfies k_h >= ceil(|C_h| / n) — the feasibility condition from the paper.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity
        for items and agents.
    :param item_categories: a dictionary mapping each category name (str) to a list of item
        names belonging to that category. Example: {'c1': ['m1', 'm2'], 'c2': ['m3']}.
    :param category_capacities: a dictionary mapping each category name (str) to the shared
        integer threshold k_h (max goods any agent may receive from that category).
        Example: {'c1': 1, 'c2': 2}.
    :param initial_agent_order: a list of agent names specifying the initial picking order,
        or None to use sorted(agents) as the default order.
        If a list, must contain each agent exactly once.
    :raises ValueError: if any of the above conditions are violated.

    # -------------------------------------------------------------------
    # Setup: base instance used across most tests
    # -------------------------------------------------------------------

    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> category_capacities = {'c1': 1}

    >>> # Valid inputs with explicit agent order — no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Bob'])

    >>> # Valid inputs with None — uses sorted agents as default, no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, None)

    # -------------------------------------------------------------------
    # Check 1: initial_agent_order is either None or a list (correct type).
    # -------------------------------------------------------------------

    >>> # [check 1] invalid — initial_agent_order is a tuple instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ('Alice', 'Bob'))
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 1] invalid — initial_agent_order is a set instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, {'Alice', 'Bob'})
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 2: item_categories is a dict (correct type).
    # -------------------------------------------------------------------

    >>> # [check 2] invalid — item_categories is a list instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, [('c1', ['m1', 'm2'])], category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 2] invalid — item_categories is None instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, None, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 3: category_capacities is a dict (correct type).
    # -------------------------------------------------------------------

    >>> # [check 3] invalid — category_capacities is a list instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, [('c1', 1)], ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 3] invalid — category_capacities is None instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, None, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 4: item_categories values are all lists (correct type).
    # -------------------------------------------------------------------

    >>> # [check 4] invalid — a category value is a tuple instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ('m1', 'm2')}, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 4] invalid — a category value is a set instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': {'m1', 'm2'}}, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 5] invalid — initial_agent_order is empty list (no agents given) raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, [])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 5] invalid — initial_agent_order is an empty list raises ValueError
    >>> # (empty valuations are rejected by the fairpyx framework before reaching this function)
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, [])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 6: At least one category exists in item_categories.
    # -------------------------------------------------------------------

    >>> # [check 6] invalid — item_categories is empty (no categories at all) raises ValueError
    >>> validate_fair_division_inputs(alloc, {}, {}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 7: Every category in item_categories has at least one item.
    # -------------------------------------------------------------------

    >>> # [check 7] invalid — category 'c1' has an empty item list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': []}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 7] invalid — one category is empty, one is not raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2'], 'c2': []}, {'c1': 1, 'c2': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 8: initial_agent_order is a valid permutation of the agents (skipped if None).
    # -------------------------------------------------------------------

    >>> # [check 8] valid — None skips the permutation check entirely, no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, None)

    >>> # [check 8] invalid — agent missing from initial_agent_order raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — duplicate agent in initial_agent_order raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Alice'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — agent in initial_agent_order not in instance raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Charlie'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — agent in initial_agent_order not in instance raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Bob', 'Charlie'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 9: category_capacities keys match item_categories keys exactly.
    # -------------------------------------------------------------------

    >>> # [check 9] invalid — category_capacities missing a threshold for an existing category raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1'], 'c2': ['m2']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 9] invalid — category_capacities has a key for a category not in item_categories raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1, 'c99': 2}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 9] invalid — category_capacities has a key for a category not in item_categories raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1, 'c2': 2, 'c99': 2}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 10: Every item listed in item_categories appears in the instance valuations.
    # -------------------------------------------------------------------

    >>> # [check 10] invalid — item 'm99' is in item_categories but not in instance valuations raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm99']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 11: Every item in the instance valuations appears in at least one category.
    # -------------------------------------------------------------------

    >>> # [check 11] invalid — item 'm2' is in the instance valuations but missing from all categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 12: No item appears in more than one category.
    # -------------------------------------------------------------------

    >>> # [check 12] invalid — item 'm1' appears in two categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2'], 'c2': ['m1']}, {'c1': 1, 'c2': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 13: All valuation values are non-negative (>= 0).
    # -------------------------------------------------------------------

    >>> # [check 13] valid — valuation of exactly 0 is allowed, no exception raised
    >>> valuations_zero = {'Alice': {'m1': 0, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}}
    >>> instance_zero = Instance(valuations=valuations_zero)
    >>> alloc_zero = AllocationBuilder(instance_zero)
    >>> validate_fair_division_inputs(alloc_zero, {'c1': ['m1', 'm2']}, {'c1': 1}, ['Alice', 'Bob'])

    >>> # [check 13] invalid — negative valuation raises ValueError
    >>> valuations_neg = {'Alice': {'m1': -1, 'm2': 3}, 'Bob': {'m1': 4, 'm2': 6}}
    >>> instance_neg = Instance(valuations=valuations_neg)
    >>> alloc_neg = AllocationBuilder(instance_neg)
    >>> validate_fair_division_inputs(alloc_neg, {'c1': ['m1', 'm2']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 14: All thresholds k_h in category_capacities are positive integers.
    # -------------------------------------------------------------------

    >>> # [check 14] invalid — k_h=0 is not a positive integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 0}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h=-1 is not a positive integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': -1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h=1.5 is not an integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1.5}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h='1' is a string not an integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': '1'}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 15: Each k_h satisfies k_h >= ceil(|C_h| / n) — feasibility condition.
    # -------------------------------------------------------------------

    >>> # [check 15] invalid — k_h=1 < ceil(3/2)=2 for 3 items and 2 agents raises ValueError
    >>> valuations_3 = {'Alice': {'m1': 5, 'm2': 3, 'm3': 1}, 'Bob': {'m1': 1, 'm2': 3, 'm3': 5}}
    >>> instance_3 = Instance(valuations=valuations_3)
    >>> alloc_3 = AllocationBuilder(instance_3)
    >>> validate_fair_division_inputs(alloc_3, {'c1': ['m1', 'm2', 'm3']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 15] valid — k_h=2 == ceil(3/2)=2, exactly on the boundary — no exception raised
    >>> validate_fair_division_inputs(alloc_3, {'c1': ['m1', 'm2', 'm3']}, {'c1': 2}, ['Alice', 'Bob'])
    """

    logger.debug("Validating inputs: %d agents, %d categories", len(list(alloc.instance.agents)), len(item_categories) if isinstance(item_categories, dict) else "?")

    # Check 1: initial_agent_order must be None or a list
    if initial_agent_order is not None and not isinstance(initial_agent_order, list):
        logger.warning("Check 1 failed: initial_agent_order is %s, expected None or list", type(initial_agent_order).__name__)
        raise ValueError(
            f"initial_agent_order must be None or a list, got {type(initial_agent_order).__name__}"
        )

    # Check 2: item_categories must be a dict
    if not isinstance(item_categories, dict):
        logger.warning("Check 2 failed: item_categories is %s, expected dict", type(item_categories).__name__)
        raise ValueError(
            f"item_categories must be a dict, got {type(item_categories).__name__}"
        )

    # Check 3: category_capacities must be a dict
    if not isinstance(category_capacities, dict):
        logger.warning("Check 3 failed: category_capacities is %s, expected dict", type(category_capacities).__name__)
        raise ValueError(
            f"category_capacities must be a dict, got {type(category_capacities).__name__}"
        )

    # Check 4: every value in item_categories must be a list
    for cat, cat_items in item_categories.items():
        if not isinstance(cat_items, list):
            logger.warning("Check 4 failed: item_categories[%r] is %s, expected list", cat, type(cat_items).__name__)
            raise ValueError(
                f"item_categories[{cat!r}] must be a list, got {type(cat_items).__name__}"
            )

    # Check 5: at least one agent (fires when initial_agent_order=[] is passed)
    agents_to_use = initial_agent_order if initial_agent_order is not None else list(alloc.instance.agents)
    if len(agents_to_use) == 0:
        logger.warning("Check 5 failed: no agents provided")
        raise ValueError("At least one agent must exist in the instance")

    # Check 6: at least one category
    if len(item_categories) == 0:
        logger.warning("Check 6 failed: item_categories is empty")
        raise ValueError("item_categories must contain at least one category")

    # Check 7: no empty category lists
    for cat, cat_items in item_categories.items():
        if len(cat_items) == 0:
            logger.warning("Check 7 failed: category %r has an empty item list", cat)
            raise ValueError(f"Category {cat!r} has an empty item list")

    # Check 8: if provided, initial_agent_order must be an exact permutation of the agents
    if initial_agent_order is not None:
        instance_agents = set(alloc.instance.agents)
        order_set = set(initial_agent_order)
        if len(initial_agent_order) != len(order_set) or order_set != instance_agents:
            logger.warning("Check 8 failed: initial_agent_order %s is not a permutation of agents %s", initial_agent_order, sorted(instance_agents))
            raise ValueError(
                f"initial_agent_order must be a permutation of the agents. "
                f"Got {initial_agent_order}, expected {sorted(instance_agents)}"
            )

    # Check 9: category_capacities keys must match item_categories keys exactly
    if set(category_capacities.keys()) != set(item_categories.keys()):
        logger.warning("Check 9 failed: category_capacities keys %s != item_categories keys %s", set(category_capacities.keys()), set(item_categories.keys()))
        raise ValueError(
            f"category_capacities keys {set(category_capacities.keys())} must match "
            f"item_categories keys {set(item_categories.keys())}"
        )

    # Check 10: every item listed in item_categories must exist in the instance
    instance_items = set(alloc.instance.items)
    for cat, cat_items in item_categories.items():
        for item in cat_items:
            if item not in instance_items:
                logger.warning("Check 10 failed: item %r in category %r does not exist in the instance", item, cat)
                raise ValueError(
                    f"Item {item!r} in category {cat!r} does not exist in the instance"
                )

    # Check 11: every item in the instance must appear in at least one category
    all_categorised = set(item for cat_items in item_categories.values() for item in cat_items)
    for item in alloc.instance.items:
        if item not in all_categorised:
            logger.warning("Check 11 failed: item %r is in the instance but not in any category", item)
            raise ValueError(
                f"Item {item!r} exists in the instance but is not listed in any category"
            )

    # Check 12: no item may appear in more than one category
    seen_items: set = set()
    for cat, cat_items in item_categories.items():
        for item in cat_items:
            if item in seen_items:
                logger.warning("Check 12 failed: item %r appears in more than one category", item)
                raise ValueError(f"Item {item!r} appears in more than one category")
            seen_items.add(item)

    # Check 13: all valuation values must be non-negative
    for agent in alloc.instance.agents:
        for item in alloc.instance.items:
            if alloc.instance.agent_item_value(agent, item) < 0:
                logger.warning("Check 13 failed: agent %r has negative valuation for item %r", agent, item)
                raise ValueError(
                    f"Valuation of agent {agent!r} for item {item!r} is negative"
                )

    # Check 14: all k_h must be positive integers (bool is excluded: isinstance(True, int) is True)
    for cat, k_h in category_capacities.items():
        if isinstance(k_h, bool) or not isinstance(k_h, int) or k_h <= 0:
            logger.warning("Check 14 failed: category_capacities[%r] = %r is not a positive integer", cat, k_h)
            raise ValueError(
                f"category_capacities[{cat!r}] = {k_h!r} must be a positive integer"
            )

    # Check 15: k_h >= ceil(|C_h| / n) — feasibility condition from the paper
    n = len(list(alloc.instance.agents))
    for cat, cat_items in item_categories.items():
        k_h = category_capacities[cat]
        min_required = math.ceil(len(cat_items) / n)
        if k_h < min_required:
            logger.warning("Check 15 failed: category %r has k_h=%d < ceil(%d/%d)=%d — feasibility violated", cat, k_h, len(cat_items), n, min_required)
            raise ValueError(
                f"category_capacities[{cat!r}] = {k_h} < ceil({len(cat_items)}/{n}) = {min_required}. "
                f"The feasibility condition k_h >= ceil(|C_h|/n) is violated."
            )

    logger.debug("All 15 validation checks passed")

if __name__ == "__main__":
    # Demo: run the algorithm on Example 10 (3 agents, 6 goods, 2 categories, k_h=1).
    # This example produces two overlapping envy cycles after C2, showing the full cycle-elimination logic.
    import logging
    from fairpyx import divide

    logging.basicConfig(
        level=logging.DEBUG,   # change to INFO to hide per-pick detail
        format="%(levelname)-5s  %(name)s: %(message)s",
    )

    valuations = {
        'a1': {'g1': 9, 'g2': 2, 'g3': 1, 'g4': 1, 'g5': 10, 'g6': 0},
        'a2': {'g1': 10, 'g2': 8, 'g3': 1, 'g4': 10, 'g5': 1, 'g6': 0},
        'a3': {'g1': 10, 'g2': 9, 'g3': 5, 'g4': 8, 'g5': 1, 'g6': 7},
    }
    item_categories = {'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']}
    category_capacities = {'C1': 1, 'C2': 1}

    instance = Instance(valuations=valuations)
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories=item_categories,
        category_capacities=category_capacities,
        initial_agent_order=['a1', 'a2', 'a3'],
    )

    # Validation failure demo: k_h=0 violates check 14 — a WARNING is logged before the ValueError
    print("\n--- Validation failure demo ---")
    from fairpyx import AllocationBuilder
    bad_alloc = AllocationBuilder(instance)
    try:
        validate_fair_division_inputs(bad_alloc, item_categories, {'C1': 0, 'C2': 1}, ['a1', 'a2', 'a3'])
    except ValueError as e:
        print("Caught ValueError:", e)