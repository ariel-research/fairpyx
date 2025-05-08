"""
Test cases for the Repeated Fair Allocation of Indivisible Items algorithm.
This module contains unit tests for the algorithm, including edge cases and fixed small examples.
It also includes randomized property tests to ensure the algorithm's correctness and fairness.
For large scale tests, it ensures that the algorithm maintains the EF1 condition and global EF property.

Programmer: Shaked Shvartz
Since: 2025-05
"""

import pytest
import random
from fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items import (
    two_agents_two_rounds,
    two_agents_even_rounds,
    multi_agent_cyclic
)
from fairpyx import Instance, AllocationBuilder


def make_instance(agent_caps, item_caps, valuations):
    inst = Instance(agent_capacities=agent_caps,
                    item_capacities=item_caps,
                    valuations=valuations)
    return AllocationBuilder(inst)

# --- Edge case tests ---

def test_empty_agents():
    """
    Verify that calling two_agents_two_rounds with no agents raises ValueError.
    """
    agent_caps = {}
    item_caps = {'a':1}
    valuations = {}
    alloc = make_instance(agent_caps, item_caps, valuations)
    with pytest.raises(ValueError):
        two_agents_two_rounds(alloc)


def test_empty_items():
    """
    When there are no items, the allocation should return empty bundles for each agent.
    """
    agent_caps = {'A':1,'B':1}
    item_caps = {}
    valuations = {'A':{},'B':{}}
    alloc = make_instance(agent_caps, item_caps, valuations)
    result = two_agents_two_rounds(alloc)
    # Each bundle list must be empty
    for round_id, bundles in result.items():
        assert bundles == {'A': [], 'B': []}


def test_invalid_k_even_rounds():
    """
    Ensure ValueError is raised if k is not even for two_agents_even_rounds.
    """
    agent_caps = {'A':2,'B':2}
    item_caps = {'x':2,'y':2}
    valuations = {'A':{'x':1,'y':1}, 'B':{'x':1,'y':1}}
    alloc = make_instance(agent_caps, item_caps, valuations)
    with pytest.raises(ValueError):
        two_agents_even_rounds(alloc, k=3)

# --- Fixed small example tests ---

def test_two_agents_two_rounds_example():
    """
    Fixed example for n=2, k=2: verify the exact EF1 per-round result.
    """
    agent_caps = {'A':2,'B':2}
    item_caps = {'a':1,'b':1}
    vals = {'A':{'a':4,'b':1}, 'B':{'a':2,'b':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_two_rounds(alloc)
    # Round 1: A picks 'a', B picks 'b'
    assert result[1] == {'A': ['a'], 'B': ['b']}
    # Round 2: A picks 'b', B picks 'a'
    assert result[2] == {'A': ['b'], 'B': ['a']}


def test_two_agents_even_rounds_example():
    """
    Fixed example for n=2, k=4: ensure EF1 holds in each round.
    """
    agent_caps = {'A':4,'B':4}
    item_caps = {'x':2,'y':2}
    vals = {'A':{'x':3,'y':1}, 'B':{'x':1,'y':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    # Check EF1: difference in utility at most max item value
    max_item = max(max(vals['A'].values()), max(vals['B'].values()))
    for s, bundles in result.items():
        uA = sum(vals['A'][it] for it in bundles['A'])
        uB = sum(vals['B'][it] for it in bundles['B'])
        assert abs(uA - uB) <= max_item


def test_multi_agent_cyclic_example():
    """
    Fixed example for n=3, k=3: exact cyclic shift of base bundles.
    """
    agent_caps = {'1':2,'2':2,'3':2}
    item_caps = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1}
    vals = {i:{ch:1 for ch in item_caps} for i in ['1','2','3']}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = multi_agent_cyclic(alloc, k=3)
    # Round 1
    assert result[1] == {'1': ['a','b'], '2': ['c','d'], '3': ['e','f']}
    # Round 2
    assert result[2] == {'1': ['c','d'], '2': ['e','f'], '3': ['a','b']}
    # Round 3
    assert result[3] == {'1': ['e','f'], '2': ['a','b'], '3': ['c','d']}
    
    
def test_two_agents_two_rounds_pareto_optimal():
    """
    Test Pareto-Optimality of the two_agents_two_rounds allocation.
    A division π is Pareto-Optimal if there is no other allocation π' such that 
    every agent i receives u_i(π'_i) ≥ u_i(π_i) and at least one agent gets a
    strict improvement.
    """
    agent_caps = {'A':2,'B':2}
    item_caps = {'a':1,'b':1}
    vals = {'A':{'a':4,'b':1}, 'B':{'a':2,'b':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_two_rounds(alloc)
    
    # Calculate utilities for current allocation
    a_utility = sum(sum(vals['A'][item] for item in result[round]['A']) for round in result)
    b_utility = sum(sum(vals['B'][item] for item in result[round]['B']) for round in result)
    
    # Check all possible alternative allocations
    possible_allocations = [
        # Round 1: A gets 'a', B gets 'b'; Round 2: A gets 'b', B gets 'a' (current allocation)
        {'A': ['a', 'b'], 'B': ['b', 'a']},
        # Round 1: A gets 'b', B gets 'a'; Round 2: A gets 'a', B gets 'b'
        {'A': ['b', 'a'], 'B': ['a', 'b']},
        # Round 1: A gets 'a', B gets 'b'; Round 2: A gets 'a', B gets 'b'
        {'A': ['a', 'a'], 'B': ['b', 'b']},
        # Round 1: A gets 'b', B gets 'a'; Round 2: A gets 'b', B gets 'a'
        {'A': ['b', 'b'], 'B': ['a', 'a']},
    ]
    
    # For each possible allocation, check if it's better for at least one agent
    # without being worse for any other agent
    for alt_alloc in possible_allocations:
        # Calculate alternative utilities
        alt_a_utility = sum(vals['A'][item] for item in alt_alloc['A'])
        alt_b_utility = sum(vals['B'][item] for item in alt_alloc['B'])
        
        # Check if this alternative is a Pareto improvement
        a_better = alt_a_utility > a_utility
        b_better = alt_b_utility > b_utility
        a_same_or_better = alt_a_utility >= a_utility
        b_same_or_better = alt_b_utility >= b_utility
        
        # If both agents are same or better AND at least one is strictly better,
        # then the current allocation is NOT Pareto-optimal
        is_pareto_improvement = (a_same_or_better and b_same_or_better and 
                                (a_better or b_better))
        
        # Verify no Pareto improvements exist
        assert not is_pareto_improvement, f"Found Pareto improvement: {alt_alloc}"
    # If we reach here, the allocation is Pareto-optimal

# --- Randomized property tests ---

def test_two_agents_even_rounds_random():
    """
    Randomized test for n=2, k=4: verify EF1 per-round.
    Ensures for each round s: |uA - uB| ≤ max_item value.
    """
    agent_caps = {'A':2,'B':2}
    item_caps = {'x':2,'y':2}
    vals = {
        'A':{'x':random.randint(1,10),'y':random.randint(1,10)},
        'B':{'x':random.randint(1,10),'y':random.randint(1,10)}
    }
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    max_item = max(max(vals['A'].values()), max(vals['B'].values()))
    for s, bundles in result.items():
        uA = sum(vals['A'][it] for it in bundles['A'])
        uB = sum(vals['B'][it] for it in bundles['B'])
        assert abs(uA - uB) <= max_item


def test_multi_agent_cyclic_random():
    """
    Randomized test for n>2: verify global EF by equal total utility.
    """
    n, m, k = 4, 8, 4
    agents = [str(i) for i in range(n)]
    items = [chr(ord('a')+i) for i in range(m)]
    agent_caps = {i:1 for i in agents}
    item_caps = {it:1 for it in items}
    vals = {i:{it:random.randint(1,10) for it in items} for i in agents}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = multi_agent_cyclic(alloc, k)
    totals = {i:0 for i in agents}
    for s in range(1, k+1):
        for i in agents:
            totals[i] += sum(vals[i][it] for it in result[s][i])
    # All totals must be identical → EF overall
    assert len(set(totals.values())) == 1

# --- Large-scale tests ---

def test_two_agents_even_rounds_large():
    """
    Large input test for n=2, k=100: ensure EF1 per-round on larger scale.
    """
    agent_caps = {'A':100,'B':100}
    # Create 50 items with capacity 4 each → total copies = 200 per agent
    items = [f'i{i}' for i in range(50)]
    item_caps = {it:4 for it in items}
    vals = {
        'A': {it: random.randint(1,100) for it in items},
        'B': {it: random.randint(1,100) for it in items}
    }
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=100)
    # Verify EF1 per-round on each of 100 rounds
    for s, bundles in result.items():
        uA = sum(vals['A'][it] for it in bundles['A'])
        uB = sum(vals['B'][it] for it in bundles['B'])
        max_item = max(max(vals['A'].values()), max(vals['B'].values()))
        assert abs(uA - uB) <= max_item


def test_multi_agent_cyclic_large():
    """
    Large-scale test for n=6, k=60: verify EF overall on larger input.
    """
    n, m, k = 6, 30, 60
    agents = [str(i) for i in range(n)]
    items = [chr(ord('a')+i%26) + str(i//26) for i in range(m)]
    agent_caps = {i:1 for i in agents}
    item_caps = {it:1 for it in items}
    vals = {i:{it: random.randint(1,50) for it in items} for i in agents}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = multi_agent_cyclic(alloc, k)
    totals = {i:0 for i in agents}
    for s in range(1, k+1):
        for i in agents:
            totals[i] += sum(vals[i][it] for it in result[s][i])
    # All totals must be equal → EF overall holds
    assert len(set(totals.values())) == 1
    
    
def test_cyclic_exact_multiple_rounds():
    """
    Exact output test for multi_agent_cyclic with multiple rounds.
    Given base partitions A1={a,b}, A2={c,d}, A3={e,f}, and k=6,
    expect two full cycles of the 3-round pattern.
    """
    # Setup
    agent_caps = {'1':2,'2':2,'3':2}
    item_caps = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1}
    vals = {i:{ch:1 for ch in item_caps} for i in ['1','2','3']}
    alloc = make_instance(agent_caps, item_caps, vals)
    k = 6
    result = multi_agent_cyclic(alloc, k)
    # Expected two cycles of the 3-round pattern
    base = {
        1: {'1': ['a','b'], '2': ['c','d'], '3': ['e','f']},
        2: {'1': ['c','d'], '2': ['e','f'], '3': ['a','b']},
        3: {'1': ['e','f'], '2': ['a','b'], '3': ['c','d']}
    }
    expected = {}
    for cycle_index in range(2):
        for r in range(1, 4):
            round_number = cycle_index * 3 + r
            expected[round_number] = base[r]
    assert result == expected

def test_cyclic_exact_many_rounds():
    """
    Exact output test for multi_agent_cyclic with many rounds (n=3, k=300).
    Expect 100 full cycles of the 3-round base pattern.
    """
    agent_caps = {'1':2,'2':2,'3':2}
    item_caps = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1}
    vals = {i:{ch:1 for ch in item_caps} for i in ['1','2','3']}
    alloc = make_instance(agent_caps, item_caps, vals)
    k = 300  # 100 cycles
    result = multi_agent_cyclic(alloc, k)
    base = {
        1: {'1': ['a','b'], '2': ['c','d'], '3': ['e','f']},
        2: {'1': ['c','d'], '2': ['e','f'], '3': ['a','b']},
        3: {'1': ['e','f'], '2': ['a','b'], '3': ['c','d']}
    }
    expected = {}
    cycles = 100
    for cycle_index in range(cycles):
        for r in range(1, 4):
            round_number = cycle_index * 3 + r
            expected[round_number] = base[r]
    assert result == expected

if __name__ == '__main__':
    pytest.main()
