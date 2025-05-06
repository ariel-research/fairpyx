"""
Test cases for the Repeated Fair Allocation of Indivisible Items algorithm.
This module contains unit tests for the algorithm, including edge cases and fixed small examples.

It also includes randomized property tests to ensure the algorithm's correctness and fairness.
    
Programmer: Shaked Shvartz
Since: 2025-05
"""



from fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items import *
from fairpyx import Instance, AllocationBuilder
import pytest
import random



def make_instance(agent_caps, item_caps, valuations):
    inst = Instance(agent_capacities=agent_caps,
                    item_capacities=item_caps,
                    valuations=valuations)
    return AllocationBuilder(inst)

# --- Edge case tests ---

def test_empty_agents():
    # no agents
    agent_caps = {}
    item_caps = {'a':1}
    valuations = {}
    alloc = make_instance(agent_caps, item_caps, valuations)
    with pytest.raises(ValueError):
        two_agents_two_rounds(alloc)


def test_empty_items():
    # no items
    agent_caps = {'A':1,'B':1}
    item_caps = {}
    valuations = {'A':{},'B':{}}
    alloc = make_instance(agent_caps, item_caps, valuations)
    result = two_agents_two_rounds(alloc)
    assert all(bundle == [] for bundle in result.values())


def test_invalid_k_even_rounds():
    # k is not even
    agent_caps = {'A':2,'B':2}
    item_caps = {'x':2,'y':2}
    valuations = {'A':{'x':1,'y':1},'B':{'x':1,'y':1}}
    alloc = make_instance(agent_caps, item_caps, valuations)
    with pytest.raises(ValueError):
        two_agents_even_rounds(alloc, k=3)

# --- Fixed small example tests ---

def test_two_agents_two_rounds_example():
    agent_caps = {'A':2,'B':2}
    item_caps = {'a':1,'b':1}
    vals = {'A':{'a':4,'b':1}, 'B':{'a':2,'b':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_two_rounds(alloc)
    # expected allocation:
    # 1: A→{a},B→{b}
    # 2: A→{b},B→{a}
    assert result[1]['A'] == ['a'] and result[2]['A'] == ['b']
    assert result[1]['B'] == ['b'] and result[2]['B'] == ['a']


def test_two_agents_even_rounds_example():
    agent_caps = {'A':4,'B':4}
    item_caps = {'x':2,'y':2}
    vals = {'A':{'x':3,'y':1}, 'B':{'x':1,'y':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    # check that the allocation is EF1 in each round
    for bundle in result.values():
        assert len(bundle) == 1


def test_multi_agent_cyclic_example():
    agent_caps = {'1':2,'2':2,'3':2}
    item_caps = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1}
    vals = {i:{ch:1 for ch in item_caps} for i in ['1','2','3']}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = multi_agent_cyclic(alloc, k=3)
    # expected allocation:
    # 1: 1→{a,b},2→{c,d},3→{e,f}
    # 2: 1→{c,d},2→{e,f},3→{a,b}
    # 3: 1→{e,f},2→{a,b},3→{c,d}
    assert result[1]['1'] == ['a','b'] and result[2]['1'] == ['c','d'] and result[3]['1'] == ['e','f']
    assert result[1]['2'] == ['c','d'] and result[2]['2'] == ['e','f'] and result[3]['2'] == ['a','b']
    assert result[1]['3'] == ['e','f'] and result[2]['3'] == ['a','b'] and result[3]['3'] == ['c','d']

# --- Randomized property tests ---


def test_two_agents_even_rounds_random():
    # check EF1 in each round:
    # check that the allocation is EF1 in each round
    agent_caps = {'A':2,'B':2}
    item_caps = {'x':2,'y':2}
    vals = {'A':{'x':random.randint(1,10),'y':random.randint(1,10)},
            'B':{'x':random.randint(1,10),'y':random.randint(1,10)}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    for s, bundles in result.items():
        uA = sum(vals['A'][item] for item in bundles['A'])
        uB = sum(vals['B'][item] for item in bundles['B'])
        max_item = max(max(vals['A'].values()), max(vals['B'].values()))
        assert abs(uA - uB) <= max_item


def test_multi_agent_cyclic_random():
    # check global fairness:
    # check that all agents have the same total value after k rounds
    n = 4
    m = 8
    k = 4
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
    assert len(set(totals.values())) == 1

if __name__ == '__main__':
    pytest.main()
