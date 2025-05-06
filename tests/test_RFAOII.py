"""
Test cases for the Repeated Fair Allocation of Indivisible Items algorithm.
This module contains unit tests for the algorithm, including edge cases and fixed small examples.

It also includes randomized property tests to ensure the algorithm's correctness and fairness.
    
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
    # אין סוכנים
    agent_caps = {}
    item_caps = {'a':1}
    valuations = {}
    alloc = make_instance(agent_caps, item_caps, valuations)
    with pytest.raises(ValueError):
        two_agents_two_rounds(alloc)


def test_empty_items():
    # אין פריטים
    agent_caps = {'A':1,'B':1}
    item_caps = {}
    valuations = {'A':{},'B':{}}
    alloc = make_instance(agent_caps, item_caps, valuations)
    # אמור להחזיר הקצאה ריקה בכל סבב
    result = two_agents_two_rounds(alloc)
    assert all(bundle == [] for bundle in result.values())


def test_invalid_k_even_rounds():
    # k זוגי אך לא שווה לסך הקיבולות
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
    # מצופה: A picks a then b; B ההפך
    assert result[1]['A'] == ['a'] and result[2]['A'] == ['b']
    assert result[1]['B'] == ['b'] and result[2]['B'] == ['a']


def test_two_agents_even_rounds_example():
    agent_caps = {'A':4,'B':4}
    item_caps = {'x':2,'y':2}
    vals = {'A':{'x':3,'y':1}, 'B':{'x':1,'y':3}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    # בכל סבב EF1
    for bundle in result.values():
        assert len(bundle) == 1


def test_multi_agent_cyclic_example():
    agent_caps = {'1':2,'2':2,'3':2}
    item_caps = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1}
    vals = {i:{ch:1 for ch in item_caps} for i in ['1','2','3']}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = multi_agent_cyclic(alloc, k=3)
    # כל סוכן מקבל בכל סבב בדיוק פריט אחד
    for s in [1,2,3]:
        assert all(len(result[s][i]) == 1 for i in ['1','2','3'])

# --- Randomized property tests ---


def test_two_agents_even_rounds_random():
    # בדיקה שרק התוצאה מקיימת EF1 ולא קנאה גלובלית
    agent_caps = {'A':2,'B':2}
    item_caps = {'x':2,'y':2}
    vals = {'A':{'x':random.randint(1,10),'y':random.randint(1,10)},
            'B':{'x':random.randint(1,10),'y':random.randint(1,10)}}
    alloc = make_instance(agent_caps, item_caps, vals)
    result = two_agents_even_rounds(alloc, k=4)
    # בדוק EF1 בכל סבב:
    # הפרש התועלות לכל סוכן בכל סבב <= max פרט
    for s, bundles in result.items():
        uA = sum(vals['A'][item] for item in bundles['A'])
        uB = sum(vals['B'][item] for item in bundles['B'])
        max_item = max(max(vals['A'].values()), max(vals['B'].values()))
        assert abs(uA - uB) <= max_item


def test_multi_agent_cyclic_random():
    # בדיקה של EF גלובלית: תועלת מצטברת זהה
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
