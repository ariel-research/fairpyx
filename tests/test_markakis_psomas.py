"""
An implementation of the algorithms in:
"On Worst-Case Allocations in the Presence of Indivisible Goods"
by Evangelos Markakis and Christos-Alexandros Psomas (2011).
https://link.springer.com/chapter/10.1007/978-3-642-25510-6_24
http://pages.cs.aueb.gr/~markakis/research/wine11-Vn.pdf
Programmer: Ibrahem Hurani
Date: 2025-05-06
"""

import pytest
from fairpyx import Instance, divide,AllocationBuilder
from fairpyx.algorithms.markakis_psomas import algorithm1_worst_case_allocation, compute_vn


def test_example_1():
    instance = Instance(
        valuations={"A": {"1": 6, "2": 3, "3": 1}, "B": {"1": 2, "2": 5, "3": 5}}
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    assert set(alloc["A"]) == {"1"}
    assert set(alloc["B"]) == {"2","3"}
    


def test_example_2():
    instance = Instance(
        valuations={
            "A": {"1": 7, "2": 2, "3": 1, "4": 1},
            "B": {"1": 3, "2": 6, "3": 1, "4": 2},
            "C": {"1": 2, "2": 3, "3": 5, "4": 5},
        }
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    for agent in instance.agents:
        bundle = alloc[agent]
        values = [instance.agent_item_value(agent, item) for item in bundle]
        total = sum(instance.agent_item_value(agent, item) for item in instance.items)
        alpha = max(values) / total if total > 0 else 0
        Vn_alpha = compute_vn(alpha, len(instance.agents))
        Vn_alpha_i=Vn_alpha*total
        assert sum(values) >= Vn_alpha_i
        assert set(alloc["A"])=={"1"}
        assert set(alloc["B"])=={"2"}
        assert set(alloc["C"])=={"3","4"}


def test_empty():
    alloc=algorithm1_worst_case_allocation(alloc=None)
    assert alloc== {}


def test_single_agent_all_items():
    instance = Instance(valuations={"A": {"1": 10, "2": 5}})
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    assert set(alloc["A"]) == {"1", "2"}


def test_large_random_instance():
    import random
    random.seed(42)
    agents = [f"A{i}" for i in range(10)]
    items = [str(i) for i in range(15)]
    valuations = {
        agent: {item: random.randint(1, 10) for item in items}
        for agent in agents
    }
    instance = Instance(valuations)
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
    for agent in agents:
        bundle = alloc[agent]
        values = [instance.agent_item_value(agent, item) for item in bundle]
        total = sum(instance.agent_item_value(agent, item) for item in instance.items)
        alpha = max(values) / total if total > 0 else 0
        Vn_alpha = compute_vn(alpha, len(agents))
        assert sum(values) >= Vn_alpha* total

def test_large():
    instance = Instance(
        valuations={
            "Alice": {"c1": 8, "c2": 6, "c3": 10},
            "Bob": {"c1": 8, "c2": 10, "c3": 6},
            "Chana": {"c1": 6, "c2": 8, "c3": 10},
            "Dana": {"c1": 6, "c2": 8, "c3": 10}
        }
    )
    alloc = divide(algorithm=algorithm1_worst_case_allocation, instance=instance)

    # בדיקה שהוקצתה חבילה לכל סוכן
    assert set(alloc.keys()) == {"Alice", "Bob", "Chana", "Dana"}

    # בדיקה שאין חפיפה בין פריטים של סוכנים שונים
    all_items = [item for bundle in alloc.values() for item in bundle]
    assert len(all_items) == len(set(all_items)), "Duplicate item assigned to multiple agents"


