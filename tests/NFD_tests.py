"""
Tests for the implementation of the algorithms in:

"Efficient Nearly-Fair Division with Capacity Constraints", by A. Hila Shoshan,  Noam Hazon,  Erel Segal-Halevi
(2023), https://arxiv.org/abs/2205.07779

Programmer: Matan Ziv.
Date: 2025-04-27.
"""

import pytest
from fairpyx import Instance, divide, AllocationBuilder
from fairpyx.algorithms.NFD import Nearly_Fair_Division, is_EF11, is_EF1, category_w_max_two_agents
import random
from typing import Any, Dict, List
from itertools import combinations



def generate_random_instance(seed=42):
    random.seed(seed)
    agents = ["A", "B"]
    num_items = 200
    categories = [f"cat{i}" for i in range(10)]
    items = [f"o{i}" for i in range(num_items)]
    valuations = {
        a: {i: random.randint(-2, 2) for i in items} for a in agents
    }
    capacities = {i: 1 for i in items}
    item_cats = {i: random.choice(categories) for i in items}
    cat_caps = {cat: 100 for cat in categories}
    return Instance(
        valuations=valuations,
        agent_capacities={a: num_items for a in agents},
        item_capacities=capacities,
        item_categories=item_cats,
        category_capacities=cat_caps,
    )


def is_pareto_optimal(instance: "Instance",
                      allocation: Dict[Any, List[Any]],
                      abs_tol: float = 1e-9) -> bool:
    """
    Local Pareto-optimality check (one-for-one swaps).

    A true return value means **no** swap of a single item between any two
    agents (restricted to items in the same category when categories exist)
    can raise *both* of their utilities.

    Parameters
    ----------
    instance    : Instance
        The problem instance (provides utilities, categories, weights, …).
    allocation  : dict[agent -> list[item]]
        The current integral allocation.
    abs_tol     : float
        Numerical slack to ignore tiny rounding errors.

    Returns
    -------
    bool
        True  ⇢ allocation is Pareto-optimal w.r.t. single exchanges.
        False ⇢ found a profitable swap ⇒ not Pareto-optimal.
            Examples
    --------
    We use a tiny helper class so the doctests are self-contained:

    >>> class ToyInst:
    ...     def __init__(self, valuations, categories=None):
    ...         self.valuations = valuations            # dict[agent][item] → v
    ...         self.agents = list(valuations.keys())
    ...         self.item_categories = categories or {} # {} ⇒ all same cat.
    ...     def agent_item_value(self, agent, item):
    ...         return self.valuations[agent][item]
    ...     def agent_bundle_value(self, agent, bundle):
    ...         return sum(self.agent_item_value(agent, it) for it in bundle)

    1. **Already optimal** – symmetric “mirror” utilities, each agent holds
       her favourite item:

    >>> inst = ToyInst({'A': {'g1': 5, 'g2': 1},
    ...                 'B': {'g1': 1, 'g2': 5}})
    >>> alloc = {'A': ['g1'], 'B': ['g2']}
    >>> is_pareto_optimal(inst, alloc)
    True

    2. **Profitable swap exists** – same instance but items mis-assigned:

    >>> alloc_bad = {'A': ['g2'], 'B': ['g1']}
    >>> is_pareto_optimal(inst, alloc_bad)
    False

    3. **Chores (negative utilities) already optimal**:

    >>> inst2 = ToyInst({'A': {'c1': -4, 'c2': -1},
    ...                  'B': {'c1': -1, 'c2': -4}})
    >>> alloc2 = {'A': ['c2'], 'B': ['c1']}
    >>> is_pareto_optimal(inst2, alloc2)
    True

    4. **Chores need swapping** – same instance but chores flipped:

    >>> alloc2_bad = {'A': ['c1'], 'B': ['c2']}
    >>> is_pareto_optimal(inst2, alloc2_bad)
    False
    """

    # Current utility of each agent for her bundle
    u = {a: instance.agent_bundle_value(a, allocation[a])
         for a in allocation}                     # :contentReference[oaicite:0]{index=0}

    # Check every unordered pair of agents
    for i, j in combinations(allocation.keys(), 2):
        for x in allocation[i]:                  # item from i
            for y in allocation[j]:              # item from j
                # If categories are defined, require them to match
                if instance.item_categories and \
                   instance.item_categories[x] != instance.item_categories[y]:
                    continue

                # Utilities after swapping x ↔ y
                new_u_i = u[i] - instance.agent_item_value(i, x) \
                                + instance.agent_item_value(i, y)
                new_u_j = u[j] - instance.agent_item_value(j, y) \
                                + instance.agent_item_value(j, x)  # :contentReference[oaicite:1]{index=1}

                # Strict improvement for *both* agents?
                if (new_u_i > u[i] + abs_tol) and (new_u_j > u[j] + abs_tol):
                    return False     # profitable exchange exists

    return True                      # no profitable exchange found


# ---------------------------
# Edge Cases
# ---------------------------
def test_single_item():
    """
    Tests an instance with a single item.
    Ensures the item is allocated and respects capacity constraints.
    """
    instance = Instance(
        valuations={"A": {"x": 10}, "B": {"x": 10}},
        agent_capacities={"A": 1, "B": 1},
        item_capacities={"x": 1},
        item_categories={"x": "cat1"},
        category_capacities={"cat1": 1},
    )
    allocation = divide(Nearly_Fair_Division, instance)
    assert sum(len(v) for v in allocation.values()) == 1
    assert any("x" in bundle for bundle in allocation.values())

def test_empty_instance_raises():
    """
    Tests that creating an instance with no items raises AssertionError.
    """
    with pytest.raises(AssertionError):
        instance = Instance(
            valuations={"A": {}, "B": {}},
            agent_capacities={"A": 0, "B": 0},
            item_capacities={},               # <- No items, so `.items` ends up as None
            item_categories={},
            category_capacities={},
        )
        divide(Nearly_Fair_Division, instance)


# ---------------------------
# Large Inputs
# ---------------------------
def test_large_balanced():
    """
    Tests a large instance with 200 items and 10 categories.
    Ensures all items are allocated and each agent respects category constraints, EF1\EF11, and PO.
    """
    agents = ["A", "B"]
    items = {f"i{k}": k%3 for k in range(200)}
    instance = Instance(
        valuations={a: items.copy() for a in agents},
        agent_capacities={"A": 200, "B": 200},
        item_capacities={f"i{k}": 1 for k in range(200)},
        item_categories={f"i{k}": f"cat{k%10}" for k in range(200)},
        category_capacities={f"cat{k}": 10 for k in range(10)},
    )
    allocation = divide(Nearly_Fair_Division, instance)
    alloc = AllocationBuilder(instance)
    alloc = Nearly_Fair_Division(alloc)
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == 200
    for agent in agents:
        for cat in range(10):
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == f"cat{cat}"]
            assert len(items_in_cat) <= 10

    assert (is_EF11(alloc.instance, alloc.bundles) is None or
            is_EF1(alloc) is None)
    assert is_pareto_optimal(instance, allocation)


# ---------------------------
# Random Inputs
# ---------------------------
def test_random_instance():
    instance = generate_random_instance()
    print("Random Instance:" , instance)
    allocation = divide(Nearly_Fair_Division, instance)
    print("Allocation:", allocation)
    alloc = AllocationBuilder(instance)
    alloc = Nearly_Fair_Division(alloc)
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == len(instance.items)
    for agent in instance.agents:
        for cat in instance.categories_capacities:
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == cat]
            assert len(items_in_cat) <= instance.categories_capacities[cat]

    assert (is_EF11(alloc.instance, alloc.bundles) is None or
            is_EF1(alloc) is None)
    assert is_pareto_optimal(instance, allocation)



def make_instance_and_allocation(
    agent_vals,           # e.g. {"A": {"x": -5, "y": 10}, "B": {"x": 3, "y": 0}}
    allocation,           # e.g. {"A": ["x"], "B": ["y"]}
    item_categories=None  # e.g. {"x": "math", "y": "math"}
):
    inst = Instance(valuations=agent_vals, item_categories=item_categories)
    builder = AllocationBuilder(inst)
    builder.give_bundles(allocation)
    return inst, builder


def test_ef11_satisfied_simple_case():
    agent_vals = {
        "A": {"x": -5, "y": 10},
        "B": {"x": 3, "y": 1}
    }
    alloc = {
        "A": ["x"],
        "B": ["y"]
    }
    cat = {"x": "math", "y": "math"}

    inst, builder = make_instance_and_allocation(agent_vals, alloc, cat)
    result = is_EF11(inst, builder.bundles)
    assert result is None


def test_ef11_violation_found():
    agent_vals = {
        "A": {"x": -5, "y": 2, "z": -1},   # A has x (bad), B has y (good), both math
        "B": {"x": 1, "y": 6, "z": -1}
    }
    alloc = {
        "A": ["x", "z"],  # A has x (bad) and z (bad)
        "B": ["y"]
    }
    cat = {"x": "math", "y": "math", "z": "math"}

    inst, builder = make_instance_and_allocation(agent_vals, alloc, cat)
    result = is_EF11(inst, builder.bundles)
    assert result == ("A", "B")


def test_large_two_agent_ef11_violation():
    agent_vals = {
        "A": {"c1": -8, "c2": -2, "c3": 1,  "c4": 0,  "c5": 2,  "c6": 0},
        "B": {"c1": -1, "c2":  0, "c3": 6,  "c4": 7,  "c5": 1,  "c6": 3},
    }
    alloc = {
        "A": ["c1", "c3", "c4"],      # Value for A: -8 + 1 + 0 = -7
        "B": ["c2", "c5", "c6"]       # Value for A: 0 + 2 + 0 = 2 → envy of 9
    }
    cat = {
        "c1": "math",
        "c2": "math",
        "c3": "science",
        "c4": "science",
        "c5": "science",
        "c6": "science"
    }

    inst, builder = make_instance_and_allocation(agent_vals, alloc, cat)
    result = is_EF11(inst, builder.bundles)
    assert result == ("A", "B")

def test_multiple_items_per_agent():
    agent_vals = {
        "A": {"x": -5, "y": 10, "z": 2},
        "B": {"x": 3, "y": 1, "z": 6}
    }
    alloc = {
        "A": ["x", "z"],
        "B": ["y"]
    }
    cat = {"x": "math", "y": "math", "z": "math"}

    inst, builder = make_instance_and_allocation(agent_vals, alloc, cat)
    result = is_EF11(inst, builder.bundles)
    assert result is None  # A can drop x (chore), gain y (good) in same category


def test_ef1_holds_with_chore_and_good():
    agent_vals = {
        "A": {"x": -1, "y": -1},
        "B": {"x": -1, "y": 2}
    }
    alloc = {
        "A": ["x"],     # value for A: -4 + 10 = 6
        "B": ["y"]           # value for A: 2
    }

    inst, builder = make_instance_and_allocation(agent_vals, alloc)
    result = is_EF1(builder)
    assert result is None


def test_ef1_violation_detected():
    agent_vals = {
        "A": {"x": -10, "y": 2, "z": 0},
        "B": {"x": -1, "y": 4,   "z": 6}
    }
    alloc = {
        "A": ["x", "z"],
        "B": ["y"]
    }

    inst, builder = make_instance_and_allocation(agent_vals, alloc)
    result = is_EF1(builder)
    assert result == ("A", "B")

def test_two_categories_equal_entitlement():
    instance = Instance(
        valuations={
            "A": {"x1": 10, "x2": 5, "y1": 7, "y2": 1},
            "B": {"x1": 1, "x2": 5, "y1": 9, "y2": 10}
        },
        item_capacities={"x1": 1, "x2": 1, "y1": 1, "y2": 1},
        item_categories={"x1": "X", "x2": "X", "y1": "Y", "y2": "Y"},
        category_capacities={"X": 1, "Y": 1},
    )
    alloc = AllocationBuilder(instance)
    category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    all_items = set(alloc.bundles["A"]) | set(alloc.bundles["B"])
    assert all_items == {"x1", "x2", "y1", "y2"}


def test_category_capacity_per_agent():
    instance = Instance(
        valuations={
            "A": {"x1": 10, "x2": 8, "y1": 7, "y2": 2},
            "B": {"x1": 2, "x2": 4, "y1": 9, "y2": 10}
        },
        agent_capacities={"A": 2, "B": 2},
        item_capacities={"x1": 1, "x2": 1, "y1": 1, "y2": 1},
        item_categories={"x1": "X", "x2": "X", "y1": "Y", "y2": "Y"},
        category_capacities={"X": 1, "Y": 1}
    )
    alloc = AllocationBuilder(instance)
    category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    for agent in instance.agents:
        for category in instance.categories[0]:
            items_in_cat = [i for i in alloc.bundles[agent] if instance.item_categories[i] == category]
            assert len(items_in_cat) <= 1


def test_conflict_blocks_agent():
    instance = Instance(
        valuations={
            "A": {"x1": 9, "y1": 1},
            "B": {"x1": 1, "y1": 10}
        },
        agent_capacities={"A": 2, "B": 2},
        item_capacities={"x1": 1, "y1": 1},
        item_categories={"x1": "X", "y1": "Y"},
        category_capacities={"X": 1, "Y": 1},
        agent_conflicts={"A": {"y1"}}
    )
    alloc = AllocationBuilder(instance)
    category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    assert "y1" not in alloc.bundles["A"]


def test_weighted_entitlement():
    instance = Instance(
        valuations={
            "A": {"x1": 20, "y1": 5},
            "B": {"x1": 10, "y1": 20}
        },
        agent_capacities={"A": 2, "B": 2},
        item_capacities={"x1": 1, "y1": 1},
        item_categories={"x1": "X", "y1": "Y"},
        category_capacities={"X": 1, "Y": 1}
    )
    alloc = AllocationBuilder(instance)
    category_w_max_two_agents(alloc, weights=[0.7, 0.3])
    assert "x1" in alloc.bundles["A"]
    assert "y1" in alloc.bundles["B"]


def test_tie_case():
    instance = Instance(
        valuations={
            "A": {"x": 10},
            "B": {"x": 10}
        },
        agent_capacities={"A": 1, "B": 1},
        item_capacities={"x": 1},
        item_categories={"x": "Z"},
        category_capacities={"Z": 1}
    )
    alloc = AllocationBuilder(instance)
    category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    total = sum("x" in bundle for bundle in alloc.bundles.values())
    assert total == 1  # Only one agent gets the item