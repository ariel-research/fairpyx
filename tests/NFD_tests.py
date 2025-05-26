import pytest

from fairpyx import Instance, divide, AllocationBuilder
from fairpyx.algorithms.NFD import Nearly_Fair_Division, is_EF11, is_EF1
import random


def generate_random_instance(seed=42):
    random.seed(seed)
    agents = ["A", "B"]
    num_items = 50
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

def is_pareto_optimal(allocation, instance):
    """
    Checks whether the allocation is Pareto optimal.
    It ensures that no exchange of one item between agents strictly improves both.
    """
    agents = list(allocation.keys())
    a1, a2 = agents
    u = {a: sum(instance.valuations[a][i] for i in allocation[a]) for a in agents}
    for i in allocation[a1]:
        for j in allocation[a2]:
            if instance.item_categories[i] == instance.item_categories[j]:
                new_u1 = u[a1] - instance.valuations[a1][i] + instance.valuations[a1][j]
                new_u2 = u[a2] - instance.valuations[a2][j] + instance.valuations[a2][i]
                if new_u1 > u[a1] and new_u2 > u[a2]:
                    return False
    return True







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
    Tests a large instance with 100 items and 10 categories.
    Ensures all items are allocated and each agent respects category constraints, EF1\EF11, and PO.
    """
    agents = ["A", "B"]
    items = {f"i{k}": k for k in range(100)}
    instance = Instance(
        valuations={a: items.copy() for a in agents},
        agent_capacities={"A": 100, "B": 100},
        item_capacities={f"i{k}": 1 for k in range(100)},
        item_categories={f"i{k}": f"cat{k%10}" for k in range(100)},
        category_capacities={f"cat{k}": 10 for k in range(10)},
    )
    allocation = divide(Nearly_Fair_Division, instance)
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == 100
    for agent in agents:
        for cat in range(10):
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == f"cat{cat}"]
            assert len(items_in_cat) <= 10

    assert is_EF11(instance, allocation) or is_EF1(instance, allocation)
    assert is_pareto_optimal(instance, allocation)


# ---------------------------
# Random Inputs
# ---------------------------
def test_random_instance():
    instance = generate_random_instance()
    print("Random Instance:" , instance)
    allocation = divide(Nearly_Fair_Division, instance)
    print("Allocation:", allocation)
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == len(instance.items)
    for agent in instance.agents:
        for cat in instance.categories_capacities:
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == cat]
            assert len(items_in_cat) <= instance.categories_capacities[cat]

    assert is_EF11(allocation) or is_EF1(allocation)
    assert is_pareto_optimal(instance, allocation)