import pytest
from fairpyx import Instance, divide
from NFD import fair_capacity_algorithm
import random

from fairpyx.utils.test_heterogeneous_matroid_constraints_algorithms_utils import is_fef1


def generate_random_instance(seed=42):
    random.seed(seed)
    agents = ["A", "B"]
    num_items = 200
    categories = [f"cat{i}" for i in range(10)]
    items = [f"o{i}" for i in range(num_items)]
    valuations = {
        a: {i: random.randint(-10, 10) for i in items} for a in agents
    }
    capacities = {i: 1 for i in items}
    item_cats = {i: random.choice(categories) for i in items}
    cat_caps = {cat: num_items // len(agents) // len(categories) for cat in categories}
    return Instance(
        valuations=valuations,
        agent_capacities={a: num_items // 2 for a in agents},
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


def is_EF1(allocation, instance):
    """
    Verifies whether the allocation satisfies EF1 (Envy-Free up to one item) in a same-sign instance.

    EF1 holds if, for every pair of agents i and j:
    - Agent i either does not envy agent j (values their bundle equally or more), OR
    - Envy can be eliminated by removing a single item from either:
        • Agent j's bundle (i envies less if j loses one item), or
        • Agent i's bundle (i feels better if they give up one item).
    """
    for i, j in [("A", "B"), ("B", "A")]:
        # Compute how much agent i values their own and j's allocations
        ui_self = sum(instance.valuations[i][o] for o in allocation[i])
        ui_other = sum(instance.valuations[i][o] for o in allocation[j])

        # No envy — EF1 is trivially satisfied
        if ui_self >= ui_other:
            continue

        # Try removing one item from j's bundle to eliminate envy
        for o in allocation[j]:
            if ui_self >= ui_other - instance.valuations[i][o]:
                break  # Envy disappears after removing o from j
        else:
            # Try removing one item from i's own bundle to eliminate envy
            for o in allocation[i]:
                if ui_self - instance.valuations[i][o] >= ui_other:
                    break  # Envy disappears after removing o from i
            else:
                # No item removal resolves the envy → not EF1
                return False

    # EF1 holds across all agent pairs
    return True


def is_EF11(allocation, instance):
    """
    Verifies whether the allocation satisfies EF[1,1] (Envy-Free up to one good and one chore)
    for a general mixed setting (goods + chores), with categorized items.

    EF[1,1] holds if, for every pair of agents i and j:
    - Agent i either does not envy agent j, OR
    - There exists:
        • A good in j's bundle (valued positively by i), and
        • A chore in i's bundle (valued negatively by i),
      such that:
        • Both items are in the SAME category, and
        • Removing the good from j and the chore from i eliminates envy:
            u_i(A_i \ {chore}) ≥ u_i(A_j \ {good})
    """
    for i, j in [("A", "B"), ("B", "A")]:
        # Agent i's valuation of their own and j's bundle
        ui_self = sum(instance.valuations[i][o] for o in allocation[i])
        ui_other = sum(instance.valuations[i][o] for o in allocation[j])

        # No envy — EF[1,1] trivially satisfied
        if ui_self >= ui_other:
            continue

        # Search for a good in j's bundle and a chore in i's, in the same category
        envy_eliminated = False
        for good in allocation[j]:
            if instance.valuations[i][good] <= 0:
                continue  # Skip if not a good for i
            for chore in allocation[i]:
                if instance.valuations[i][chore] >= 0:
                    continue  # Skip if not a chore for i
                if instance.item_categories[good] != instance.item_categories[chore]:
                    continue  # Must be in the same category

                # Check if envy is eliminated after removing chore and good
                new_ui_self = ui_self - instance.valuations[i][chore]
                new_ui_other = ui_other - instance.valuations[i][good]
                if new_ui_self >= new_ui_other:
                    envy_eliminated = True
                    break
            if envy_eliminated:
                break

        # If no valid good-chore pair eliminated envy, fail
        if not envy_eliminated:
            return False

    # EF[1,1] holds across all agent pairs
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
    allocation = divide(fair_capacity_algorithm, instance)
    assert sum(len(v) for v in allocation.values()) == 1
    assert any("x" in bundle for bundle in allocation.values())

def test_empty_instance():
    """
    Tests an instance with no items or valuations.
    Ensures allocation returns empty bundles for both agents.
    """
    instance = Instance(
        valuations={"A": {}, "B": {}},
        agent_capacities={"A": 0, "B": 0},
        item_capacities={},
        item_categories={},
        category_capacities={},
    )
    allocation = divide(fair_capacity_algorithm, instance)
    assert allocation == {"A": [], "B": []}

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
        agent_capacities={"A": 50, "B": 50},
        item_capacities={f"i{k}": 1 for k in range(100)},
        item_categories={f"i{k}": f"cat{k%10}" for k in range(100)},
        category_capacities={f"cat{k}": 10 for k in range(10)},
    )
    allocation = divide(fair_capacity_algorithm, instance)
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
    allocation = divide(fair_capacity_algorithm, instance)
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == len(instance.item_capacities)
    for agent in instance.valuations:
        for cat in instance.category_capacities:
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == cat]
            assert len(items_in_cat) <= instance.category_capacities[cat]

    assert is_EF11(instance, allocation) or is_EF1(instance, allocation)
    assert is_pareto_optimal(instance, allocation)