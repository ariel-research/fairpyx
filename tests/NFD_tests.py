import pytest
from fairpyx import Instance, divide
from fairpyx.algorithms.NFD import Nearly_Fair_Division
import random
from typing import Optional, Tuple, Any, Dict, List
from itertools import combinations


##############################################################################
#  EF[1,1]  – goods + chores with categories
##############################################################################

def is_EF11(instance: "Instance",
            allocation: dict[Any, list[Any]],
            abs_tol: float = 1e-9
           ) -> Optional[Tuple[Any, Any]]:
    """
    Return None  ⇢ allocation is EF[1,1]
    Return (i,j) ⇢ agent *i* still envies *j* even after every
                  same-category good+chore trade.

    Parameters
    ----------
    instance   : the problem Instance (needed for utilities & categories)
    allocation : mapping agent → iterable of items
    abs_tol    : slack for floating-point arithmetic
    """
    # cache own bundle values
    own_val = {a: instance.agent_bundle_value(a, allocation[a]) for a in instance.agents}

    for i in instance.agents:
        Ai, v_iAi = allocation[i], own_val[i]

        for j in instance.agents:
            if i == j:
                continue
            Aj       = allocation[j]
            v_iAj    = instance.agent_bundle_value(i, Aj)

            # ① already no envy
            if v_iAi + abs_tol >= v_iAj:
                continue

            # ② search for a good-chore pair in the SAME category
            envy_gone = False
            for g in Aj:                         # candidate good from j
                val_g = instance.agent_item_value(i, g)
                if val_g <= 0:
                    continue
                cat = instance.item_categories[g] if instance.item_categories else None

                for h in Ai:                     # candidate chore from i
                    val_h = instance.agent_item_value(i, h)
                    if val_h >= 0:
                        continue
                    if instance.item_categories and instance.item_categories[h] != cat:
                        continue

                    # EF[1,1] inequality after removing g and h
                    if v_iAi - val_h + abs_tol >= v_iAj - val_g:
                        envy_gone = True
                        break
                if envy_gone:
                    break

            if not envy_gone:        # i still envies j
                return (i, j)

    return None

##############################################################################
#  EF1  – envy-free up to one item
##############################################################################
def is_EF1(instance: "Instance",
           allocation: dict[Any, list[Any]],
           abs_tol: float = 1e-9
          ) -> Optional[Tuple[Any, Any]]:
    """
    Return None  ⇢ allocation is EF1
    Return (i,j) ⇢ agent *i* still envies *j* even after removing one item.
    """
    own_val = {a: instance.agent_bundle_value(a, allocation[a]) for a in instance.agents}

    for i in instance.agents:
        Ai, v_iAi = allocation[i], own_val[i]

        for j in instance.agents:
            if i == j:
                continue
            Aj    = allocation[j]
            v_iAj = instance.agent_bundle_value(i, Aj)

            # no envy
            if v_iAi + abs_tol >= v_iAj:
                continue

            envy_gone = False

            # drop one item from j
            for x in Aj:
                if v_iAi + abs_tol >= v_iAj - instance.agent_item_value(i, x):
                    envy_gone = True
                    break

            # ③ or drop one item from i
            if not envy_gone:
                for y in Ai:
                    if v_iAi - instance.agent_item_value(i, y) + abs_tol >= v_iAj:
                        envy_gone = True
                        break

            if not envy_gone:
                return (i, j)

    return None



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
    total_allocated = sum(len(v) for v in allocation.values())
    assert total_allocated == 200
    for agent in agents:
        for cat in range(10):
            items_in_cat = [i for i in allocation[agent] if instance.item_categories[i] == f"cat{cat}"]
            assert len(items_in_cat) <= 10

    assert (is_EF11(instance, allocation) is None or
            is_EF1(instance, allocation) is None)
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

    assert (is_EF11(instance, allocation) is None or
            is_EF1(instance, allocation) is None)
    assert is_pareto_optimal(instance, allocation)