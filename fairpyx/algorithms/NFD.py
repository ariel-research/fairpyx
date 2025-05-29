"""
An implementation of the algorithms in:

"Efficient Nearly-Fair Division with Capacity Constraints", by A. Hila Shoshan,  Noam Hazon,  Erel Segal-Halevi
(2023), https://arxiv.org/abs/2205.07779

Programmer: Matan Ziv.
Date: 2025-04-27.
"""

# The end-users of the algorithm feed the input into an "Instance" variable, which tracks the original input (agents, items and their capacities).
# But the algorithm implementation uses an "AllocationBuilder" variable, which tracks both the ongoing allocation and the remaining input (the remaining capacities of agents and items).
# The function `divide` is an adaptor - it converts an Instance to an AllocationBuilder with an empty allocation.
from fairpyx import Instance, AllocationBuilder, divide
from typing import List, Tuple, Optional
from typing import Mapping, Sequence, List, Any



# The end-users of the algorithm feed the input into an "Instance" variable, which tracks the original input (agents, items and their capacities).
# The input of the algorithm is:
# a dict of a dict witch tells the valuations of the items for each player
# a dict of a sets, dict of category each category is a set contains the items in the category
# a dict of categories and the capacities



# The `logging` facility is used both for debugging and for illustrating the steps of the algorithm.
# It can be used to automatically generate running examples or explanations.
import logging



logger = logging.getLogger(__name__)

item_categories = {
    "o1": "cat1", "o2": "cat1", "o3": "cat1", "o4": "cat1",  # 4 items in category 1
    "o5": "cat2", "o6": "cat2"  # 2 items in category 2
}

# ---------------------------
# Example instance with categories and capacities:
# ---------------------------
example_instance = Instance(
    valuations={
        "Agent1": {"o1": 0, "o2": -1, "o3": -4, "o4": -5, "o5": 0, "o6": 2},
        "Agent2": {"o1": 0, "o2": -1, "o3": -2, "o4": -1, "o5": -1, "o6": 0},
    },
    # agent_capacities={"Agent1": 3, "Agent2": 3},
    item_categories={
        "o1": "cat1", "o2": "cat1", "o3": "cat1", "o4": "cat1",  # 4 items in category 1
        "o5": "cat2", "o6": "cat2"                               # 2 items in category 2
    },
    item_capacities={item: 1 for item in item_categories.keys()},
    # item_capacities={"o1": 1, "o2": 1, "o3": 1, "o4": 1, "o5": 1, "o6": 1},
    category_capacities={"cat1": 2, "cat2": 1},  # each agent can get at most 2 from cat1, 1 from cat2
)


def Nearly_Fair_Division(alloc: AllocationBuilder) -> None:
    """
    Implements the EF[1,1] + PO allocation algorithm from:
    "Efficient Nearly-Fair Division with Capacity Constraints" (AAMAS 2023).

    Parameters:
        alloc (AllocationBuilder): allocation builder with capacity-tracked state

    Returns:


    Examples and tests:
    >>> divide(Nearly_Fair_Division, instance=example_instance)
    {'Agent1': ['o1', 'o2', 'o5'], 'Agent2': ['o3', 'o4', 'o6']}

    >>> instance2 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1},
    ...         "B": {"o1": -1},
    ...     },
    ...     item_capacities={"o1": 1},
    ...     item_categories={"o1": "g"},
    ...     category_capacities={"g": 1},
    ... )
    >>> divide(Nearly_Fair_Division, instance=instance2)
    {'A': ['o1'], 'B': []}


    >>> instance3 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1, "o2": -1},
    ...         "B": {"o1": 1, "o2": -1},
    ...     },
    ...     item_capacities={"o1": 1, "o2": 1},
    ...     item_categories={"o1": "g", "o2": "g"},
    ...     category_capacities={"g": 1},
    ... )
    >>> divide(Nearly_Fair_Division, instance=instance3)
    {'A': ['o1'], 'B': ['o2']}

    >>> instance4 = Instance(
    ...     valuations={
    ...         "A": {"o1": 1, "o2": -1, "o3": 1, "o4": 1.5},
    ...         "B": {"o1": 1, "o2": -1, "o3": 1, "o4": 1},
    ...     },
    ...     item_capacities={"o1": 1, "o2": 1, "o3": 1, "o4": 1},
    ...     item_categories={"o1": "g", "o2": "g", "o3": "c", "o4": "c"},
    ...     category_capacities={"g": 1, "c": 2},
    ... )
    >>> divide(Nearly_Fair_Division, instance=instance4)
    {'A': ['o2', 'o3', 'o4'], 'B': ['o1']}

    """
    # 1. Find W-maxallocation
    # 2. Is EF[1,1] --> if so return
    # 3. Is Agent 2 envy? --> if no switch names (pointers)
    # 4. Build a set of item-pairs whose replacement increases agent 2’s utility and sort
    # 5. Switch items in order until an EF[1,1] allocation is found

    logger.info("valuations = %s", alloc.instance)
    logger.info("remaining_item_capacities = %s", alloc.remaining_item_capacities)

    # ---------- 0) Setup ----------
    inst = alloc.instance
    agents: List[str] = list(inst.agents)
    if len(agents) != 2:
        raise NotImplementedError("Current implementation supports exactly two agents.")

    # ---------- 1) W-max greedy assignment ----------
    # ----------------------------------------------------------------------
    #  Category–wise W-max
    # ----------------------------------------------------------------------
    logger.info("remaining_item_capacities before w-max = %s", alloc.remaining_item_capacities)
    w_max_two_agents(alloc, weights=[0.5, 0.5])
    logger.info("remaining_item_capacities after w-max = %s", alloc.remaining_item_capacities)

    # ---------- 2) already fair? ----------
    status = is_EF11(alloc)
    if status is not None:
        logger.info(f"W-max allocation is not EF[1,1], starting envy-elimination loop.{status}")

    # ---------- 3) envy-elimination loop ----------
        def candidates_r_ratio(instance, bundles, envier, envied):
            """
            Find all same-category (o_envier, o_envied) pairs whose swap would improve
            *envier*’s utility, compute their r-ratio, and return them in **descending**
            r order.

            Parameters
            ----------
            instance : Instance
                The original problem instance (gives utilities & categories).
            bundles : dict[str, set[str] | list[str]]
                Current allocation, e.g. alloc.bundles.
            envier : str
                Agent who currently envies.
            envied : str
                The agent being envied.

            Returns
            -------
            list[tuple[float, str, str]]
                List of triples ``(r, o_envier, o_envied)`` sorted from best to worst.
                Empty list ⇒ no beneficial swap exists.

            Notes
            -----
            * We set inf to pairs whose denominator is zero (identical utilities) or whose
              swap would *lower* the envier’s utility.
            * Works with either lists or sets inside *bundles*.

            """
            # Turn ∪/list into a list to iterate cheaply
            envier_items = list(bundles[envier])
            envied_items = list(bundles[envied])

            pairs = []
            for o_i in envier_items:  # item owned by envier
                cat = instance.item_categories[o_i]
                for o_j in envied_items:  # item owned by envied
                    if instance.item_categories[o_j] != cat:
                        continue  # must share category
                    # Would the swap help the envier?
                    delta_envier = (
                            instance.agent_item_value(envier, o_j)
                            - instance.agent_item_value(envier, o_i)
                    )
                    if delta_envier <= 0:
                        continue  # no improvement

                    # r = (u_envied(o_i) − u_envied(o_j)) / (u_envier(o_i) − u_envier(o_j))
                    numer = instance.agent_item_value(envier, o_i) - instance.agent_item_value(envier, o_j)
                    denom = instance.agent_item_value(envied, o_i) - instance.agent_item_value(envied, o_j)
                    if numer == 0:
                        r = 0
                    elif denom == 0:  # avoid /0
                        if numer <= 0:
                            r = float('inf')
                        else:
                            r = -float('inf')
                    else:
                        r = numer / denom

                    pairs.append((r, o_i, o_j))

            # Best first
            pairs.sort(key=lambda t: t[0], reverse=True)
            return pairs

        while True:
            status = is_EF11(alloc)
            if is_EF1(alloc) is None or status is None:
                break

            envier, envied = status
            # Candidate items that envier can still accept
            candidates = candidates_r_ratio(inst, alloc.bundles, envier, envied)
            if candidates:
                r_best, o_envier, o_envied = candidates[0]
                # preform the swap
                logger.info("Swapping items %s (envier) and %s (envied) with r = %.2f",
                            o_envier, o_envied, r_best)
                logger.info("Before swap: %s", alloc.bundles)
                logger.info("instance.item_capacity(o_envier) = %d, instance.item_capacity(o_envied) = %d",
                            alloc.instance.item_capacity(o_envier), alloc.instance.item_capacity(o_envied))
                logger.info("remaining_item_capacities = %s",
                            alloc.remaining_item_capacities)
                alloc.remove_item(envier, o_envier)
                alloc.remove_item(envied, o_envied)
                alloc.give(envier, o_envied, logger=None)
                alloc.give(envied, o_envier, logger=None)


# ----------------------------------------------------------------------
#  W-maximisation for two agents
# ----------------------------------------------------------------------
def w_max_two_agents(alloc: "AllocationBuilder",
                     weights: Mapping[str, float] | Sequence[float]
                     ) -> None:
    """
    Fill ``alloc`` with a bundle profile that *maximises* the weighted
    sum of utilities  Σᵢ wᵢ·uᵢ  subject to **per-category capacities**.

    Parameters
    ----------
    alloc : AllocationBuilder
        Fresh builder whose bundles are still empty.  The function
        mutates it *in place* using ``alloc.give`` – exactly the same
        way the rest of *fairpyx* algorithms do.
    weights : dict | list | tuple
        Two strictly-positive weights, one per agent, in the order
        returned by ``list(alloc.instance.agents)``  **or** a mapping
        {agent → weight}.  Only the *ratio* matters; the sum need not
        be 1.

    Restrictions
    ------------
    * Exactly **two agents** in the instance.
    * Each item’s capacity equals 1 (the paper and our code add dummy
      items when a capacity > 1 is required, so this is WLOG).
    * All agents share the same category capacities
      ``instance.categories_capacities`` — matching the setting of the
      AAMAS-23 algorithm.

    Logic
    -----
    For every category *c* with capacity *s_c* per agent:

        • Compute Δ(o) := w₁·u₁(o) − w₂·u₂(o)  for each item o∈c.
        • Sort the items in *descending* Δ order.
        • Give the first s_c items to agent 1, the remainder to agent 2.

    Proof sketch
    ------------
    The Δ-sorting is a classic greedy proof of the *rearrangement
    inequality*.  Because each agent must take exactly s_c items from
    category *c*, the rule above yields the unique permutation of
    bundles that maximises  w₁·u₁ + w₂·u₂  inside that category; summing
    over independent categories preserves optimality globally.

    Example
    -------
    >>> inst = example_instance
    >>> builder = AllocationBuilder(inst)
    >>> w_max_two_agents(builder, [0.5, 0.5])
    >>> builder.sorted()
    {'Agent1': ['o1', 'o2', 'o6'], 'Agent2': ['o3', 'o4', 'o5']}

    >>> builder = AllocationBuilder(inst)
    >>> w_max_two_agents(builder, [1.0, 0.1])  # Agent 1 gets everything
    >>> builder.sorted()
    {'Agent1': ['o1', 'o2', 'o6'], 'Agent2': ['o3', 'o4', 'o5']}

    >>> builder = AllocationBuilder(inst)
    >>> w_max_two_agents(builder, [0.1, 1.0])  # Agent 2 gets everything
    >>> builder.sorted()
    {'Agent1': ['o2', 'o3', 'o5'], 'Agent2': ['o1', 'o4', 'o6']}

    >>> builder = AllocationBuilder(inst)
    >>> w_max_two_agents(builder, {'Agent1': 2.0, 'Agent2': 1.0})  # bias toward Agent1
    >>> sorted(len(v) for v in builder.bundles.values())
    [3, 3]

    >>> builder = AllocationBuilder(inst)
    >>> w_max_two_agents(builder, [-1, 1])
    Traceback (most recent call last):
        ...
    ValueError: weights must be strictly positive
    """
    inst = alloc.instance
    agents: List[str] = list(inst.agents)
    if len(agents) != 2:
        raise ValueError("w_max_two_agents currently supports *exactly* two agents.")
    a1, a2 = agents # fixed order

    # Normalise the weight container → dict {agent: weight}
    if not isinstance(weights, dict):
        w1, w2 = weights
        weights = {a1: float(w1), a2: float(w2)}
    w1, w2 = weights[a1], weights[a2]
    if w1 <= 0 or w2 <= 0:
        raise ValueError("weights must be strictly positive")

    # If the instance has *no* categories, pretend all items share one.
    if inst.categories_items is None:
        all_items = list(inst.items)
        inst.categories_items = {"_all": all_items}
        inst.categories_capacities = {"_all": min(len(all_items)//2, len(all_items))}

    # One independent W-max step per category
    for cat, items in inst.categories_items.items():
        s_c = inst.categories_capacities[cat]          # per-agent quota
        if s_c == 0 or not items:
            continue

        # Sort by Δ(o) = w1·u1(o) − w2·u2(o)
        deltas: dict[str, float] = {
            o: w1 * inst.agent_item_value(a1, o) - w2 * inst.agent_item_value(a2, o)
            for o in items
        }
        deltas_absolute = [(o, abs(delta)) for o, delta in deltas.items()]
        deltas_absolute.sort(key=lambda pair: pair[1], reverse=True)

        # Split the most positive to a1 and the most negative to a2
        for o, delta in deltas_absolute:
            if (deltas[o] >= 0 or alloc.remaining_agent_capacities[a2] == 0) and alloc.agents_category_capacities[a1][inst.item_categories[o]] > 0:
                # Give to agent 1 if positive or agent 2 has no capacity left
                alloc.give(a1, o)
            elif alloc.agents_category_capacities[a2][inst.item_categories[o]] == 0:
                # No capacity left for agent 2, so give to agent 1
                alloc.give(a1, o)
            elif alloc.agents_category_capacities[a1][inst.item_categories[o]] == 0 and alloc.agents_category_capacities[a2][inst.item_categories[o]] == 0:
                logger.warning("Both agents have no capacity left, cannot allocate item %s", o)
            else:
                # negative and agent 2 has capacity left
                alloc.give(a2, o)

def is_EF11(allocation: "AllocationBuilder",
            abs_tol: float = 1e-9
            ) -> Optional[Tuple[any, any]]:
        r"""
        Check whether the current allocation satisfies EF[1,1].

        Parameters
        ----------
        allocation : AllocationBuilder
            The incremental allocation object whose `.bundles` and
            `.instance` fields we inspect.
        abs_tol : float, optional
            A tiny slack to absorb floating-point noise.

        Returns
        -------
        None                     –– if every pair (i,j) meets EF[1,1].
        (i, j) : Tuple[agent, agent]
            The first counter-example found:
            • `i` is the envious agent,
            • `j` is the agent she envies **after** all EF[1,1] patches fail.

        Notes
        -----
        EF[1,1] says that for every pair of agents *i, j*,
            uᵢ(Aᵢ) ≥ uᵢ(Aⱼ)                                         (no-envy),  OR
            ∃ (good g ∈ Aⱼ, chore h ∈ Aᵢ) in the *same* category s.t.
            uᵢ(Aᵢ \ {h}) ≥ uᵢ(Aⱼ \ {g}).

        • A **good** is an item with uᵢ(item) > 0 for agent *i*;
          a **chore** has uᵢ(item) < 0.
        • When `instance.item_categories is None` we treat all
          items as belonging to one universal category.
        """
        inst = allocation.instance
        bundles = allocation.bundles  # maps agent → set/list of items :contentReference[oaicite:0]{index=0}

        # Pre-compute each agent's value for her own bundle once.
        bundle_value = {
            agent: inst.agent_bundle_value(agent, bundles[agent])
            for agent in inst.agents
        }

        # Iterate over ordered pairs (i, j).
        for i in inst.agents:
            for j in inst.agents:
                if i == j:
                    continue

                # i's utility for j's bundle
                u_i_Aj = inst.agent_bundle_value(i, bundles[j])
                # i's utility for its own bundle
                u_i_Ai = bundle_value[i]

                # If i already does not envy j, move on.
                if u_i_Ai + abs_tol >= u_i_Aj:
                    continue

                # Else: look for a good–chore pair in the same category
                envy_eliminated = False
                for g in bundles[j]:
                    val_g = inst.agent_item_value(i, g)
                    if val_g <= 0:  # not a good for i
                        continue
                    cat_g = (inst.item_categories[g]
                             if inst.item_categories is not None else None)

                    for h in bundles[i]:
                        val_h = inst.agent_item_value(i, h)
                        if val_h >= 0:  # not a chore for i
                            continue
                        # same category? (trivial if categories are disabled)
                        if (inst.item_categories is not None
                                and inst.item_categories[h] != cat_g):
                            continue

                        # Check the EF[1,1] inequality:
                        if (u_i_Ai - val_h + abs_tol) >= (u_i_Aj - val_g):
                            envy_eliminated = True
                            break
                    if envy_eliminated:
                        break

                # No patch found ⇒ true envy remains.
                if not envy_eliminated:
                    return (i, j)

        # All pairs passed
        return None

def is_EF1(allocation: "AllocationBuilder",
           abs_tol: float = 1e-9
          ) -> Optional[Tuple[Any, Any]]:
    """
    Determine whether the current allocation is EF1 (Envy-Free up to one
    item) for *mixed* goods/chores.

    EF1 holds if, for every ordered pair of agents (i, j):

        u_i(A_i) ≥ u_i(A_j)                                       (no envy), OR
        ∃ x ∈ A_j  such that u_i(A_i) ≥ u_i(A_j \\ {x}), OR
        ∃ y ∈ A_i  such that u_i(A_i \\ {y}) ≥ u_i(A_j).

    Parameters
    ----------
    allocation : AllocationBuilder
        Object whose `.bundles` field stores the current bundles and whose
        `.instance` field points at the underlying `Instance` object.
    abs_tol : float
        Numerical slack to absorb floating-point error (default 1e-9).

    Returns
    -------
    None
        If the allocation satisfies EF1.
    (i, j) : tuple
        The *first* pair found where agent *i* still envies agent *j* even
        after every single-item removal test.

    Notes
    -----
    * Works for any sign of item values (goods or chores); it never looks
      at categories.
    * Runs in O(n² · m) where n = #agents, m = max-bundle-size.
    """
    inst     = allocation.instance
    bundles  = allocation.bundles                       # :contentReference[oaicite:0]{index=0}
    value_of = inst.agent_bundle_value                  # :contentReference[oaicite:1]{index=1}

    # Cache each agent’s own-bundle value once
    own_val = {a: value_of(a, bundles[a]) for a in inst.agents}

    for i in inst.agents:
        A_i   = bundles[i]
        v_iAi = own_val[i]

        for j in inst.agents:
            if i == j:
                continue

            A_j   = bundles[j]
            v_iAj = value_of(i, A_j)

            # 1️⃣  Plain no-envy
            if v_iAi + abs_tol >= v_iAj:
                continue

            envy_gone = False

            # 2️⃣  Remove one item from j’s bundle
            for x in A_j:
                if v_iAi + abs_tol >= v_iAj - inst.agent_item_value(i, x):
                    envy_gone = True
                    break

            # 3️⃣  Otherwise, try removing one item from i’s own bundle
            if not envy_gone:
                for y in A_i:
                    if v_iAi - inst.agent_item_value(i, y) + abs_tol >= v_iAj:
                        envy_gone = True
                        break

            if not envy_gone:          # EF1 violated for (i, j)
                return (i, j)

    return None    # All pairs satisfied EF1


if __name__ == "__main__":
    import doctest
    import sys
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    print(doctest.testmod())


