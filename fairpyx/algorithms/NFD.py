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
from typing import Mapping, Sequence, List, Any, Dict, Iterable, Hashable, Tuple, Optional
import networkx as nx
from collections import defaultdict
from math import inf

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


def Nearly_Fair_Division(alloc: AllocationBuilder) -> Optional[AllocationBuilder]:
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
    {'A': ['o2'], 'B': ['o1']}

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

    logger.info("Starting NFD run with instance = %s", alloc.instance)

    # ---------- 0) Setup ----------
    inst = alloc.instance
    agents: List[str] = list(inst.agents)
    if len(agents) != 2:
        raise NotImplementedError("Current implementation supports exactly two agents.")

    # ----------------------------------------------------------------------
    #  Category–wise W-max
    # ----------------------------------------------------------------------
    logger.info("Allocating W-maximal allocation for two agents with weights [0.5, 0.5]")

    logger.debug("remaining_item_capacities before w-max = %s", alloc.remaining_item_capacities)
    category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    logger.debug("remaining_item_capacities after w-max = %s", alloc.remaining_item_capacities)
    logger.debug("Allocation after W-max: %s", alloc.bundles)


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
        status = is_EF11(alloc.instance, alloc.bundles)
        if is_EF1(alloc) is None or status is None:
            break

        logger.info(f"W-max allocation is not EF[1,1], starting envy-elimination iteration. {status}")
        envier, envied = status

        # Candidate items that envier can still accept
        logger.info("Finding candidates to swap for envier %s and envied %s", envier, envied)
        candidates = candidates_r_ratio(inst, alloc.bundles, envier, envied)
        if candidates:
            r_best, o_envier, o_envied = candidates[0]
            logger.info("Swapping items %s (envier) and %s (envied) with r = %.2f",
                        o_envier, o_envied, r_best)
            # preform the swap
            alloc.swap(envier, o_envier, envied, o_envied)

        else:
            logger.warning("No more candidates to swap, breaking the loop, the envy cant be fixed!!.")
            return None

    logger.info("We have a final NFD allocation: %s \n\n", alloc.bundles)
    return alloc

# ---------------------------------------------------------------------------
#  W-MAXIMAL ALLOCATION FOR TWO AGENTS
# ---------------------------------------------------------------------------
#
#  Paper reference: “Efficient Nearly-Fair Division with Capacity Constraints”
#  (AAMAS 2023), Def. 4.1 + Prop. 4.3.
#
#  Given   – an AllocationBuilder `alloc` whose state still contains all
#            unassigned items and whose capacity / conflict structures are
#            up-to-date;
#          – a list  `weights = [w_a, w_b]`, one non-negative weight per agent
#            that sums to 1 (the usual entitlements ½,½ can be passed as
#            `[0.5, 0.5]`).
#
#  Does    – builds the bipartite graph  G₍w₎  exactly as in the definition
#            (we replicate both agents and items to linearise capacities);
#          – calls NetworkX’ `max_weight_matching` with `maxcardinality=True`
#            so that we first fill as many edges as possible and then pick the
#            maximum-weight solution;
#          – translates the matching back into genuine allocations by invoking
#            `alloc.give(…)`, thereby *updating* the capacities that
#            `Nearly_Fair_Division` relies on afterwards.
#
#  Guarantees – because the graph is balanced (|V₁| = |V₂|) and we asked for
#               `maxcardinality=True`, the resulting allocation is a *perfect*
#               matching on every copy that could possibly be matched; hence
#               it is a w-maximal allocation in the sense of the paper.
#
# ---------------------------------------------------------------------------
def w_max_two_agents2(alloc: "AllocationBuilder", *, weights: List[float]) -> None:
    """
    Compute a w-maximal allocation on the still unassigned part of `alloc`
    for the *two* agents that are present in the instance.

    After this call `alloc` already contains the bundles chosen by the
    w-max algorithm; remaining capacities / conflicts are updated, so the
    rest of `Nearly_Fair_Division` can continue immediately.

    Parameters
    ----------
    alloc   : AllocationBuilder
        Current state of the allocation.
    weights : list[float] of length 2
        (w₁, w₂) with  w₁, w₂ ≥ 0  and  w₁ + w₂ = 1.
    """
    # ------------------------------------------------------------------
    # (0) Preparations & sanity checks
    # ------------------------------------------------------------------
    inst           = alloc.instance
    agents         = list(inst.agents)
    if len(agents) != 2:
        raise ValueError("w_max_two_agents currently supports exactly two agents.")
    a1, a2         = agents                          # deterministic order
    w1, w2         = weights
    if abs(w1 + w2 - 1) > 1e-9 or min(w1, w2) < 0:
        raise ValueError("weights must be non-negative and sum to 1.")

    # Helper lambdas – shorter to read below
    u = inst.agent_item_value
    item_cap   : Dict[str, int] = alloc.remaining_item_capacities
    agent_cap  : Dict[str, int] = alloc.remaining_agent_capacities
    cat_cap    : Dict[str, Dict[str, int]] | None = getattr(alloc, "agents_category_capacities", None)
    item_cat   = inst.item_categories or {}

    # ------------------------------------------------------------------
    # (1) Build the bipartite graph  G₍w₎  as in Definition 4.1 in the paper
    # ------------------------------------------------------------------
    G = nx.Graph()

    # (1a) COPY NODES FOR AGENTS
    #
    # One “unit” copy per *remaining* unit of agent capacity.
    # We keep a mapping  copy_id -> original_agent  for later.
    agent_copy_to_agent : Dict[str, str] = {}
    for ag in agents:
        for k in range(agent_cap.get(ag, 0)):
            node = f"AG[{ag}]#{k}"
            agent_copy_to_agent[node] = ag
            G.add_node(node, bipartite=0)

    # (1b) COPY NODES FOR ITEMS
    #
    # One copy for every still available unit of each item’s capacity.
    item_copy_to_item : Dict[str, str] = {}
    for itm, cap in item_cap.items():
        for k in range(cap):
            node = f"IT[{itm}]#{k}"
            item_copy_to_item[node] = itm
            G.add_node(node, bipartite=1)

    # (1c) EDGES + WEIGHTS
    #
    # Every agent copy connects to every item copy that is *feasible* for the
    # underlying agent (respecting agent/item conflicts and
    # category-capacities).
    #
    # Edge weight  =  w_i · u_i(o)
    FORBIDDEN = alloc.instance.item_capacity  # just something hashable

    def feasible(agent: str, item: str) -> bool:
        # Conflict checks: agent-item conflicts AND category quota (if used).
        if (agent, item) in alloc.remaining_conflicts:
            return False
        if cat_cap is not None:
            cat = item_cat[item]
            if cat_cap[agent][cat] <= 0:
                return False
        return True

    for a_copy, ag in agent_copy_to_agent.items():
        # Pull the w_i once so we don’t fetch it in every inner loop.
        w = w1 if ag == a1 else w2
        for i_copy, itm in item_copy_to_item.items():
            if feasible(ag, itm):
                val = u(ag, itm)
                weight = w * val
                G.add_edge(a_copy, i_copy, weight=weight)

    # ------------------------------------------------------------------
    # (2) Maximum-cardinality, maximum-weight matching on G
    # ------------------------------------------------------------------
    # maxcardinality=True  ==>  among all maximum-cardinality matchings
    #                         choose the one of max weight.
    matching : set[Tuple[str, str]] = nx.algorithms.matching.max_weight_matching(
        G, maxcardinality=True, weight="weight"
    )

    # ------------------------------------------------------------------
    # (3) Translate the matching back into real allocations
    # ------------------------------------------------------------------
    # Because copies are unique, each matched edge corresponds to exactly one
    # “real” item → agent transfer.
    for v1, v2 in matching:
        if v1 in agent_copy_to_agent:
            a_copy, i_copy = v1, v2
        else:
            a_copy, i_copy = v2, v1
        agent = agent_copy_to_agent[a_copy]
        item  = item_copy_to_item[i_copy]

        # The give() call decrements capacities and adds conflicts inside
        # AllocationBuilder, which is exactly what we want here.
        try:
            alloc.give(agent, item)
        except ValueError:
            # This should not happen, but guard against race conditions in
            # case external code manipulated alloc between building G and now.
            continue

    # Done – `alloc` now encodes a w-maximal allocation.


# ---------------------------------------------------------------------------
#  CATEGORY-WISE W-MAX FOR TWO AGENTS
# ---------------------------------------------------------------------------


def category_w_max_two_agents(alloc: "AllocationBuilder", *, weights: List[float]) -> None:
    """
    Perform Definition 4.1 *within each category separately*.

    After the call, `alloc` already contains every item that can be assigned
    by the w-max procedure while respecting:
        • remaining item capacities,
        • remaining agent capacities,
        • remaining per-agent category capacities.

    Parameters
    ----------
    alloc   : AllocationBuilder
        Current partial allocation (will be updated in-place).
    weights : list[float] of length 2
        Entitlement weights (must sum to 1).

    Examples
    --------

    >>> instance = Instance(
    ...     valuations = {
    ...         "A": {"x1": 10, "x2": 5, "y1": 7, "y2": 1},
    ...         "B": {"x1": 1,  "x2": 5, "y1": 9, "y2": 10}
    ...     },
    ...     item_capacities = {"x1": 1, "x2": 1, "y1": 1, "y2": 1},
    ...     item_categories = {"x1": "X", "x2": "X", "y1": "Y", "y2": "Y"},
    ...     category_capacities = {"X": 1, "Y": 1},
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    >>> sorted(list(alloc.bundles["A"]) + list(alloc.bundles["B"]))
    ['x1', 'x2', 'y1', 'y2']

    >>> instance = Instance(
    ...     valuations = {
    ...         "A": {"x1": 10, "x2": 8, "y1": 7, "y2": 2},
    ...         "B": {"x1": 2,  "x2": 4, "y1": 9, "y2": 10}
    ...     },
    ...     agent_capacities={"A": 2, "B": 2},
    ...     item_capacities = {"x1": 1, "x2": 1, "y1": 1, "y2": 1},
    ...     item_categories = {"x1": "X", "x2": "X", "y1": "Y", "y2": "Y"},
    ...     category_capacities = {"X": 1, "Y": 1}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    >>> all(len([i for i in alloc.bundles[a] if instance.item_categories[i] == c]) <= 1 for a in instance.agents for c in instance.categories[0])
    True

    >>> instance = Instance(
    ...     valuations = {
    ...         "A": {"x1": 9, "y1": 1},
    ...         "B": {"x1": 1, "y1": 10}
    ...     },
    ...     agent_capacities={"A": 2, "B": 2},
    ...     item_capacities = {"x1": 1, "y1": 1},
    ...     item_categories = {"x1": "X", "y1": "Y"},
    ...     category_capacities = {"X": 1, "Y": 1},
    ...     agent_conflicts = {"A": {"y1"}}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    >>> 'y1' in alloc.bundles["A"]
    False

    >>> instance = Instance(
    ...     valuations = {
    ...         "A": {"x1": 20, "y1": 5},
    ...         "B": {"x1": 10, "y1": 20}
    ...     },
    ...     agent_capacities={"A": 2, "B": 2},
    ...     item_capacities = {"x1": 1, "y1": 1},
    ...     item_categories = {"x1": "X", "y1": "Y"},
    ...     category_capacities = {"X": 1, "Y": 1}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> category_w_max_two_agents(alloc, weights=[0.7, 0.3])
    >>> sorted(alloc.sorted().items())
    [('A', ['x1']), ('B', ['y1'])]

    >>> instance = Instance(
    ...     valuations = {
    ...         "A": {"x": 10},
    ...         "B": {"x": 10}
    ...     },
    ...     agent_capacities = {"A": 1, "B": 1},
    ...     item_capacities = {"x": 1},
    ...     item_categories = {"x": "Z"},
    ...     category_capacities = {"Z": 1}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> category_w_max_two_agents(alloc, weights=[0.5, 0.5])
    >>> sorted(alloc.sorted().values()) in [[['x'], []], [[], ['x']]]
    True
    """

    inst = alloc.instance
    if len(inst.agents) != 2:
        raise ValueError("category_w_max_two_agents supports exactly two agents.")
    agents = list(inst.agents)

    # --- Build a *live* view of remaining items per category ----------------
    items_by_cat = defaultdict(list)
    for itm in alloc.remaining_items():
        cat = inst.item_categories.get(itm, "__NO_CAT__")
        items_by_cat[cat].append(itm)

    # --- Solve a small w-max instance for each category ---------------------
    for cat, items_in_cat in items_by_cat.items():
        # Skip if nothing left in this category.
        if not items_in_cat:
            continue

        # ----------------------------------------------------------------
        #  Prepare a *temporary* AllocationBuilder that contains only
        #  the still-free items **of this category**, so that the inner
        #  w_max_two_agents() from earlier cannot over-allocate.
        # ----------------------------------------------------------------
        tmp_alloc = AllocationBuilder(
            # alloc.remaining_instance()        # same valuations/conflicts…
            alloc.instance       # same valuations/conflicts…
        )

        # 1. Keep only items from this category.
        for itm in list(tmp_alloc.remaining_item_capacities):
            if itm not in items_in_cat:
                tmp_alloc.remove_item_from_loop(itm)

        # 2. Throttle agent copies to *category* capacity instead of full
        #    remaining capacity.  (We do that by manually shrinking the
        #    per-agent capacity counters.)
        if inst.agents_category_capacities is not None:
            for ag in agents:
                # remaining seats agent `ag` may still take in this category
                seats_left = alloc.agents_category_capacities[ag][cat]
                tmp_alloc.remaining_agent_capacities[ag] = min(
                    tmp_alloc.remaining_agent_capacities.get(ag, 0),
                    seats_left,
                )

        # If after trimming an agent has capacity 0, remove them so the helper
        # does not create useless copies.
        for ag in list(tmp_alloc.remaining_agent_capacities):
            if tmp_alloc.remaining_agent_capacities[ag] <= 0:
                tmp_alloc.remove_agent_from_loop(ag)

        # 3. Run the *plain* w-max on this trimmed instance.
        if not tmp_alloc.isdone():
            w_max_two_agents2(tmp_alloc, weights=weights)

        # 4. Copy the matched items back into the real allocation.
        alloc.give_bundles(tmp_alloc.bundles)
        # 5. Loop to next category (capacities already updated by give()).


# Note: this methdod do not check the edge case of no good or no chores because its EF1
def is_EF11(
        instance: "Instance",
        bundles: Dict[Hashable, Iterable[Hashable]],
) -> Optional[Tuple[Hashable, Hashable]]:
    """
    Return **True** iff the given allocation is envy-free-up-to-one-good-and-one-chore
    (EF[1,1]) with respect to the *instance*.
    Fit only for 1 good and 1 chore only goods and chores will be checked using EF1

    Parameters
    ----------
    instance : Instance
        The instance that supplies all valuations and (optionally) item categories
        via `instance.agent_item_value(a,i)`  and `instance.item_categories[i]`.
    bundles : dict(agent -> iterable(item))
        The final allocation, e.g. the `bundles` attribute of an
        `AllocationBuilder`, or its `.sorted()` result.

    The definition implemented is the one highlighted in the paper:

        For every ordered pair of agents (i, j)
        either i already values her bundle at least as much as j’s bundle, **or**
        there exists
            • one *chore* (negatively-valued item for i) in i’s bundle and
            • one *good*  (positively-valued item for i) in j’s bundle
        that lie in the *same category* (if categories are used), such that after
        simultaneously removing those two items i does not envy j.

    Complexity
    ----------
    *Pre-processing*          Θ(Σ |B_a|) (one pass over the allocation)
    *Pair check* (worst-case) Θ(|C|) for each ordered pair (i, j) –
                              thus overall  Θ(n²·|C|) where |C| ≤ number of items.
                              In typical course-allocation sizes this is negligible.
    The procedure is streaming and keeps only O(n·|C|) numbers.
    """
    agents = list(bundles.keys())

    # ---------- Helper: category of an item ----------
    if getattr(instance, "item_categories", None) is None:
        # If no categories are supplied, treat all items as belonging to the single
        # dummy category None, which exactly matches the paper's definition with no
        # category restriction.
        cat_of = lambda item: None
    else:
        item_cat = instance.item_categories
        cat_of   = lambda item: item_cat[item]

    # ---------- Pre-compute per agent ----------
    #
    # total_value[a]                – v_a(B_a)
    # min_neg[a][c]   (optional)    – least (i.e. most negative) value in category c
    # max_pos[a][c]   (optional)    – highest positive value in category c
    #
    min_neg = {a: {} for a in agents}
    max_pos = {a: {} for a in agents}
    total_value = {}

    for a, B in bundles.items():
        tot = 0.0
        for item in B:
            v = instance.agent_item_value(a, item)
            tot += v
            c = cat_of(item)

            if v < 0:                          # chore for a
                d = min_neg[a].get(c,  inf)
                if v < d:                       # store *most* negative
                    min_neg[a][c] = v
            elif v > 0:                         # good for a
                d = max_pos[a].get(c, -inf)
                if v > d:
                    max_pos[a][c] = v
            # v == 0 never helps either side, so ignore
        total_value[a] = tot

    # ---------- Check every ordered pair (i, j) ----------
    for i in agents:
        for j in agents:
            if i == j:
                continue

            gap = total_value[i] - total_value[j]      # positive ⇒ no envy
            if gap >= 0:
                continue                               # i already happy

            # i envies j.  We must bridge |gap| by removing one chore from i
            # and one good from j in the *same* category.
            needed = -gap                              # amount we must gain

            # Iterate only over categories where both sides *could* help.
            #
            # Note:  min_neg[i]       – categories where i owns at least one chore
            #        max_pos[j]       – categories where j owns at least one good
            candidate_categories = set(min_neg[i]).intersection(max_pos[j])

            for c in candidate_categories:
                gain = -min_neg[i][c] + max_pos[j][c]  # −v_i(chore) + v_i(good)
                if gain >= needed:                     # envy eliminated
                    break
            else:
                return (i, j)  # Found a violating pair

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

            # 1 Plain no-envy
            if v_iAi + abs_tol >= v_iAj:
                continue
            envy_gone = False

            # 2 Remove one item from j’s bundle
            for x in A_j:
                if v_iAi + abs_tol >= v_iAj - inst.agent_item_value(i, x):
                    envy_gone = True
                    break

            # 3 Otherwise, try removing one item from i’s own bundle
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
    # logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    print(doctest.testmod())







#________________________ Private functions havent been tested ________________________
# ----------------------------------------------------------------------
#  W-maximisation for two agents my greedy algorithm, hevent been tested
# ----------------------------------------------------------------------
# Hevent been tested
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

# Hevent been tested and the complexity is too large
# def is_EF11(allocation: "AllocationBuilder",
#         abs_tol: float = 1e-9
#         ) -> Optional[Tuple[any, any]]:
# r"""
# Check whether the current allocation satisfies EF[1,1].
#
# Parameters
# ----------
# allocation : AllocationBuilder
#     The incremental allocation object whose `.bundles` and
#     `.instance` fields we inspect.
# abs_tol : float, optional
#     A tiny slack to absorb floating-point noise.
#
# Returns
# -------
# None                     –– if every pair (i,j) meets EF[1,1].
# (i, j) : Tuple[agent, agent]
#     The first counter-example found:
#     • `i` is the envious agent,
#     • `j` is the agent she envies **after** all EF[1,1] patches fail.
#
# Notes
# -----
# EF[1,1] says that for every pair of agents *i, j*,
#     uᵢ(Aᵢ) ≥ uᵢ(Aⱼ)                                         (no-envy),  OR
#     ∃ (good g ∈ Aⱼ, chore h ∈ Aᵢ) in the *same* category s.t.
#     uᵢ(Aᵢ \ {h}) ≥ uᵢ(Aⱼ \ {g}).
#
# • A **good** is an item with uᵢ(item) > 0 for agent *i*;
#   a **chore** has uᵢ(item) < 0.
# • When `instance.item_categories is None` we treat all
#   items as belonging to one universal category.
# """
# inst = allocation.instance
# bundles = allocation.bundles  # maps agent → set/list of items :contentReference[oaicite:0]{index=0}
#
# # Pre-compute each agent's value for her own bundle once.
# bundle_value = {
#     agent: inst.agent_bundle_value(agent, bundles[agent])
#     for agent in inst.agents
# }
#
# # Iterate over ordered pairs (i, j).
# for i in inst.agents:
#     for j in inst.agents:
#         if i == j:
#             continue
#
#         # i's utility for j's bundle
#         u_i_Aj = inst.agent_bundle_value(i, bundles[j])
#         # i's utility for its own bundle
#         u_i_Ai = bundle_value[i]
#
#         # If i already does not envy j, move on.
#         if u_i_Ai + abs_tol >= u_i_Aj:
#             continue
#
#         # Else: look for a good–chore pair in the same category
#         envy_eliminated = False
#         for g in bundles[j]:
#             val_g = inst.agent_item_value(i, g)
#             if val_g <= 0:  # not a good for i
#                 continue
#             cat_g = (inst.item_categories[g]
#                      if inst.item_categories is not None else None)
#
#             for h in bundles[i]:
#                 val_h = inst.agent_item_value(i, h)
#                 if val_h >= 0:  # not a chore for i
#                     continue
#                 # same category? (trivial if categories are disabled)
#                 if (inst.item_categories is not None
#                         and inst.item_categories[h] != cat_g):
#                     continue
#
#                 # Check the EF[1,1] inequality:
#                 # take only tha smallest and the biggest
#                 if (u_i_Ai - val_h + abs_tol) >= (u_i_Aj - val_g):
#                     envy_eliminated = True
#                     break
#             if envy_eliminated:
#                 break
#
#         # No patch found ⇒ true envy remains.
#         if not envy_eliminated:
#             return (i, j)
#
# # All pairs passed
# return None
