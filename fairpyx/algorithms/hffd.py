"""
An implementation of the algorithm in:
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer: Nadav Shabtai
Date : 2025-05
"""

from typing import Any, Mapping, Sequence
import logging

logger = logging.getLogger(__name__)          # bundle trace goes here
__all__ = ["hffd"]


# ---------------------------------------------------------------------------
# main algorithm
# ---------------------------------------------------------------------------

def hffd(
    builder,
    *,
    thresholds: Mapping[Any, float] | Sequence[float],
    universal_order: Sequence[Any] | None = None,
) -> None:
    """
    Allocates chores to agents with heterogeneous costs under Identical-Order
    Preference (IDO), creating bundles A₁…Aₙ and giving each to an agent whose
    total cost stays ≤ τₐ.

Parameters:
    - builder: AllocationBuilder – mutable helper that stores the instance and lets the algorithm assign items with
      give(agent, item).
    - thresholds: Mapping[Any, float] – per-agent cost limit τₐ that a bundle may not exceed.
    - universal_order: Sequence[Any] | None – identical ranking of chores (largest → smallest); None means use
      builder.remaining_items.

Returns:
    None – the function mutates *builder* in‑place.

Examples :

    >>> import fairpyx, numpy as np

    # 1. perfect fit – two identical agents, τ = 10
    >>> vals = np.array([[8, 5, 5, 2],
    ...                  [8, 5, 5, 2]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
    {0: [0, 3], 1: [1, 2]}

    # 2. unequal thresholds, one item left over
    >>> vals = np.array([[9, 8, 4],
    ...                  [9, 8, 4]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst, thresholds={0: 9, 1: 12})
    {0: [0], 1: [1, 2]}

    # 3. three agents, six chores, asymmetric values
    >>> vals = np.array([[7, 6, 1, 1, 1, 1],
    ...                  [5, 5, 3, 3, 3, 3],
    ...                  [8, 7, 3, 2, 1, 1]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst,
    ...                thresholds={0:10, 1:6, 2:12})
    {0: [0, 2, 3, 4], 1: [], 2: [1, 5]}

    # 4. canonical 15-item example from the paper (Example 1, §4)
    >>> A = [51, 28, 27, 27, 27, 26, 12, 12, 11, 11, 11, 11, 11, 11, 10]
    >>> B = [51, 28, 27, 27, 27, 24, 21, 20, 10, 10, 10,  9,  9,  9,  9]
    >>> vals = np.array([B, A, A, A])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst,
    ...                thresholds={0: 75, 1: 75, 2: 75, 3: 75})
    {0: [0, 5], 1: [1, 2, 6], 2: [3, 4, 7], 3: [8, 9, 10, 11, 12, 13]}

    # 5. random 4×12 instance – just check feasibility
    >>> rng = np.random.default_rng(0)
    >>> vals = rng.integers(1, 9, size=(4, 12))
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> out = fairpyx.divide(hffd, inst,
    ...                      thresholds={a: 13 for a in range(4)})
    >>> all(sum(vals[a, i] for i in items) <= 13
    ...     for a, items in out.items())
    True
    """
    inst     = builder.instance
    agents   = list(builder.remaining_agents())     # authoritative API
    items_0  = list(builder.remaining_items())

    # ---------- normalise thresholds ---------------------------------------
    if not isinstance(thresholds, Mapping):
        if len(thresholds) != len(agents):
            raise ValueError("threshold list length must equal #agents")
        thresholds = dict(zip(agents, thresholds))
    if set(thresholds) != set(agents):
        raise ValueError("thresholds must specify every agent exactly once")

    tau = {a: float(thresholds[a]) for a in agents}

    # ---------- decide common order ----------------------------------------
    order = list(universal_order) if universal_order is not None else items_0
    if set(order) != set(items_0):
        raise ValueError("universal_order must match remaining items exactly")

    # ---------- pre-compute cost lookup ------------------------------------
    cost = {(a, i): inst.agent_item_value(a, i) for a in agents for i in order}

    # ---------- build bundles A₁, A₂, … ------------------------------------
    remaining_items = set(order)
    agents_left     = agents.copy()
    bundle_no       = 0

    while agents_left and remaining_items:
        bundle: list[Any] = []

        # pass 1 – collect chores into current bundle
        for i in order:
            if i not in remaining_items:
                continue
            if any(sum(cost[(a, j)] for j in bundle) + cost[(a, i)] <= tau[a]
                   for a in agents_left):
                bundle.append(i)

        if not bundle:         # safeguard (should not occur under IDO)
            break

        # pass 2 – choose first agent who can accept full bundle
        chosen = next(a for a in agents_left
                      if sum(cost[(a, j)] for j in bundle) <= tau[a])

        bundle_cost = sum(cost[(chosen, j)] for j in bundle)
        bundle_no  += 1
        logger.info("Bundle #%d → agent %s : %s  (cost %.0f)",
                    bundle_no, chosen, bundle, bundle_cost)

        # assign bundle (item-by-item API)
        for j in bundle:
            builder.give(chosen, j)
            remaining_items.remove(j)

        tau[chosen] -= bundle_cost
        agents_left.remove(chosen)

    # ---------- final report -----------------------------------------------
    if remaining_items:
        logger.info("Unallocated chores: %s", sorted(remaining_items))
    else:
        logger.info("All chores allocated.")

if __name__=="__main__":
    import doctest
    print(doctest.testmod())

    import fairpyx
    from fairpyx.adaptors import divide
    valuations = {"Alice": {"c1": 10, "c2": 8, "c3": 6}, "Bob": {"c1": 10, "c2": 8, "c3": 6}, "Chana": {"c1": 6, "c2": 8, "c3": 10}, "Dana": {"c1": 6, "c2": 8, "c3": 10}}
    instance = fairpyx.Instance(valuations=valuations)
    divide(hffd, instance=instance, thresholds = [9,4,3,3], logger=print)
