"""
An implementation of the algorithm in:
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer: Nadav Shabtai
Date : 2025-05
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Mapping, Sequence

__all__ = ["hffd"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_list(obj) -> list[Any]:
    """Return a concrete list even if *obj* is a zero-arg method."""
    return list(obj() if callable(obj) else obj)

def _log(msg: str, sink: Callable | logging.Logger | None) -> None:
    """Flexibly emit *msg* to Logger / print / or stay silent."""
    if sink is None:
        return
    if isinstance(sink, logging.Logger):
        sink.info(msg)
    else:        # assume print-like
        sink(msg)

# ---------------------------------------------------------------------------
# main algorithm
# ---------------------------------------------------------------------------

def hffd(
    builder,
    *,
    thresholds: Mapping[Any, float],
    universal_order: Sequence[Any] | None = None,
    logger: Callable | logging.Logger | None = None,
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
    - logger : logging.Logger | Callable | None - If given, prints a short line for every bundle and leftover chores.

    Returns:
    None – the function mutates *builder* in‑place.


    Examples
--------
>>> import fairpyx, numpy as np

# 1. perfect fit for both agents
>>> vals = np.array([[8, 5, 5, 2],
...                  [8, 5, 5, 2]])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
{0: [0, 3], 1: [1, 2]}

# 2. second chore left unallocated
>>> vals = np.array([[9, 8, 4],
...                  [9, 8, 4]])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
{0: [0], 1: [1]}

# 3. tight thresholds give asymmetric bundles
>>> vals = np.array([[6, 5, 2],
...                  [6, 5, 2]])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 7, 1: 7})
{0: [0], 1: [1, 2]}

# 4. canonical 15-item example from the paper
>>> A = [51, 28, 27, 27, 27, 26, 12, 12, 11, 11, 11, 11, 11, 11, 10]
>>> B = [51, 28, 27, 27, 27, 24, 21, 20, 10, 10, 10,  9,  9,  9,  9]
>>> vals = np.array([B, A, A, A])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 75, 1: 75, 2: 75, 3: 75})
{0: [0, 5], 1: [1, 2, 6], 2: [3, 4, 7], 3: [8, 9, 10, 11, 12, 13]}

# 5. sanity with identical agents
>>> vals = np.array([[8, 7, 2, 1],
...                  [8, 7, 2, 1]])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
{0: [0, 2], 1: [1, 3]}

    """
    # ---- static data -------------------------------------------------------
    all_agents = _as_list(getattr(builder, "agents",
                                  builder.instance.agents))
    items_left = _as_list(getattr(builder, "remaining_items",
                                  builder.remaining_items))

    if not isinstance(thresholds, Mapping):
        thresholds = dict(zip(all_agents, thresholds))
    if len(thresholds) != len(all_agents):
        raise ValueError("thresholds size must equal number of agents")

    tau = {a: float(thresholds[a]) for a in all_agents}

    inst = builder.instance
    cost = {(a, i): inst.agent_item_value(a, i)
            for a in all_agents for i in items_left}

    order = list(universal_order) if universal_order is not None else items_left
    if set(order) != set(items_left):
        raise ValueError("universal_order must be a permutation of items")

    # ---- dynamic state -----------------------------------------------------
    agents_left      = all_agents.copy()   # T   (paper)
    remaining_items  = set(items_left)     # R

    bundle_count = 0

    # ---- outer loop: build bundles ----------------------------------------
    while agents_left and remaining_items:
        Ak: list[Any] = []                 # current bundle
        # Pass 1: gather chores into Ak
        for i in order:
            if i not in remaining_items:
                continue
            if any(sum(cost[(a, j)] for j in Ak) + cost[(a, i)] <= tau[a]
                   for a in agents_left):
                Ak.append(i)

        if not Ak:        # safeguard
            break

        # Pass 2: choose agent that fits Ak
        chosen = next(a for a in agents_left
                      if sum(cost[(a, j)] for j in Ak) <= tau[a])

        bundle_cost = sum(cost[(chosen, j)] for j in Ak)
        _log(f"Bundle #{bundle_count+1}: give {Ak} (cost {bundle_cost}) -> "
             f"agent {chosen}", logger)
        bundle_count += 1

        # Assign bundle
        for j in Ak:
            builder.give(chosen, j)        # single-item API
            remaining_items.remove(j)

        tau[chosen] -= bundle_cost
        agents_left.remove(chosen)

    # ---- final report ------------------------------------------------------
    if remaining_items:
        _log(f"Unallocated chores: {sorted(remaining_items)}", logger)
    else:
        _log("All chores allocated.", logger)


if __name__=="__main__":
    import doctest
    print(doctest.testmod())

    import fairpyx
    from fairpyx.adaptors import divide
    valuations = {"Alice": {"c1": 10, "c2": 8, "c3": 6}, "Bob": {"c1": 10, "c2": 8, "c3": 6}, "Chana": {"c1": 6, "c2": 8, "c3": 10}, "Dana": {"c1": 6, "c2": 8, "c3": 10}}
    instance = fairpyx.Instance(valuations=valuations)
    divide(hffd, instance=instance, thresholds = [9,4,3,3], logger=print)
