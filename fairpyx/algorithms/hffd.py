"""
An implementation of the algorithm in:
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer: Nadav Shabtai
Date : 2025-05
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["hffd"]

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _as_list(obj) -> list[Any]:
    """Return a concrete *list* out of an attribute or a zero‑arg method."""
    return list(obj() if callable(obj) else obj)

# ---------------------------------------------------------------------------
# algorithm
# ---------------------------------------------------------------------------

def hffd(
    builder,
    *,
    thresholds: Mapping[Any, float],
    universal_order: Sequence[Any] | None = None,
) -> None:
    """
    Allocates chores to agents with heterogeneous costs under Identical-Order Preference (IDO),
    creating n bundles in non-increasing cost order, assigning each to an agent within their threshold.

    Parameters:
    - builder: AllocationBuilder – mutable helper that stores the instance and lets the algorithm assign items with
      give(agent, item).
    - thresholds: Mapping[Any, float] – per-agent cost limit τₐ that a bundle may not exceed.
    - universal_order: Sequence[Any] | None – identical ranking of chores (largest → smallest); None means use
      builder.remaining_items.

    Returns:
    None – the function mutates *builder* in‑place.

    Examples:
    >>> import fairpyx, numpy as np
    >>> vals = np.array([[8, 5, 5, 2], [8, 5, 5, 2]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
    {0: [0, 3], 1: [1, 2]}

    >>> vals = np.array([[9, 8, 4], [9, 8, 4]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
    {0: [0], 1: [1]}

    >>> vals = np.array([[6, 5, 2], [6, 5, 2]])
    >>> inst = fairpyx.Instance(valuations=vals)
    >>> fairpyx.divide(hffd, inst, thresholds={0: 7, 1: 7})
    {0: [0], 1: [1, 2]}

    >>> rng = np.random.default_rng(0)
>>> vals = rng.integers(1, 8, size=(3, 7))
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 12, 1: 12, 2: 12})
{0: [0, 1, 5], 1: [2, 3], 2: [4, 6]}

    >>> vals_A = [51, 28, 27, 27, 27, 26, 12, 12, 11, 11, 11, 11, 11, 11, 10]
>>> vals_B = [51, 28, 27, 27, 27, 24, 21, 20, 10, 10, 10, 9, 9, 9, 9]
>>> vals = np.array([vals_B, vals_A, vals_A, vals_A])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 75, 1: 75, 2: 75, 3: 75})
{0: [0, 5], 1: [1, 2, 6], 2: [3, 4, 7], 3: [8, 9, 10, 11, 12, 13]}


    >>> vals = np.array([[8, 7, 2, 1], [8, 7, 2, 1]])
>>> inst = fairpyx.Instance(valuations=vals)
>>> fairpyx.divide(hffd, inst, thresholds={0: 10, 1: 10})
{0: [0, 2], 1: [1, 3]}
    """
    # Normalise dynamic lists from builder
    agents = _as_list(getattr(builder, "agents", builder.instance.agents))
    items  = _as_list(getattr(builder, "remaining_items", builder.remaining_items))

    # validate thresholds length & coerce if a list is passed
    if not isinstance(thresholds, Mapping):
        thresholds = dict(zip(agents, thresholds))
    if len(thresholds) != len(agents):
        raise ValueError("thresholds size must match number of agents")

    # residual thresholds τ̂ₐ
    tau = {a: float(thresholds[a]) for a in agents}

    # cost lookup
    inst = builder.instance
    cost = {(a, i): inst.agent_item_value(a, i) for a in agents for i in items}

    # decide order
    order = list(universal_order) if universal_order is not None else items
    if set(order) != set(items):
        raise ValueError("universal_order must be a permutation of items")
    
    # First‑Fit loop
    for i in order:
        for a in agents:
            if cost[(a, i)] <= tau[a]:
                builder.give(a, i)
                tau[a] -= cost[(a, i)]
                break  # next item



