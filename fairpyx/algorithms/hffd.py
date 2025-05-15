"""
An implementation of the algorithm in:
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer : Nadav Shabtai
Date : 2025-05
"""


from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import fairpyx


def hffd(
    builder: "fairpyx.AllocationBuilder",
    /,
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """
    The Algorithm packs chores into each agent’s bundle in non-increasing cost order while
    respecting the agent-specific threshold (maximum total cost allowed).

    Parameters:
    - builder : AllocationBuilder            # mutable allocation helper
    - thresholds : list[int] | None          # cost‐threshold per agent (not capacity)
    - rng : np.random.Generator | None       # optional tie-breaking source

    Returns:
    - None – the allocation is written in-place to builder.

    >>> import fairpyx, numpy as np
    >>> from fairpyx.algorithms.hffd import hffd

    >>> # Example 1 – perfect allocation
    >>> vals = np.array([[8, 5, 5, 2],
    ...                  [8, 5, 5, 2]])
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[10, 10],
    ...                         agent_thresholds=[10, 10])
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}

    >>> # Example 2 – intentional failure
    >>> vals = np.array([[9, 8, 4],
    ...                  [9, 8, 4]])
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[10, 10],
    ...                         agent_thresholds=[10, 10])
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}

    >>> # Example 3 – exercise every *if* branch
    >>> vals = np.array([[6, 5, 2],
    ...                  [6, 5, 2]])
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[7, 7],
    ...                         agent_thresholds=[7, 7])
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}

    >>> # Example 4 – typical random instance
    >>> rng = np.random.default_rng(0)
    >>> vals = rng.integers(1, 8, size=(3, 7))
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[12]*3,
    ...                         agent_thresholds=[12]*3)
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}

    >>> # Example 5 – paper counter-example (Example 1)
    >>> vals_A = [51,28,27,27,27,26,12,12,11,11,11,11,11,11,10]
    >>> vals_B = [51,28,27,27,27,24,21,20,10,10,10, 9, 9, 9, 9]
    >>> vals = np.array([vals_B, vals_A, vals_A, vals_A])  # BAAA
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[15]*4,
    ...                         agent_thresholds=[75]*4)
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}

    >>> # Example 6 – minimal sanity check
    >>> vals = np.array([[8, 7, 2, 1],
    ...                  [8, 7, 2, 1]])
    >>> inst = fairpyx.Instance(valuations=vals,
    ...                         agent_capacities=[10, 10],
    ...                         agent_thresholds=[10, 10])
    >>> fairpyx.divide(hffd, inst)  # doctest: +ELLIPSIS
    {...}
    """
    # Give an empty bundle to every agent.
    for agent in builder.agents:
        builder.give(agent, [])  # type: ignore[arg-type]


__all__: List[str] = ["hffd"]

