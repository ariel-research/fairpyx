"""
An implementation of the algorithm in:
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer : Nadav Shabtai
Date : 2025-05
"""

from typing import List, Dict, Union, Optional
import numpy as np
from fairpyx import AllocationBuilder

__all__ = ["hffd"]

def hffd(
    builder: AllocationBuilder,
    universal_order: Optional[List[Union[str, int]]]=None,
    thresholds: Optional[Union[float, List[float], Dict[Union[str, int], float]]]=None
) -> None:
    """
    Heterogeneous First Fit Decreasing (HFFD) algorithm for fair chore allocation.

    Allocates chores to agents with heterogeneous costs under Identical-Order Preference (IDO),
    creating n bundles in non-increasing cost order, assigning each to an agent within their threshold.

    Parameters:
    - builder - AllocationBuilder - Mutable allocation helper, tracks instance and assignments.
    - universal_order - Optional[List[Union[str, int]]] - Chore order (largest to smallest cost); defaults to index order.
    - thresholds - Optional[Union[float, List[float], Dict[Union[str, int], float]]] - Max cost per agent; defaults to alpha * MMS.

    Returns:
        None. Modifies builder in-place with chore assignments.

    Raises:
        NotImplementedError: If agent_conflicts are provided.
        ValueError: If universal_order or thresholds are invalid.

    Examples:
        >>> import fairpyx, numpy as np
        >>> # Example 1: Perfect allocation
        >>> vals = np.array([[8, 5, 5, 2], [8, 5, 5, 2]])
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[10, 10])
        >>> fairpyx.divide(hffd, inst, thresholds=[10, 10])
        {0: [0, 3], 1: [1, 2]}

        >>> # Example 2: Incomplete allocation
        >>> vals = np.array([[9, 8, 4], [9, 8, 4]])
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[10, 10])
        >>> fairpyx.divide(hffd, inst, thresholds=[10, 10])
        {0: [0], 1: [2]}

        >>> # Example 3: Branch coverage
        >>> vals = np.array([[6, 5, 2], [6, 5, 2]])
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[7, 7])
        >>> fairpyx.divide(hffd, inst, thresholds=[7, 7])
        {0: [0], 1: [1, 2]}

        >>> # Example 4: Random instance
        >>> rng = np.random.default_rng(0)
        >>> vals = rng.integers(1, 8, size=(3, 7))
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[12]*3)
        >>> fairpyx.divide(hffd, inst, thresholds=[12]*3)
        {0: [0, 4], 1: [1, 5], 2: [2]}

        >>> # Example 5: Paper counter-example
        >>> vals_A = [51, 28, 27, 27, 27, 26, 12, 12, 11, 11, 11, 11, 11, 11, 10]
        >>> vals_B = [51, 28, 27, 27, 27, 24, 21, 20, 10, 10, 10, 9, 9, 9, 9]
        >>> vals = np.array([vals_B, vals_A, vals_A, vals_A])
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[75]*4)
        >>> fairpyx.divide(hffd, inst, thresholds=[75]*4)
        {0: [0, 5], 1: [1, 6, 7], 2: [2, 3, 8, 9], 3: [4, 10, 11, 12, 13]}

        >>> # Example 6: Sanity check
        >>> vals = np.array([[8, 7, 2, 1], [8, 7, 2, 1]])
        >>> inst = fairpyx.Instance(valuations=vals, agent_capacities=[10, 10])
        >>> fairpyx.divide(hffd, inst, thresholds=[10, 10])
        {0: [0, 3], 1: [1, 2]}
    """
    pass  # Implementation to be added
