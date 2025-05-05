"""
Implementation of:

    "A Reduction from Chores Allocation to Job Scheduling",
    Xin Huang & Erel Segal-Halevi (2024)
    https://arxiv.org/abs/2302.04581

Algorithm 1 — Heterogeneous First Fit Decreasing (HFFD)

Programmer: Nadav <family-name>
Since: 2025-05
"""

from fairpyx import AllocationBuilder
import logging
logger = logging.getLogger(__name__)

def hffd(alloc: AllocationBuilder) -> None:
    """
    Allocate chores among heterogeneous agents using HFFD.

    Parameters
    ----------
    alloc : AllocationBuilder
        Tracks the partial allocation + remaining capacities.

    Returns
    -------
    None
        Mutates `alloc` in place.

    Notes
    -----
    • Algorithm #1 in the paper.  
    • *Empty implementation* → intended to fail current tests.

    Examples
    --------
    >>> from fairpyx import Instance, divide
    >>> inst = Instance(valuations=[[8,5,5,2],[8,5,5,2]], agent_capacities=[10,10])
    >>> divide(hffd, inst)  # doctest: +SKIP
    {'0': [], '1': []}
    """
    logger.info("HFFD started with %s items, %s agents",
                len(alloc.remaining_items()), len(alloc.remaining_agents()))
    # TODO: implement


