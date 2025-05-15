"""
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer : Nadav Shabtai
Date : 2025-05
"""

import random
import numpy as np
import pytest
import fairpyx
from fairpyx.algorithms.hffd import hffd


# ------------------------------------------------------------------
# A session-wide fixture that preserves the global RNG state.
# This prevents our tests from interfering with other tests that rely
# on Python’s or NumPy’s random module.
# ------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _preserve_rng_state():
    state_py, state_np = random.getstate(), np.random.get_state()
    yield
    random.setstate(state_py)
    np.random.set_state(state_np)


# ========== 1. Edge-case sanity checks  ============================

@pytest.mark.parametrize(
    "valuations, agent_caps",
    [
        ({},              []),   # completely empty instance
        ([[0, 0, 0]],      [1]), # single agent, all zero values
        ([[5]],            [0]), # agent with zero capacity
    ]
)
def test_edge_cases_expect_non_empty_allocation(valuations, agent_caps):
    """Even in degenerate cases HFFD should return *something* (once implemented)."""
    inst = fairpyx.Instance(
        valuations=valuations,
        agent_capacities=agent_caps or None
    )
    allocation = fairpyx.divide(hffd, inst)

    # At this stage we expect an empty allocation => the test should FAIL.
    assert any(allocation.values()), "HFFD returned an empty allocation"


# ========== 2. Large-scale instance  ===============================

def test_large_input_allocates_everything_for_now():
    """Large random instance – HFFD is expected to mis-behave until implemented."""
    np.random.seed(0)
    inst = fairpyx.Instance(
        valuations=np.random.randint(1, 100, (40, 2000)),
        agent_capacities=[500] * 40
    )
    allocation = fairpyx.divide(hffd, inst)

    # Necessary condition: no items should remain unallocated (will FAIL for now).
    assert not inst.remaining_items(allocation), \
        "HFFD failed to allocate all items on a large input"


# ========== 3. Random instance vs. baseline FFD ====================

def test_random_instance_vs_ffd_baseline():
    """Compare item-coverage of HFFD to classic FFD on a random instance."""
    np.random.seed(1)
    agents, items = 5, 20
    vals = np.random.randint(1, 50, (agents, items))
    inst = fairpyx.Instance(valuations=vals, agent_capacities=[10] * agents)

    # HFFD allocation (currently empty)
    alloc_hffd = fairpyx.divide(hffd, inst)

    # Baseline: FFD with identical valuations for all agents
    uniform_vals = np.tile(vals.max(axis=0), (agents, 1))
    alloc_ffd = fairpyx.divide(
        fairpyx.algorithms.ffd,
        inst.clone(valuations=uniform_vals)
    )

    total_hffd = sum(map(len, alloc_hffd.values()))
    total_ffd  = sum(map(len, alloc_ffd.values()))

    # For a working HFFD we expect >=, right now we assert equality and FAIL.
    assert total_hffd == total_ffd, \
        "HFFD allocates fewer items than the baseline FFD"

