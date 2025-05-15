"""
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer: Nadav Shabtai
Date: 2025-05
"""

import random
import numpy as np
import pytest
import fairpyx
from fairpyx.algorithms.hffd import hffd


# Preserve RNG state
@pytest.fixture(autouse=True)
def _preserve_rng_state():
    state_py, state_np = random.getstate(), np.random.get_state()
    yield
    random.setstate(state_py)
    np.random.set_state(state_np)


# ========== 1. Edge-case sanity checks ============================

@pytest.mark.parametrize(
    "valuations, agent_caps, thresholds, expected",
    [
        ({}, [], [], {}),  # Empty instance
        ([[0, 0, 0]], [1], [1], {0: [0, 1, 2]}),  # Zero valuations
        ([[5]], [0], [0], {0: []}),  # Zero threshold
    ],
    ids=["empty", "zero_vals", "zero_threshold"]
)
def test_edge_cases(valuations, agent_caps, thresholds, expected):
    """Test HFFD on degenerate cases; fails without implementation."""
    inst = fairpyx.Instance(valuations=valuations, agent_capacities=agent_caps)
    allocation = fairpyx.divide(hffd, inst, thresholds=thresholds)
    assert allocation == expected, f"Expected {expected}, got {allocation}"


# ========== 2. Small instances from doctests ======================

@pytest.mark.parametrize(
    "valuations, thresholds, expected",
    [
        # Example 1: Perfect allocation
        ([[8, 5, 5, 2], [8, 5, 5, 2]], [10, 10], {0: [0, 3], 1: [1, 2]}),
        # Example 2: Incomplete allocation
        ([[9, 8, 4], [9, 8, 4]], [10, 10], {0: [0], 1: [1]}),
        # Example 3: Branch coverage
        ([[6, 5, 2], [6, 5, 2]], [7, 7], {0: [0], 1: [1, 2]}),
        # Example 6: Sanity check
        ([[8, 7, 2, 1], [8, 7, 2, 1]], [10, 10], {0: [0, 2], 1: [1, 3]}),
    ],
    ids=["perfect", "incomplete", "branch", "sanity"]
)
def test_small_instances(valuations, thresholds, expected):
    """Test HFFD on small instances; fails without implementation."""
    inst = fairpyx.Instance(valuations=valuations, agent_capacities=thresholds)
    allocation = fairpyx.divide(hffd, inst, thresholds=thresholds)
    assert allocation == expected, f"Expected {expected}, got {allocation}"
    # Will verify thresholds once implemented
    for agent, items in allocation.items():
        cost = sum(valuations[agent][item] for item in items)
        assert cost <= thresholds[agent], f"Agent {agent} cost {cost} exceeds threshold {thresholds[agent]}"


# ========== 3. Large-scale instance ==============================

def test_large_input():
    """Test HFFD on a large instance; fails if no allocation made."""
    np.random.seed(0)
    valuations = np.random.randint(1, 100, (40, 2000))
    thresholds = [500] * 40
    inst = fairpyx.Instance(valuations=valuations, agent_capacities=thresholds)
    allocation = fairpyx.divide(hffd, inst, thresholds=thresholds)

    # Check some allocation exists
    assert any(allocation.values()), "No chores allocated"
    # Check no overlapping allocations
    allocated = set()
    for items in allocation.values():
        assert not (set(items) & allocated), "Overlapping chores allocated"
        allocated.update(items)


# ========== 4. Thresholds default ================================

def test_default_thresholds():
    """Test HFFD with default thresholds; fails without implementation."""
    valuations = [[8, 5, 5, 2], [8, 5, 5, 2]]
    inst = fairpyx.Instance(valuations=valuations, agent_capacities=[10, 10])
    allocation = fairpyx.divide(hffd, inst)  # No thresholds, expect default
    expected = {0: [0, 3], 1: [1, 2]}  # Same as Example 1
    assert allocation == expected, f"Expected {expected}, got {allocation}"
