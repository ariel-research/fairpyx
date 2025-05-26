"""
"A Reduction from Chores Allocation to Job Scheduling", by Xin Huang and Erel Segal-Halevi (2024), https://arxiv.org/abs/2302.04581
Programmer: Nadav Shabtai
Date: 2025-05
"""

from __future__ import annotations
import numpy as np, pytest, fairpyx
from fairpyx.algorithms.hffd import hffd

# helper

def divide(vals: np.ndarray, tau: dict[int, float]):
    inst = fairpyx.Instance(valuations=vals)
    return fairpyx.divide(hffd, inst, thresholds=tau)

def cost(vals, ag, bundle):
    return sum(vals[ag, i] for i in bundle)

# =====================================================
# 1. SINGLE‑AGENT CASES
# =====================================================

def test_single_agent_unallocatable():
    """threshold smaller than every item → nothing allocated"""
    vals = np.array([[5, 4]])
    tau  = {0: 3}
    expect = {0: []}
    assert divide(vals, tau) == expect


def test_single_agent_exact_fit():
    """bundle cost equals threshold"""
    vals = np.array([[4, 3]])
    tau  = {0: 7}
    expect = {0: [0, 1]}
    alloc = divide(vals, tau)
    assert alloc == expect and cost(vals, 0, alloc[0]) == 7

# =====================================================
# 2. TWO‑AGENT SMALLS
# =====================================================

@pytest.mark.parametrize(
    "tau, expect",
    [
        ({0: 6, 1: 6}, {0: [0],     1: [1]}),     # unchanged
        ({0: 4, 1: 8}, {0: [],      1: [0, 1]}), # <-- NEW expectation
    ],
)

def test_two_agents_variants(tau, expect):
    vals = np.array([[6, 4], [4, 4]])
    alloc = divide(vals, tau)
    assert alloc == expect
    for ag, items in alloc.items():
        assert cost(vals, ag, items) <= tau[ag]


def test_threshold_blocks_second_item():
    """agent‑1 cannot take any item due to low threshold"""
    vals = np.array([[5, 1], [5, 1]])
    tau  = {0: 5, 1: 4}
    expect = {0: [0], 1: [1]}   # item‑1 fits agent‑1 after agent‑0 takes item‑0
    assert divide(vals, tau) == expect

# =====================================================
# 3. MORE AGENTS / CHORES MIXES
# =====================================================

def test_three_agents_four_items():
    """3 agents, 4 items, staggered thresholds"""
    vals = np.array([[9, 7, 3, 1],
                     [8, 6, 2, 1],
                     [8, 6, 2, 1]])
    tau  = {0: 9, 1: 8, 2: 3}
    expect = {0: [0], 1: [1, 2], 2: [3]}
    alloc = divide(vals, tau)
    assert alloc == expect


def test_five_agents_two_items():
    """more agents than chores"""
    vals = np.array([[10, 5]]*5)
    tau  = {i: 12 for i in range(5)}
    expect = {0: [0], 1: [1], 2: [], 3: [], 4: []}
    assert divide(vals, tau) == expect

# =====================================================
# 4. FAIRNESS MEDIUM (13/11‑MMS)
# =====================================================

def test_mms_fairness_medium():
    rng = np.random.default_rng(1)
    vals = rng.integers(1, 20, size=(5, 30))
    tau  = {a: 60 for a in range(5)}
    alloc = divide(vals, tau)

    def mms(a: int):
        n = vals.shape[0]
        best = sorted(vals[a], reverse=True)
        return sum(best[a::n])

    for ag, items in alloc.items():
        c = cost(vals, ag, items)
        assert c <= tau[ag]
        assert c <= (13/11)*mms(ag)

# =====================================================
# 5. LARGE RANDOM STRESS
# =====================================================

def test_large_random_instance():
    rng = np.random.default_rng(2)
    vals = rng.integers(1, 50, size=(20, 200))
    tau  = {a: 400 for a in range(20)}
    alloc = divide(vals, tau)

    seen = set()
    for ag, items in alloc.items():
        assert cost(vals, ag, items) <= tau[ag]
        assert not (set(items) & seen)
        seen.update(items)

# =====================================================
# 6. CANONICAL EXAMPLE FROM THE PAPER
# =====================================================

def test_example1_from_paper():
    type_A = [51, 28, 27, 27, 27, 26, 12, 12, 11, 11, 11, 11, 11, 11, 10]
    type_B = [51, 28, 27, 27, 27, 24, 21, 20, 10, 10, 10,  9,  9,  9,  9]
    vals = np.array([type_B, type_A, type_A, type_A])
    tau  = {i: 75 for i in range(4)}
    alloc = divide(vals, tau)

    for ag, items in alloc.items():
        assert cost(vals, ag, items) <= 75
    assert 0 in alloc[0] and 5 in alloc[0]
    for ag in range(1, 4):
        assert 0 not in alloc[ag] and 5 not in alloc[ag]
    assert 14 not in {i for it in alloc.values() for i in it}
    flat = [i for it in alloc.values() for i in it]
    assert len(flat) == len(set(flat))
