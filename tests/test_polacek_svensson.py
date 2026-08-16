"""
Unit tests for the Quasi-Polynomial Local Search algorithm.

Tests use the restricted max-min allocation model:
each item j has a fixed size p_j, and each agent either values it at p_j or 0.

Programmer: Rotem Melamed
Date: 08/11/2025
"""

import pytest
import random
import fairpyx
from fairpyx import Instance
from fairpyx.adaptors import divide
from fairpyx.algorithms.polacek_svensson import (
    qp_max_min_allocation,
)


def assert_threshold(instance: Instance, result: dict, opt: float, epsilon: float = 0.1):
    """
    Assert that every agent's bundle value >= OPT / (4 + epsilon).
    """
    alpha = 4 + epsilon
    required = opt / alpha
    for agent, bundle in result.items():
        value = instance.agent_bundle_value(agent, list(bundle))
        assert value >= required - 1e-6, \
            f"Agent {agent} has value {value:.4f}, expected >= OPT/(4+eps) = {required:.4f} " \
            f"(OPT={opt:.4f}, epsilon={epsilon})"


# ==============================================================================
#  Basic tests
# ==============================================================================

def test_single_agent_single_item():
    """
    Single agent, single item (size 10).
    """
    instance = Instance(valuations={"p0": {"r1": 10}})

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    assert result == {'p0': ['r1']}
    fairpyx.validate_allocation(instance, result, title="test_single_agent_single_item")
    assert_threshold(instance, result, opt=10)


def test_two_agents_shared_item():
    """
    r1=10, r2=5, r3=10.
    p0 eligible for {r1, r2}, p1 eligible for {r2, r3}.
    """
    instance = Instance(valuations={
        "p0": {"r1": 10, "r2": 5, "r3": 0},
        "p1": {"r1": 0, "r2": 5, "r3": 10},
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_two_agents_shared_item")
    assert_threshold(instance, result, opt=10)  # p0={r1}=10, p1={r3}=10


def test_two_agents_unequal_items():
    """
    r1=3, r2=3, r3=10.
    p0 eligible for {r1, r2, r3}, p1 eligible for {r1, r2}.
    """
    instance = Instance(valuations={
        "p0": {"r1": 3, "r2": 3, "r3": 10},
        "p1": {"r1": 3, "r2": 3, "r3": 0},
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_two_agents_unequal_items")
    assert_threshold(instance, result, opt=6)  # p0={r3}=10, p1={r1,r2}=6


def test_three_agents_four_items():
    """
    r1=10, r2=5, r3=8, r4=3.
    p0 eligible for {r1, r2}, p1 eligible for {r2, r3, r4}, p2 eligible for {r3, r4}.
    """
    instance = Instance(valuations={
        "p0": {"r1": 10, "r2": 5, "r3": 0, "r4": 0},
        "p1": {"r1": 0, "r2": 5, "r3": 8, "r4": 3},
        "p2": {"r1": 0, "r2": 0, "r3": 8, "r4": 3},
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_three_agents_four_items")
    assert_threshold(instance, result, opt=8)  # p0={r1}=10, p1={r2,r4}=8, p2={r3}=8


def test_symmetric_eligibility():
    """
    r1=8, r2=6, r3=6, r4=8.
    p0 eligible for {r1, r2}, p1 eligible for {r3, r4}.
    """
    instance = Instance(valuations={
        "p0": {"r1": 8, "r2": 6, "r3": 0, "r4": 0},
        "p1": {"r1": 0, "r2": 0, "r3": 6, "r4": 8},
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_symmetric_eligibility")
    assert_threshold(instance, result, opt=14)  # p0={r1,r2}=14, p1={r3,r4}=14


# ==============================================================================
#  Edge cases
# ==============================================================================

def test_zero_value_item():
    """
    Single agent, single item with size 0.
    """
    instance = Instance(valuations={"p0": {"r1": 0}})

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    assert result == {"p0": []}


def test_single_dominant_item():
    """
    One large item (r1=100) and small items (r2-r5=2 each).
    All agents eligible for all items.
    """
    val = {"r1": 100, "r2": 2, "r3": 2, "r4": 2, "r5": 2}
    instance = Instance(valuations={
        "p0": val, "p1": val, "p2": val, "p3": val,
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_single_dominant_item")
    assert_threshold(instance, result, opt=2)  # 4 agents sharing 4 small items of size 2


def test_all_agents_eligible_for_everything():
    """
    All agents eligible for all items.
    r1=12, r2=8, r3=6, r4=4.
    """
    val = {"r1": 12, "r2": 8, "r3": 6, "r4": 4}
    instance = Instance(valuations={"p0": val, "p1": val})

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_all_agents_eligible_for_everything")
    assert_threshold(instance, result, opt=14)  # p0={r1,r4}=16, p1={r2,r3}=14


def test_disjoint_eligibility():
    """
    Agents have completely disjoint eligible sets.
    """
    instance = Instance(valuations={
        "p0": {"r1": 7, "r2": 3, "r3": 0, "r4": 0, "r5": 0, "r6": 0},  # total = 10
        "p1": {"r1": 0, "r2": 0, "r3": 5, "r4": 5, "r5": 0, "r6": 0},  # total = 10
        "p2": {"r1": 0, "r2": 0, "r3": 0, "r4": 0, "r5": 4, "r6": 6},  # total = 10
    })

    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    fairpyx.validate_allocation(instance, result, title="test_disjoint_eligibility")
    assert_threshold(instance, result, opt=10)  # each agent gets their 2 items totaling 10


# ==============================================================================
#  Large input
# ==============================================================================

def test_large_input():
    """
    Large instance (10 agents, 30 items) with random valuations.
    Verifies the algorithm produces a valid allocation at scale.
    """
    rng = random.Random(42)
    items = [f"r{j}" for j in range(30)]
    item_sizes = {item: rng.randint(5, 20) for item in items}
    agents = [f"p{i}" for i in range(10)]
    instance = Instance(valuations={agent: dict(item_sizes) for agent in agents})
    result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)

    fairpyx.validate_allocation(instance, result, title="test_large_input")