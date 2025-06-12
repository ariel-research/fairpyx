"""
Tests for 'The Santa Claus Problem' algorithms.

Paper: The Santa Claus Problem
Authors: Nikhil Bansal, Maxim Sviridenko
Link: https://dl.acm.org/doi/10.1145/1132516.1132557 (Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006)

This file tests the implementations in santa_claus_problem.py, focusing on
the O(log log m / log log log m) approximation algorithm for the restricted
assignment case.

Programmers: [Your Name(s) Here - User will fill this]
Date: 2025-05-29
"""

import pytest
import numpy as np
import random
from fairpyx import Instance, divide
from fairpyx.algorithms.santa_claus_problem import (
    santa_claus_algorithm, 
    find_optimal_target_value, 
    configuration_lp_solver,
    create_super_machines,
    round_small_configurations,
    construct_final_allocation
)

def test_santa_claus_basic():
    """Test the Santa Claus algorithm on a simple example"""
    # Basic test instance
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 5},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 5},
            "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
        },
        agent_capacities={"Child1": 1, "Child2": 1, "Child3": 1},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1}
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    # Verify that each child gets exactly 1 gift
    assert len(allocation["Child1"]) == 1
    assert len(allocation["Child2"]) == 1
    assert len(allocation["Child3"]) == 1
    
    # Verify that each gift is allocated to exactly one child
    all_gifts = []
    for child, gifts in allocation.items():
        all_gifts.extend(gifts)
    assert sorted(all_gifts) == ["gift1", "gift2", "gift3"]
    
    # In the optimal solution, Child1 should get gift1, Child2 should get gift2, Child3 should get gift3
    # This would give each child a value of 10, maximizing the minimum happiness
    # We can't test this yet because we only have an empty implementation, but we'll verify the structure

def test_restricted_assignment_case():
    """Test the Santa Claus algorithm on the restricted assignment case"""
    # Instance where gifts have a base value and are either available (pij = pj) or unavailable (pij = 0)
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 0, "gift4": 10, "gift5": 1},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 0, "gift4": 0, "gift5": 1},
            "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 10, "gift5": 1}
        },
        agent_capacities={"Child1": 2, "Child2": 2, "Child3": 2},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    
    # Verify capacity constraints
    for child, gifts in allocation.items():
        assert len(gifts) <= instance.agent_capacities[child]
    
    # Check that each gift is allocated at most once
    all_gifts = []
    for child, gifts in allocation.items():
        all_gifts.extend(gifts)
    assert len(all_gifts) == len(set(all_gifts))
    
    # The optimal solution would give gift1 and gift4 to Child1, gift2 and gift5 to Child2, gift3 to Child3
    # This would give values of 20, 11, and 10, so the minimum is 10
    # We can't verify this with the empty implementation

def test_empty_instance():
    """Test the Santa Claus algorithm on an empty instance (edge case)"""
    # Instance with no gifts
    instance = Instance(
        valuations={"Child1": {}, "Child2": {}},
        agent_capacities={"Child1": 1, "Child2": 1},
        item_capacities={}
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    assert allocation == {"Child1": [], "Child2": []}

def test_single_child():
    """Test the Santa Claus algorithm with only one child (edge case)"""
    # Instance with one child
    instance = Instance(
        valuations={"Child1": {"gift1": 10, "gift2": 20}},
        agent_capacities={"Child1": 2},
        item_capacities={"gift1": 1, "gift2": 1}
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    assert len(allocation["Child1"]) <= 2
    for gift in allocation["Child1"]:
        assert gift in ["gift1", "gift2"]

def test_single_gift():
    """Test the Santa Claus algorithm with only one gift (edge case)"""
    # Instance with one gift
    instance = Instance(
        valuations={"Child1": {"gift1": 10}, "Child2": {"gift1": 20}},
        agent_capacities={"Child1": 1, "Child2": 1},
        item_capacities={"gift1": 1}
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    all_gifts = []
    for child, gifts in allocation.items():
        all_gifts.extend(gifts)
    assert len(all_gifts) <= 1
    if all_gifts:
        assert all_gifts[0] == "gift1"
        # The optimal solution would give gift1 to Child2 (value 20)
        # We can't verify this with the empty implementation

def test_large_instance():
    """Test the Santa Claus algorithm on a large instance"""
    # Create a large instance with 20 children and 50 gifts
    children = [f"Child{i}" for i in range(1, 21)]
    gifts = [f"gift{i}" for i in range(1, 51)]
    
    # Create random valuations for the restricted assignment case
    # Each gift has a base value, and each child can only use some of the gifts
    valuations = {}
    for child in children:
        child_valuations = {}
        for gift in gifts:
            # For each gift, with 40% probability the child can use it, otherwise value is 0
            if random.random() < 0.4:
                # Base value of the gift is between 1 and 100
                base_value = random.randint(1, 100)
                child_valuations[gift] = base_value
            else:
                child_valuations[gift] = 0
        valuations[child] = child_valuations
    
    instance = Instance(
        valuations=valuations,
        agent_capacities={child: 5 for child in children},  # Each child can get up to 5 gifts
        item_capacities={gift: 1 for gift in gifts}         # Each gift can be allocated once
    )
    
    allocation = divide(santa_claus_algorithm, instance)
    
    # Verify capacity constraints
    for child, gifts in allocation.items():
        assert len(gifts) <= instance.agent_capacities[child]
    
    # Check that each gift is allocated at most once
    all_gifts = []
    for child, gifts in allocation.items():
        all_gifts.extend(gifts)
    assert len(all_gifts) == len(set(all_gifts))
    
    # Check that each child only gets gifts they value (value > 0)
    for child, gifts_list in allocation.items():
        for gift in gifts_list:
            assert instance.valuations[child][gift] > 0

def test_random_instances():
    """Test the Santa Claus algorithm on multiple random instances"""
    for seed in range(1, 4):  # Test with 3 different random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Random number of children and gifts
        num_children = random.randint(3, 10)
        num_gifts = random.randint(5, 20)
        
        children = [f"Child{i}" for i in range(1, num_children+1)]
        gifts = [f"gift{i}" for i in range(1, num_gifts+1)]
        
        # Create random valuations
        valuations = {}
        for child in children:
            child_valuations = {}
            for gift in gifts:
                # With 50% probability, the gift has value for this child
                if random.random() < 0.5:
                    child_valuations[gift] = random.randint(1, 20)
                else:
                    child_valuations[gift] = 0
            valuations[child] = child_valuations
        
        # Random capacities
        agent_capacities = {child: random.randint(1, 5) for child in children}
        item_capacities = {gift: 1 for gift in gifts}  # Each gift can be allocated once
        
        instance = Instance(
            valuations=valuations,
            agent_capacities=agent_capacities,
            item_capacities=item_capacities
        )
        
        allocation = divide(santa_claus_algorithm, instance)
        
        # Verify capacity constraints
        for child, gifts_list in allocation.items():
            assert len(gifts_list) <= instance.agent_capacities[child]
        
        # Check that each gift is allocated at most once
        all_gifts = []
        for child, gifts_list in allocation.items():
            all_gifts.extend(gifts_list)
        assert len(all_gifts) == len(set(all_gifts))
        
        # Check that each child only gets gifts they value (value > 0)
        for child, gifts_list in allocation.items():
            for gift in gifts_list:
                assert instance.valuations[child][gift] > 0

def test_find_optimal_target_value():
    """Test the binary search for optimal target value"""
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 5},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 5},
            "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
        },
        agent_capacities={"Child1": 1, "Child2": 1, "Child3": 1},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1}
    )
    
    # Currently we have an empty implementation, so we can't test the actual value
    # But we can test that the function runs without errors
    builder = divide(find_optimal_target_value, instance, return_builder=True)
    assert isinstance(builder.allocation, dict)

def test_configuration_lp_solver():
    """Test the Configuration LP solver"""
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 5},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 5},
            "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
        },
        agent_capacities={"Child1": 1, "Child2": 1, "Child3": 1},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1}
    )
    
    builder = divide(lambda a: configuration_lp_solver(a, 10, 2), instance, return_builder=True)
    assert isinstance(builder.allocation, dict)

def test_create_super_machines():
    """Test the creation of super-machines"""
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 5},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 5},
            "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
        },
        agent_capacities={"Child1": 1, "Child2": 1, "Child3": 1},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1}
    )
    
    builder = divide(lambda a: create_super_machines(a, {}, {"gift1", "gift2"}), instance, return_builder=True)
    assert isinstance(builder.allocation, dict)

def test_round_small_configurations():
    """Test the rounding of small gift configurations"""
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 0, "gift4": 10, "gift5": 1},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 0, "gift4": 0, "gift5": 1},
            "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 10, "gift5": 1}
        },
        agent_capacities={"Child1": 2, "Child2": 2, "Child3": 2},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
    )
    
    small_gifts = {"gift5"}
    super_machines = [(["Child1", "Child2"], ["gift1", "gift2"])]
    
    builder = divide(lambda a: round_small_configurations(a, super_machines, small_gifts), instance, return_builder=True)
    assert isinstance(builder.allocation, dict)

def test_construct_final_allocation():
    """Test the construction of the final allocation"""
    instance = Instance(
        valuations={
            "Child1": {"gift1": 10, "gift2": 0, "gift3": 0, "gift4": 10, "gift5": 1},
            "Child2": {"gift1": 0, "gift2": 10, "gift3": 0, "gift4": 0, "gift5": 1},
            "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 10, "gift5": 1}
        },
        agent_capacities={"Child1": 2, "Child2": 2, "Child3": 2},
        item_capacities={"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
    )
    
    super_machines = [(["Child1", "Child2"], ["gift1", "gift2"])]
    rounded_solution = {0: {"child": "Child1", "gifts": ["gift5"]}}
    
    allocation = divide(lambda a: construct_final_allocation(a, super_machines, rounded_solution), instance)
    assert isinstance(allocation, dict)
    
    # Since we have an empty implementation, we can't test the actual allocation
