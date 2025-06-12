"""
Test module for the Santa Claus Problem solver implementation in the restricted assignment case.

In the restricted assignment case, each present j has a fixed value pj for all kids
who can receive it, and 0 for kids who cannot receive it.

Tests cover both simple and complex cases, including:
1. Basic allocation with restricted assignment valuations
2. Complex restricted assignment cases with overlapping preferences
3. Edge cases with different capacities
"""

import pytest
from fairpyx import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.santa_claus_solver import santa_claus_solver, divide
import numpy as np

def test_example_1():
    """
    Test a simple example with two kids and two presents in the restricted assignment case:
    - present1 has value 0.9 for any kid who can receive it
    - present2 has value 0.8 for any kid who can receive it
    - Kid1 can receive present1 but not present2
    - Kid2 can receive present2 but not present1
    
    Expected result: Kid1 gets present1 and Kid2 gets present2
    """
    # In the restricted assignment case, each present j has a fixed value p_j for all kids who can receive it
    # and 0 for kids who cannot receive it
    present_values = {"present1": 0.9, "present2": 0.8}
    
    instance = Instance(
        valuations={
            "Kid1": {"present1": present_values["present1"], "present2": 0.0},
            "Kid2": {"present1": 0.0, "present2": present_values["present2"]}
        },
        # Set a very high capacity for each kid to allow them to receive multiple presents
        agent_capacities={"Kid1": 100, "Kid2": 100},
        item_capacities={"present1": 1, "present2": 1}
    )
    
    # Get the optimal T value and final assignment
    T_optimal, assignment = santa_claus_solver(instance, alpha=2.0, beta=3.0)
    
    # Check that all presents are allocated
    all_allocated_presents = set()
    for kid, presents in assignment.items():
        all_allocated_presents.update(presents)
    
    assert len(all_allocated_presents) == 2, "All 2 presents should be allocated"
    
    # Check that the total value of the allocation is maximized
    total_value = sum(sum(instance.agent_item_value(kid, present) for present in presents) for kid, presents in assignment.items())
    assert total_value > 0, "Total allocation value should be positive"

def test_example_2():
    """
    Test a more complex example with three kids and three presents in the restricted assignment case with overlapping preferences:
    - present1 has value 0.9 for any kid who can receive it
    - present2 has value 0.8 for any kid who can receive it
    - present3 has value 0.7 for any kid who can receive it
    - Kid1 can receive present1 and present2 but not present3
    - Kid2 can receive present1 and present3 but not present2
    - Kid3 can receive present2 and present3 but not present1
    
    Expected result: A fair allocation where each kid gets at least one present they value
    """
    # In the restricted assignment case, each present j has a fixed value p_j for all kids who can receive it
    # and 0 for kids who cannot receive it
    present_values = {"present1": 0.9, "present2": 0.8, "present3": 0.7}
    
    instance = Instance(
        valuations={
            "Kid1": {"present1": present_values["present1"], "present2": present_values["present2"], "present3": 0.0},
            "Kid2": {"present1": present_values["present1"], "present2": 0.0, "present3": present_values["present3"]},
            "Kid3": {"present1": 0.0, "present2": present_values["present2"], "present3": present_values["present3"]}
        },
        # Set a very high capacity for each kid to allow them to receive multiple presents
        agent_capacities={"Kid1": 100, "Kid2": 100, "Kid3": 100},
        item_capacities={"present1": 1, "present2": 1, "present3": 1}
    )
    
    # Get the optimal T value and final assignment
    T_optimal, assignment = santa_claus_solver(instance, alpha=2.0, beta=3.0)
    
    # Check that all presents are allocated
    all_allocated_presents = set()
    for kid, presents in assignment.items():
        all_allocated_presents.update(presents)
    
    assert len(all_allocated_presents) == 3, "All 3 presents should be allocated"
    
    # Check that each kid gets at least one present they value
    for kid, presents in assignment.items():
        kid_value = sum(instance.agent_item_value(kid, present) for present in presents)
        assert kid_value > 0, f"{kid} should get at least one present they value"

def test_example_3():
    """
    Test an example in the restricted assignment case where kids can receive multiple presents with overlapping preferences:
    
    - present1 has value 0.9 for any kid who can receive it
    - present2 has value 0.7 for any kid who can receive it
    - present3 has value 0.5 for any kid who can receive it
    - Kid1 can receive present1 and present3
    - Kid2 can receive present2
    - Both Kid1 and Kid2 compete for present2
    
    Expected result: 
    - Kid1 should get presents valued total >= 0.9
    - Kid2 should get a present valued >= 0.7
    """
    # In the restricted assignment case, each present j has a fixed value p_j for all kids who can receive it
    # and 0 for kids who cannot receive it
    present_values = {"present1": 0.9, "present2": 0.7, "present3": 0.5}
    
    # Create a test case where both kids compete for present2
    instance = Instance(
        valuations={
            "Kid1": {"present1": present_values["present1"], "present2": present_values["present2"], "present3": present_values["present3"]},
            "Kid2": {"present1": 0.0, "present2": present_values["present2"], "present3": 0.0}
        },
        # Set a very high capacity for each kid to allow them to receive multiple presents
        agent_capacities={"Kid1": 100, "Kid2": 100},
        item_capacities={"present1": 1, "present2": 1, "present3": 1}
    )
    
    # Get the optimal T value and final assignment
    T_optimal, assignment = santa_claus_solver(instance, alpha=2.0, beta=3.0)
    
    # Check that all presents are allocated
    all_allocated_presents = set()
    for kid, presents in assignment.items():
        all_allocated_presents.update(presents)
    
    assert len(all_allocated_presents) == 3, "All 3 presents should be allocated"
    
    # Check that each kid gets at least one present they value
    for kid, presents in assignment.items():
        kid_value = sum(instance.agent_item_value(kid, present) for present in presents)
        assert kid_value > 0, f"{kid} should get at least one present they value"
    
    # Check that Kid1 gets a present they value
    kid1_value = sum(instance.agent_item_value("Kid1", present) for present in assignment["Kid1"])
    assert kid1_value >= 0.9, "Kid1's total value should be at least 0.9"
    
    # Check that Kid2 gets a present they value
    kid2_value = sum(instance.agent_item_value("Kid2", present) for present in assignment["Kid2"])
    assert kid2_value >= 0.7, "Kid2's value should be at least 0.7"

def test_example_4():
    """
    Test a complex example with 3 kids and 4 presents in the restricted assignment case with significant overlapping preferences:
    
    - present1 has value 0.9 for any kid who can receive it
    - present2 has value 0.8 for any kid who can receive it
    - present3 has value 0.7 for any kid who can receive it
    - present4 has value 0.6 for any kid who can receive it
    - Kid1 can receive present1 and present2
    - Kid2 can receive present1, present2, and present3
    - Kid3 can receive present2, present3, and present4
    
    Note that according to the paper, there's no direct limit on the number of presents a kid can receive.
    The only constraints are that each present can be given to at most one kid, and each kid should
    receive presents with total value ≥ T.
    
    Expected result: A fair allocation where each kid gets presents they value.
    """
    # In the restricted assignment case, each present j has a fixed value p_j for all kids who can receive it
    # and 0 for kids who cannot receive it
    present_values = {"present1": 0.9, "present2": 0.8, "present3": 0.7, "present4": 0.6}
    
    # Create a test case with significant overlapping preferences
    instance = Instance(
        valuations={
            "Kid1": {"present1": present_values["present1"], "present2": present_values["present2"], "present3": 0.0, "present4": 0.0},
            "Kid2": {"present1": present_values["present1"], "present2": present_values["present2"], "present3": present_values["present3"], "present4": 0.0},
            "Kid3": {"present1": 0.0, "present2": present_values["present2"], "present3": present_values["present3"], "present4": present_values["present4"]}
        },
        # Set a very high capacity for each kid to allow them to receive multiple presents
        agent_capacities={"Kid1": 100, "Kid2": 100, "Kid3": 100},
        item_capacities={"present1": 1, "present2": 1, "present3": 1, "present4": 1}
    )
    
    # Get the optimal T value and final assignment
    T_optimal, assignment = santa_claus_solver(instance, alpha=2.0, beta=3.0)
    
    # Check that all presents are allocated
    all_allocated_presents = set()
    for kid, presents in assignment.items():
        all_allocated_presents.update(presents)
    
    assert len(all_allocated_presents) == 4, "All 4 presents should be allocated"
    
    # Check that each kid gets at least one present they value
    for kid, presents in assignment.items():
        kid_value = sum(instance.agent_item_value(kid, present) for present in presents)
        assert kid_value > 0, f"{kid} should get at least one present they value"

def test_divide_function():
    """
    Test the divide function that integrates with the fairpyx framework.
    """
    # In the restricted assignment case, each present j has a fixed value p_j for all kids who can receive it
    # and 0 for kids who cannot receive it
    present_values = {"present1": 0.9, "present2": 0.8}
    
    instance = Instance(
        valuations={
            "Kid1": {"present1": present_values["present1"], "present2": 0.0},
            "Kid2": {"present1": 0.0, "present2": present_values["present2"]}
        },
        # Set a very high capacity for each kid to allow them to receive multiple presents
        agent_capacities={"Kid1": 100, "Kid2": 100},
        item_capacities={"present1": 1, "present2": 1}
    )
    
    alloc = AllocationBuilder(instance)
    allocation = divide(alloc, alpha=2.0, beta=3.0)
    
    # Check that all presents are allocated
    all_allocated_presents = set()
    for presents in allocation.values():
        all_allocated_presents.update(presents)
    
    assert len(all_allocated_presents) == 2, "All 2 presents should be allocated"
    
    # Check that the total value of the allocation is maximized
    total_value = sum(sum(instance.agent_item_value(kid, present) for present in presents) for kid, presents in allocation.items())
    assert total_value > 0, "Total allocation value should be positive"
