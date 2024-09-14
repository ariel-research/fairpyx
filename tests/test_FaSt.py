"""
Test the FaSt algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-Gpt and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import Instance
from fairpyx.algorithms.Optimization_Matching.FaSt import FaSt
from fairpyx import Instance, AllocationBuilder, ExplanationLogger #for FaSt

def test_FaSt_basic_case():
    """
    Basic test case for the FaSt algorithm.
    """
    # Define the instance
    agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"}  # Student set=S
    items = {"c1", "c2", "c3"}  # College set=C
    valuation = {"S1": {"c1": 9, "c2": 8, "c3": 7},
    "S2": {"c1": 8, "c2": 7, "c3": 6},
    "S3": {"c1": 7, "c2": 6, "c3": 5},
    "S4": {"c1": 6, "c2": 5, "c3": 4},
    "S5": {"c1": 5, "c2": 4, "c3": 3},
    "S6": {"c1": 4, "c2": 3, "c3": 2},
    "S7": {"c1": 3, "c2": 2, "c3": 1}}  # V[i][j] is the valuation of Si for matching with Cj
    ins = Instance(agents=agents, items=items, valuations=valuation)
    alloc = AllocationBuilder(instance=ins)
    # Run the FaSt algorithm
    ans= FaSt(alloc=alloc)

    # Define the expected allocation
    expected_allocation = {1: [1, 2], 2: [4, 3], 3: [7, 6, 5]}
    # Assert the result
    assert ans == expected_allocation, "FaSt algorithm basic case failed"

def test_FaSt_edge_cases():
    """
    Test edge cases for the FaSt algorithm.
    """
    # Edge case 2: Single student and single course
    # Define the instance
    agents = {"s1"}  # Student set=S
    items = {"c1"}  # College set=C
    valuation = {"S1": {"c1": 9}}  # V[i][j] is the valuation of Si for matching with Cj
    ins = Instance(agents=agents, items=items, valuations=valuation)
    alloc = AllocationBuilder(instance=ins)
    # Run the FaSt algorithm
    ans = FaSt(alloc=alloc)

    #{"s1": ["c1"]}
    assert ans == {1: [1]}, "FaSt algorithm failed on single student and single course"

def test_FaSt_large_input():
    """
    Test the FaSt algorithm on a large input.
    """
    # Define the instance with a large number of students and courses
    num_students = 100
    num_colleges = 50
    # Define the instance
    agents = {f"s{i}" for i in range(1, num_students + 1)} # Student set=S
    items = {f"c{i}" for i in range(1, num_colleges + 1)} # College set=C
    #valuation = {"S1": {"c1": 9}}  # V[i][j] is the valuation of Si for matching with Cj
    valuation = {}
    value = num_students * num_colleges
    for i in range(1,num_students+1):
        valuation[f"s{i}"] = {}
        for j in range(1,num_colleges+1):
            valuation[f"s{i}"][f"c{j}"] = value
            value -= 1

    instance_large = Instance(agents=agents, items=items, valuations=valuation)
    allocation_large = AllocationBuilder(instance=instance_large)

    # Run the FaSt algorithm
    ans = FaSt(allocation_large)

    # Ensure that all students are assigned to a course
    #assert len(assigned_students) == 10, f"Incorrect number of students assigned to {course}"
    assert len(ans) == 50 # num of student assigned for colleges
    #assert ans == {}
if __name__ == "__main__":
    pytest.main(["-v", __file__])
