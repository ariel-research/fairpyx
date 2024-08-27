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
    num_courses = 50
    students = [f"s{i}" for i in range(1, num_students + 1)]
    courses = [f"c{i}" for i in range(1, num_courses + 1)]
    valuations = {course: students for course in courses}

    instance_large = Instance(agent_capacities={student: 1 for student in students}, item_capacities={course: 10 for course in courses}, valuations=valuations)

    # Run the FaSt algorithm
    allocation_large = FaSt(instance_large)

    # Add assertions
    # Ensure that all students are assigned to a course
    assert len(allocation_large) == len(students), "Not all students are assigned to a course"

    # Ensure that each course has the correct number of students assigned
    for course, assigned_students in allocation_large.items():
        assert len(assigned_students) == 10, f"Incorrect number of students assigned to {course}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
