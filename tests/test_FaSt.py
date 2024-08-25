"""
Test the FaSt algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-Gpt and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import Instance
from fairpyx.algorithms.Optimization_Matching.FaSt import FaSt

def test_FaSt_basic_case():
    """
    Basic test case for the FaSt algorithm.
    """
    # Define the instance
    S = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    C = ["c1", "c2", "c3"]
    V = {
        "c1": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "c2": ["s2", "s4", "s1", "s3", "s5", "s6", "s7"],
        "c3": ["s3", "s5", "s6", "s1", "s2", "s4", "s7"]
    }
    instance = Instance(agent_capacities={student: 1 for student in S}, item_capacities={course: 1 for course in C}, valuations=V)

    # Run the FaSt algorithm
    allocation = FaSt(instance)

    # Define the expected allocation
    expected_allocation = {'s1': ['c2'], 's2': ['c1'], 's3': ['c3'], 's4': ['c3'], 's5': ['c3'], 's6': ['c1'], 's7': ['c2']}

    # Assert the result
    assert allocation == expected_allocation, "FaSt algorithm basic case failed"

def test_FaSt_edge_cases():
    """
    Test edge cases for the FaSt algorithm.
    """
    # Edge case 1: Empty input
    instance_empty = Instance(agent_capacities={}, item_capacities={}, valuations={})
    allocation_empty = FaSt(instance_empty)
    assert allocation_empty == {}, "FaSt algorithm failed on empty input"

    # Edge case 2: Single student and single course
    instance_single = Instance(agent_capacities={"s1": 1}, item_capacities={"c1": 1}, valuations={"c1": ["s1"]})
    allocation_single = FaSt(instance_single)
    assert allocation_single == {"s1": ["c1"]}, "FaSt algorithm failed on single student and single course"

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
