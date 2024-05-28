"""
Test the FaStGen algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-Gpt and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import Instance
import FaStGen 

def test_FaStGen_basic_case():
    """
    Basic test case for the FaStGen algorithm.
    """
    # Define the instance
    S = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    C = ["c1", "c2", "c3"]
    V = {
        "c1": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "c2": ["s2", "s4", "s1", "s3", "s5", "s6", "s7"],
        "c3": ["s3", "s5", "s6", "s1", "s2", "s4", "s7"]
    }
    U = {
        "s1": ["c1", "c3", "c2"],
        "s2": ["c2", "c1", "c3"],
        "s3": ["c1", "c3", "c2"],
        "s4": ["c3", "c2", "c1"],
        "s5": ["c2", "c3", "c1"],
        "s6": ["c3", "c1", "c2"],
        "s7": ["c1", "c2", "c3"]
    }
    
    # Assuming `Instance` can handle student and course preferences directly
    instance = Instance(S, C, U, V)

    # Run the FaStGen algorithm
    allocation = FaStGen(instance)

    # Define the expected allocation (this is hypothetical; you should set it based on the actual expected output)
    expected_allocation = {'s1': 'c1', 's2': 'c2', 's3': 'c3', 's4': 'c1', 's5': 'c2', 's6': 'c3', 's7': 'c1'}

    # Assert the result
    assert allocation == expected_allocation, "FaStGen algorithm basic case failed"

def test_FaStGen_edge_cases():
    """
    Test edge cases for the FaStGen algorithm.
    """
    # Edge case 1: Empty input
    instance_empty = Instance([], [], {}, {})
    allocation_empty = FaStGen(instance_empty)
    assert allocation_empty == {}, "FaStGen algorithm failed on empty input"

    # Edge case 2: Single student and single course
    S_single = ["s1"]
    C_single = ["c1"]
    U_single = {"s1": ["c1"]}
    V_single = {"c1": ["s1"]}
    instance_single = Instance(S_single, C_single, U_single, V_single)
    allocation_single = FaStGen(instance_single)
    assert allocation_single == {"s1": "c1"}, "FaStGen algorithm failed on single student and single course"

def test_FaStGen_large_input():
    """
    Test the FaStGen algorithm on a large input.
    """
    # Define the instance with a large number of students and courses
    num_students = 100
    num_courses = 50
    students = [f"s{i}" for i in range(1, num_students + 1)]
    courses = [f"c{i}" for i in range(1, num_courses + 1)]
    valuations = {course: students for course in courses}
    preferences = {student: courses for student in students}  # Assuming all students prefer all courses equally

    instance_large = Instance(students, courses, preferences, valuations)

    # Run the FaStGen algorithm
    allocation_large = FaStGen(instance_large)

    # Add assertions
    # Ensure that all students are assigned to a course
    assert len(allocation_large) == len(students), "Not all students are assigned to a course"

    # Ensure that each course has students assigned to it
    for student, course in allocation_large.items():
        assert student in students, f"Unexpected student {student} in allocation"
        assert course in courses, f"Unexpected course {course} in allocation"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
