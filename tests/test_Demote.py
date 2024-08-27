"""
Test that Demote algorithm returns a feasible solution.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-GPT and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import AllocationBuilder, Instance
from fairpyx.algorithms.Optimization_Matching.FaSt import Demote
from fairpyx.algorithms.Optimization_Matching.FaStGen import create_stable_matching


def test_demote_simple():
    """
    Test the Demote algorithm with a simple scenario.
    """
    matching = {1: [1, 6], 2: [2, 3], 3: [4, 5]}
    UP = 1
    DOWN = 3
    I = 2
    # Apply Demote algorithm
    ans= Demote(matching, I, DOWN, UP)

    # Expected result after demotion
    expected_result = {1: [6], 2: [3, 1], 3: [4, 5, 2]}

    # Check if the result matches the expected result
    assert ans == expected_result, "Demote algorithm test failed"

def test_demote_edge_cases():
    """
    Test edge cases for the Demote algorithm.
    """
    # Case 1: Moving the last student to the second college
    matching = {1: [1, 6], 2: [2, 3], 3: [4, 5]}
    ans= Demote(matching, 6, 2, 1)
    # Expected result after demotion
    expected_result = {1: [1], 2: [2, 3, 6], 3: [4, 5]}

    # Check if the result matches the expected result
    assert ans == expected_result, "Demote algorithm test failed"

    # Case 2: multiple steps
    # Moving the second student to the last college
    matching = {1: [1, 6], 2: [3 ,2], 3: [4, 5]}
    expected_result = {1: [1, 6], 2: [3], 3: [4, 5, 2]}
    ans = Demote(matching, student_index=2, down_index=3, up_index=2)
    assert ans == expected_result, "Demote algorithm test failed"

    # Case 3: not present student
    # Student not present in the initial college
    matching = {1: [1, 6], 2: [3], 3: [4, 5]}
    try:
        ans = Demote(matching, student_index=2, down_index=2, up_index=1)
    except ValueError as e:
        assert str(e) == "Student 2 should be in matching to college 1", "Demote algorithm test failed"
    else:
        assert False, "Demote algorithm test failed: expected ValueError"

    # Case 4: Demoting when the student is already in the lowest-ranked college
    matching = {1: [1, 6], 2: [3, 2], 3: [4, 5]}
    expected_result = matching.copy()
    ans = Demote(matching, student_index=2, down_index=3, up_index=1)
    assert ans == expected_result, "Demote algorithm test failed"

    # Case 5: Moving a student up instead of down (should raise an error)
    matching = {1: [1, 6], 2: [2, 3], 3: [4, 5]}
    try:
        ans = Demote(matching, student_index=2, down_index=2, up_index=3)
    except ValueError as e:
        # Invalid index values: up_index=3, down_index=2
        assert str(e) == "Student 2 should be in matching to college 1", "Demote algorithm test failed"
    else:
        assert False, "Demote algorithm test failed: expected ValueError"

def test_demote_large_input():
    """
    Test the Demote algorithm on a large input.
    """

    def test_demote_large_input():
        # Create a large matching scenario
        num_students = 1000
        matching = {i: list(range(i, num_students + i, 10)) for i in range(1, 11)}

        # We will demote a student who is initially in college 9
        student_index = 991  # Adjust this to match the pattern in the matching dictionary
        down_index = 10
        up_index = 9

        # Ensure that the student_index exists in the correct college
        assert student_index in matching[up_index], f"Student {student_index} not found in college {up_index}"

        # Expected result: the student should move from college 9 to college 10
        expected_result = matching.copy()
        expected_result[up_index].remove(student_index)
        expected_result[down_index].append(student_index)

        # Run the Demote function
        ans = Demote(matching, student_index, down_index, up_index)

        # Check that the resulting matching is as expected
        assert ans == expected_result, "Demote algorithm large input test failed"

def test_demote_empty_input():
    """
    Test the Demote algorithm with empty input.
    """
    try:
        ans = Demote({}, 0, 0, 0)
    except ValueError as e:
        assert str(e) == "Demote algorithm failed on empty input", "Demote algorithm test failed"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
