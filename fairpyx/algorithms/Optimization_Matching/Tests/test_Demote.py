"""
Test that Demote algorithm returns a feasible solution.

Programmers: Hadar Bitan, Yuval Ben-Simhon
"""

import pytest
from fairpyx import AllocationBuilder
from your_module import Demote

def test_demote_simple():
    """
    Test the Demote algorithm with a simple scenario.
    """
    # Create an AllocationBuilder instance
    alloc = AllocationBuilder(agent_capacities={"s1": 1, "s2": 1, "s3": 1, "s4": 1, "s5": 1},
                              item_capacities={"c1": 2, "c2": 2, "c3": 2})
    
    # Initial matching
    alloc.add_allocation(0, 0)  # s1 -> c1
    alloc.add_allocation(1, 1)  # s2 -> c2
    alloc.add_allocation(2, 1)  # s3 -> c2
    alloc.add_allocation(3, 2)  # s4 -> c3
    alloc.add_allocation(4, 2)  # s5 -> c3

    # Apply Demote algorithm
    Demote(alloc, 2, 2, 1)

    # Expected result after demotion
    expected_result = {'s1': ['c1'], 's2': ['c2'], 's3': ['c3'], 's4': ['c3'], 's5': ['c3']}
    
    # Check if the result matches the expected result
    assert alloc.get_allocation() == expected_result, "Demote algorithm test failed"

def test_demote_edge_cases():
    """
    Test edge cases for the Demote algorithm.
    """
    # Case 1: Moving the last student to the first college
    alloc1 = AllocationBuilder(agent_capacities={"s1": 1}, item_capacities={"c1": 1})
    alloc1.add_allocation(0, 0)  # s1 -> c1
    Demote(alloc1, 0, 1, 0)
    assert alloc1.get_allocation() == {'s1': ['c1']}, "Demote algorithm edge case 1 failed"

    # Case 2: Moving the first student to the last college
    alloc2 = AllocationBuilder(agent_capacities={"s1": 1}, item_capacities={"c1": 1})
    alloc2.add_allocation(0, 0)  # s1 -> c1
    Demote(alloc2, 0, 2, 0)
    assert alloc2.get_allocation() == {'s1': ['c3']}, "Demote algorithm edge case 2 failed"

    # Case 3: Moving a student to the same college (no change expected)
    alloc3 = AllocationBuilder(agent_capacities={"s1": 1}, item_capacities={"c1": 1})
    alloc3.add_allocation(0, 0)  # s1 -> c1
    Demote(alloc3, 0, 0, 0)
    assert alloc3.get_allocation() == {'s1': ['c1']}, "Demote algorithm edge case 3 failed"

    # Case 4: Moving a student when all other colleges are full
    alloc4 = AllocationBuilder(agent_capacities={"s1": 1}, item_capacities={"c1": 1, "c2": 1})
    alloc4.add_allocation(0, 0)  # s1 -> c1
    alloc4.add_allocation(1, 1)  # s2 -> c2
    Demote(alloc4, 0, 2, 0)
    assert alloc4.get_allocation() == {'s1': ['c1'], 's2': ['c2']}, "Demote algorithm edge case 4 failed"

def test_demote_large_input():
    """
    Test the Demote algorithm on a large input.
    """
    num_students = 1000
    num_colleges = 100
    student_indices = [f"s{i}" for i in range(1, num_students + 1)]
    college_indices = [f"c{i}" for i in range(1, num_colleges + 1)]

    agent_capacities = {student: 1 for student in student_indices}
    item_capacities = {college: 10 for college in college_indices}

    alloc = AllocationBuilder(agent_capacities=agent_capacities, item_capacities=item_capacities)

    # Initial allocation
    for i, student in enumerate(student_indices):
        alloc.add_allocation(i, i % num_colleges)

    # Move the last student to the last college
    Demote(alloc, num_students - 1, num_colleges - 1, 0)

    allocation = alloc.get_allocation()
    assert len(allocation) == num_students, "Demote algorithm large input failed"
    assert all(len(courses) <= 10 for courses in allocation.values()), "Demote algorithm large input capacity failed"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
