"""
Test the FaStGen algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-Gpt and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import Instance
from fairpyx.algorithms.Optimization_Matching.FaStGen import FaStGen

def test_FaStGen_basic_case():
    """
    Basic test case for the FaStGen algorithm.
    """
    # Define the instance
    S = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    C = ["c1", "c2", "c3", "c4"]
    V = {       #the colleges valuations
        "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, 
        "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, 
        "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
        "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    }                               
    U = {       #the students valuations   
        "s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, 
        "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, 
        "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, 
        "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},
        "s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, 
        "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, 
        "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}
    }     
                          
    
    # Assuming `Instance` can handle student and course preferences directly
    instance = Instance(agents=S, items=C)

    # Run the FaStGen algorithm
    allocation = FaStGen(instance, agents_valuations=U, items_valuations=V)

    # Define the expected allocation (this is hypothetical; you should set it based on the actual expected output)
    expected_allocation = {"c1" : ["s1","s2","s3"], "c2" : ["s4"], "c3" : ["s5"], "c4" : ["s7", "s6"]}

    # Assert the result
    assert allocation == expected_allocation, "FaStGen algorithm basic case failed"

def test_FaStGen_edge_cases():
    """
    Test edge cases for the FaStGen algorithm.
    """
    # Edge case 1: Empty input
    instance_empty = Instance([], [])
    allocation_empty = FaStGen(instance_empty, {}, {})
    assert allocation_empty == {}, "FaStGen algorithm failed on empty input"

    # Edge case 2: Single student and single course
    S_single = ["s1"]
    C_single = ["c1"]
    U_single = {"s1": {"c1":100}}
    V_single = {"c1": {"s1":50}}
    instance_single = Instance(S_single, C_single)
    allocation_single = FaStGen(instance_single, U_single, V_single)
    assert allocation_single == {"c1": ["s1"]}, "FaStGen algorithm failed on single student and single course"

def test_FaStGen_large_input():
    """
    Test the FaStGen algorithm on a large input.
    """
    # Define the instance with a large number of students and courses
    num_students = 100
    num_colleges = 50
    students = [f"s{i}" for i in range(1, num_students + 1)]
    colleges = [f"c{i}" for i in range(1, num_colleges + 1)]
    colleges_valuations = {college: students for college in colleges}
    students_valuations = {student: college for student in students}  # Assuming all students prefer all courses equally

    instance_large = Instance(students, colleges)

    # Run the FaStGen algorithm
    allocation_large = FaStGen(instance_large, agents_valuations=students_valuations, items_valuations=colleges_valuations)

    # Add assertions
    # Ensure that all students are assigned to a course
    assert len(allocation_large) == len(students), "Not all students are assigned to a course"

    # Ensure that each course has students assigned to it
    for student, college in allocation_large.items():
        assert student in students, f"Unexpected student {student} in allocation"
        assert college in colleges, f"Unexpected course {college} in allocation"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
