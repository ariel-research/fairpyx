"""
Test Allocate course seats using Gale-Shapley pareto-dominant market mechanism.

Programmer: Zachi Ben Shitrit
Since: 2024-05
"""

import pytest
import fairpyx

def test_regular_case():
    
    s1 = {"c1": 40, "c2": 60}
    s2 = {"c1": 70, "c2": 30}
    s3 = {"c1": 70, "c2": 30}
    s4 = {"c1": 40, "c2": 60}
    s5 = {"c1": 50, "c2": 50}
    agent_capacities = {"Alice": 1, "Bob": 1, "Chana": 1, "Dana": 1, "Dor": 1}
    course_capacities = {"c1": 3, "c2": 2}
    valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    course_order_per_student = {"Alice": ["c2", "c1"], "Bob": ["c1", "c2"], "Chana": ["c1", "c2"], "Dana": ["c2", "c1"], "Dor": ["c1", "c2"]}
    tie_braking_lottery = {"Alice": 0.9, "Bob": 0.1, "Chana": 0.2, "Dana": 0.6, "Dor": 0.4}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    assert allocation == {'Alice': ['c2'], 'Bob': ['c1'], 'Chana': ['c1'], 'Dana': ['c2'], 'Dor': ['c1']}, "allocation's did not match"

def test_order_does_not_align_with_bids():
    s1 = {"c1": 20, "c2": 15, "c3": 35, "c4": 10, "c5": 20}
    s2 = {"c1": 30, "c2": 15, "c3": 20, "c4": 20, "c5": 15}
    s3 = {"c1": 40, "c2": 10, "c3": 25, "c4": 10, "c5": 15}
    s4 = {"c1": 10, "c2": 10, "c3": 15, "c4": 30, "c5": 35}
    s5 = {"c1": 25, "c2": 20, "c3": 30, "c4": 10, "c5": 15}
    agent_capacities = {"Alice": 3, "Bob": 3, "Chana": 3, "Dana": 3, "Dor": 3}
    course_capacities = {"c1": 4, "c2": 4, "c3": 2, "c4": 3, "c5": 2}
    valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    course_order_per_student = {"Alice": ["c5", "c3", "c1", "c2", "c4"], "Bob": ["c1", "c4", "c5", "c2", "c3"], "Chana": ["c5", "c1", "c4", "c3", "c2"], "Dana": ["c3", "c4", "c1", "c5", "c2"], "Dor": ["c5", "c1", "c4", "c3", "c2"]}
    tie_braking_lottery = {"Alice": 0.6, "Bob": 0.4, "Chana": 0.3, "Dana": 0.8, "Dor": 0.2}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    assert allocation == {'Alice': ['c1', 'c3', 'c5'], 'Bob': ['c1', 'c2', 'c4'], 'Chana': ['c1', 'c2', 'c4'], 'Dana': ['c2', 'c4', 'c5'], 'Dor': ['c1', 'c2', 'c3']}, "allocation's did not match"

def test_one_agent():
    s1 = {"c1": 40, "c2": 60}
    agent_capacities = {"Alice": 1}
    course_capacities = {"c1": 1, "c2": 1}
    valuations = {"Alice": s1}
    course_order_per_student = {"Alice": ["c2", "c1"]}
    tie_braking_lottery = {"Alice": 0.9}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    assert allocation == {'Alice': ['c2']}, "allocation's did not match"

def test_empty_input():
    agent_capacities = {}
    course_capacities = {}
    valuations = {}
    course_order_per_student = {}
    tie_braking_lottery = {}
    with pytest.raises(StopIteration):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                                instance=instance, 
                                course_order_per_student=course_order_per_student, 
                                tie_braking_lottery=tie_braking_lottery)

def test_wrong_input_type_agent_capacities():
    agent_capacities = "not a dict"
    course_capacities = {"c1": 3, "c2": 2}
    valuations = {"Alice": {"c1": 40, "c2": 60}}
    course_order_per_student = {"Alice": ["c2", "c1"]}
    tie_braking_lottery = {"Alice": 0.9}
    with pytest.raises(TypeError):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                        instance=instance, 
                        course_order_per_student=course_order_per_student, 
                        tie_braking_lottery=tie_braking_lottery)

def test_wrong_input_type_course_capacities():
    agent_capacities = {"Alice": 1}
    course_capacities = "not a dict"
    valuations = {"Alice": {"c1": 40, "c2": 60}}
    course_order_per_student = {"Alice": ["c2", "c1"]}
    tie_braking_lottery = {"Alice": 0.9}
    with pytest.raises(TypeError):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                        instance=instance, 
                        course_order_per_student=course_order_per_student, 
                        tie_braking_lottery=tie_braking_lottery)

def test_wrong_input_type_valuations():
    agent_capacities = {"Alice": 1}
    course_capacities = {"c1": 3, "c2": 2}
    valuations = "not a dict"
    course_order_per_student = {"Alice": ["c2", "c1"]}
    tie_braking_lottery = {"Alice": 0.9}
    with pytest.raises(TypeError):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                        instance=instance, 
                        course_order_per_student=course_order_per_student, 
                        tie_braking_lottery=tie_braking_lottery)

def test_wrong_input_type_course_order_per_student():
    agent_capacities = {"Alice": 1}
    course_capacities = {"c1": 3, "c2": 2}
    valuations = {"Alice": {"c1": 40, "c2": 60}}
    course_order_per_student = "not a dict"
    tie_braking_lottery = {"Alice": 0.9}
    with pytest.raises(TypeError):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                        instance=instance, 
                        course_order_per_student=course_order_per_student, 
                        tie_braking_lottery=tie_braking_lottery)

def test_wrong_input_type_tie_braking_lottery():
    agent_capacities = {"Alice": 1}
    course_capacities = {"c1": 3, "c2": 2}
    valuations = {"Alice": {"c1": 40, "c2": 60}}
    course_order_per_student = {"Alice": ["c2", "c1"]}
    tie_braking_lottery = "not a dict"
    with pytest.raises(TypeError):
        instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                    item_capacities=course_capacities, 
                                    valuations=valuations)
        fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                        instance=instance, 
                        course_order_per_student=course_order_per_student, 
                        tie_braking_lottery=tie_braking_lottery)


def test_large_input():
    num_students = 1000
    num_courses = 50
    agent_capacities = {f"Student_{i}": 1 for i in range(num_students)}
    course_capacities = {f"Course_{i}": num_students // num_courses for i in range(num_courses)}
    valuations = {f"Student_{i}": {f"Course_{j}": (i+j) % 100 for j in range(num_courses)} for i in range(num_students)}
    course_order_per_student = {f"Student_{i}": [f"Course_{j}" for j in range(num_courses)] for i in range(num_students)}
    tie_braking_lottery = {f"Student_{i}": i / num_students for i in range(num_students)}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    # Validate that the allocation is valid
    assert len(allocation) == num_students, "length of allocation did not match the number of students"
    assert all(len(courses) == 1 for courses in allocation.values()), "not every course got exactly 1 student"
    fairpyx.validate_allocation(instance, allocation, title=f"gale_shapley")

def test_large_number_of_students_and_courses():
    s1 = {"c1": 20, "c2": 15, "c3": 35, "c4": 10, "c5": 20, "c6": 30, "c7": 25, "c8": 30, "c9": 15, "c10": 20, "c11": 25, "c12": 10, "c13": 30, "c14": 20, "c15": 15, "c16": 35, "c17": 20, "c18": 10, "c19": 25, "c20": 30}  # sum = 440
    s2 = {"c1": 30, "c2": 15, "c3": 20, "c4": 20, "c5": 15, "c6": 25, "c7": 10, "c8": 20, "c9": 30, "c10": 25, "c11": 20, "c12": 15, "c13": 10, "c14": 20, "c15": 30, "c16": 15, "c17": 25, "c18": 20, "c19": 10, "c20": 35}  # sum = 440
    s3 = {"c1": 40, "c2": 10, "c3": 25, "c4": 10, "c5": 15, "c6": 20, "c7": 25, "c8": 30, "c9": 35, "c10": 20, "c11": 15, "c12": 10, "c13": 20, "c14": 25, "c15": 30, "c16": 15, "c17": 10, "c18": 20, "c19": 25, "c20": 30}  # sum = 440
    s4 = {"c1": 10, "c2": 10, "c3": 15, "c4": 30, "c5": 35, "c6": 20, "c7": 15, "c8": 10, "c9": 25, "c10": 20, "c11": 30, "c12": 15, "c13": 10, "c14": 25, "c15": 20, "c16": 30, "c17": 35, "c18": 20, "c19": 15, "c20": 30}  # sum = 440
    s5 = {"c1": 25, "c2": 20, "c3": 30, "c4": 10, "c5": 15, "c6": 35, "c7": 25, "c8": 20, "c9": 10, "c10": 30, "c11": 15, "c12": 20, "c13": 25, "c14": 10, "c15": 35, "c16": 20, "c17": 15, "c18": 10, "c19": 30, "c20": 25}  # sum = 440
    agent_capacities = {"Alice": 15, "Bob": 20, "Chana": 18, "Dana": 17, "Dor": 16}
    course_capacities = {"c1": 10, "c2": 10, "c3": 8, "c4": 7, "c5": 6, "c6": 5, "c7": 4, "c8": 3, "c9": 2, "c10": 1, "c11": 10, "c12": 9, "c13": 8, "c14": 7, "c15": 6, "c16": 5, "c17": 4, "c18": 3, "c19": 2, "c20": 1}
    valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    course_order_per_student = {"Alice": ["c5", "c3", "c1", "c2", "c4", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"], "Bob": ["c1", "c4", "c5", "c2", "c3", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"], "Chana": ["c5", "c1", "c4", "c3", "c2", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"], "Dana": ["c3", "c4", "c1", "c5", "c2", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"], "Dor": ["c5", "c1", "c4", "c3", "c2", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"]}
    tie_braking_lottery = {"Alice": 0.6, "Bob": 0.4, "Chana": 0.3, "Dana": 0.8, "Dor": 0.2}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    # Validate that the allocation is valid
    fairpyx.validate_allocation(instance, allocation, title=f"gale_shapley")


def test_edge_case_tie():
    s1 = {"c1": 50, "c2": 50}
    s2 = {"c1": 50, "c2": 50}
    agent_capacities = {"Alice": 1, "Bob": 1}
    course_capacities = {"c1": 1, "c2": 1}
    valuations = {"Alice": s1, "Bob": s2}
    course_order_per_student = {"Alice": ["c1", "c2"], "Bob": ["c1", "c2"]}
    tie_braking_lottery = {"Alice": 0.5, "Bob": 0.5}
    instance = fairpyx.Instance(agent_capacities=agent_capacities, 
                                item_capacities=course_capacities, 
                                valuations=valuations)
    allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                            instance=instance, 
                            course_order_per_student=course_order_per_student, 
                            tie_braking_lottery=tie_braking_lottery)
    assert set(allocation.keys()) == {"Alice", "Bob"}, "the keys in the allocation did not match 'Alice', 'Bob'"
    assert set(allocation["Alice"] + allocation["Bob"]) == {"c1", "c2"}, "the total allocation of courses for Alice and Bob did not match 'c1', 'c2'"

if __name__ == "__main__":
     pytest.main(["-v",__file__])