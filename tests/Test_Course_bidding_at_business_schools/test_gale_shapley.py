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
    assert allocation == {'Alice': ['c2'], 'Bob': ['c1'], 'Chana': ['c1'], 'Dana': ['c2'], 'Dor': ['c1']}, "failed"

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
    assert allocation == {'Alice': ['c2']}, "failed"

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
    assert len(allocation) == num_students, "failed"
    assert all(len(courses) == 1 for courses in allocation.values()), "failed"
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
    assert set(allocation.keys()) == {"Alice", "Bob"}, "failed"
    assert set(allocation["Alice"] + allocation["Bob"]) == {"c1", "c2"}, "failed"

if __name__ == "__main__":
     pytest.main(["-v",__file__])