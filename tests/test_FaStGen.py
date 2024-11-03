
"""
Test the FaStGen algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-Gpt and our friends from the university for ideas of cases.
"""

from fairpyx.allocations import AllocationBuilder
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
    V = {"c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},"c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}}
    U = {"s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},"s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}}                 
    
    # Assuming Instance can handle student and course preferences directly
    ins = Instance(agents=S, items=C, valuations=U)
    alloc = AllocationBuilder(instance=ins)

    # Run the FaStGen algorithm
    result = FaStGen(alloc=alloc, items_valuations=V)

    # Define the expected allocation (this is hypothetical; you should set it based on the actual expected output)
    expected_allocation =  {'c1': ['s1', 's2', 's3'], 'c2': ['s4'], 'c3': ['s5'], 'c4': ['s7', 's6']}

    # Assert the result
    assert result == expected_allocation, "FaStGen algorithm basic case failed"

def test_FaStGen_edge_cases():
    """
    Test edge cases for the FaStGen algorithm.
    """
    # Edge case 1: Empty input
    instance_empty = Instance(agents=[], items=[], valuations={"s1": {"c1":100}})
    alloc = AllocationBuilder(instance=instance_empty)
    with pytest.raises(Exception):
        FaStGen(instance_empty, {})

    # Edge case 2: Single student and single course
    S_single = ["s1"]
    C_single = ["c1"]
    U_single = {"s1": {"c1":100}}
    V_single = {"c1": {"s1":50}}
    instance_single = Instance(agents=S_single, items=C_single, valuations=U_single)
    alloc = AllocationBuilder(instance=instance_single)

    # Run the FaStGen algorithm
    result = FaStGen(alloc=alloc, items_valuations=V_single)
    assert result == {"c1": ["s1"]}, "FaStGen algorithm failed on single student and single course"

if __name__ == "__main__":
    pytest.main(["-v", __file__])