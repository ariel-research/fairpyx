"""
Test the Look Ahead Routine algorithm for course allocation.

Programmers: Hadar Bitan, Yuval Ben-Simhon
Date: 19.5.2024
We used chat-GPT and our friends from the university for ideas of cases.
"""

import pytest
from fairpyx import Instance
from fairpyx.algorithms.Optimization_Matching.FaStGen import LookAheadRoutine

def test_look_ahead_routine_basic_case():
    """
    Basic test case for the Look Ahead Routine algorithm.
    """
    # Define the instance
    S = ["s1", "s2", "s3", "s4", "s5"]
    C = ["c1", "c2", "c3", "c4"]
    V = {"c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10},"c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26},"c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28},"c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15}}
    U = {"s1" : {"c1":16,"c2":10,"c3":6,"c4":5},"s2" : {"c1":36,"c2":20,"c3":10,"c4":1},"s3" : {"c1":29,"c2":24,"c3":12,"c4":10},"s4" : {"c1":41,"c2":24,"c3":5,"c4":3},"s5" : {"c1":36,"c2":19,"c3":9,"c4":6}}
    match = {1 : [1,2],2 : [3],3 : [4],4 : [5]}
    I = (S,C,U,V)
    down = 4
    LowerFix = [4]
    UpperFix = [1]
    SoftFix = []

    # Run the Look Ahead Routine algorithm
    new_match, new_LowerFix, new_UpperFix, new_SoftFix = LookAheadRoutine(I, match, down, LowerFix, UpperFix, SoftFix)

    # Define the expected output
    expected_new_match = {"c1": ["s1", "s2"], "c2": ["s3"], "c3" : ["s4"], "c4" : ["s5"]}
    expected_new_LowerFix = [4]
    expected_new_UpperFix = [1, 4]
    expected_new_SoftFix = []

    # Assert the result
    assert new_match == expected_new_match, "Look Ahead Routine algorithm basic case failed"
    assert new_LowerFix == expected_new_LowerFix, "Look Ahead Routine algorithm basic case failed on LowerFix"
    assert new_UpperFix == expected_new_UpperFix, "Look Ahead Routine algorithm basic case failed on UpperFix"
    assert new_SoftFix == expected_new_SoftFix, "Look Ahead Routine algorithm basic case failed on SoftFix"

def test_look_ahead_routine_edge_cases():
    """
    Test edge cases for the Look Ahead Routine algorithm.
    """
    # Edge case 1: Empty input
    I_empty = ([], [], {}, {})
    match_empty = {}
    down_empty = 0
    LowerFix_empty = []
    UpperFix_empty = []
    SoftFix_empty = []

    with pytest.raises(Exception):
        LookAheadRoutine(I_empty, match_empty, down_empty, LowerFix_empty, UpperFix_empty, SoftFix_empty)

    # Edge case 2: Single student and single course
    I_single = ({"s1"}, {"c1"}, {"s1": {"c1":100}}, {"c1": {"s1":80}})
    match_single = {1: [1]}
    down_single = 1
    LowerFix_single = []
    UpperFix_single = []
    SoftFix_single = []

    new_match_single, new_LowerFix_single, new_UpperFix_single, new_SoftFix_single = LookAheadRoutine(I_single, match_single, down_single, LowerFix_single, UpperFix_single, SoftFix_single)
    assert new_match_single == {"c1": ["s1"]}, "Look Ahead Routine algorithm failed on single student and single course"
    assert new_LowerFix_single == [], "Look Ahead Routine algorithm failed on single student and single course (LowerFix)"
    assert new_UpperFix_single == [], "Look Ahead Routine algorithm failed on single student and single course (UpperFix)"
    assert new_SoftFix_single == [], "Look Ahead Routine algorithm failed on single student and single course (SoftFix)"

if __name__ == "__main__":
    pytest.main(["-v", __file__])