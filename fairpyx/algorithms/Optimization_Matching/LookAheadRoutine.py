"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
logger = logging.getLogger(__name__)

def LookAheadRoutine(I:tuple, match:dict, down:str, LowerFix:list, UpperFix:list, SoftFix:list)->(dict,list,list,list):
    """
    Algorithem 4-LookAheadRoutine: Designed to handle cases where a decrease in the leximin value
      may be balanced by future changes in the pairing,
      the goal is to ensure that the sumi pairing will maintain a good leximin value or even improve it.
    

    :param I: A presentation of the problem, aka a tuple that contain the list of students(S), the list of colleges(C) when the capacity
    of each college is n-1 where n is the number of students, student valuation function(U), college valuation function(V).
    :param match: The current match of the students and colleges.
    :param down: The lowest ranked unaffixed college
    :param LowerFix: The group of colleges whose lower limit is fixed
    :param UpperFix: The group of colleges whose upper limit is fixed.
    :param SoftFix: A set of temporary upper limits.
    

    >>> from fairpyx.adaptors import divide
    >>> S = {"s1", "s2", "s3", "s4", "s5"}
    >>> C = {"c1", "c2", "c3", "c4"}
    >>> V = {
        "c1" : ["s1","s2","s3","s4","s5"], 
        "c2" : ["s2","s1","s3","s4","s5"], 
        "c3" : ["s3","s2","s1","s4","s5"], 
        "c4" : ["s4","s3","s2","s1","s5"]}           #the colleges valuations                    
    >>> U = {
        "s1" : ["c1","c3","c2","c4"], 
        "s2" : ["c2","c3","c4","c1"], 
        "s3" : ["c3","c4","c1","c2"], 
        "s4" : ["c4","c1","c2","c3"], 
        "s5" : ["c1","c3","c2","c4"]}        #the students valuations                       # 5 seats available
    >>> I = (S, C, U ,V)
    >>> match = {
        "c1" : ["s1","s2"], 
        "c2" : ["s3","s5"], 
        "c3" : ["s4"], 
        "c4" : []}
    >>> down = "c4"
    >>> LowerFix = []
    >>> UpperFix = []
    >>> SoftFix = []
    >>> LookAheadRoutine(I, match, down, LowerFix, UpperFix, SoftFix)
    ({'c1': ['s1', 's2'], 'c2': ['s5'], 'c3' : ['s3'], 'c4' : ['s4']}, ['c2'], [], [])
    """


if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())
