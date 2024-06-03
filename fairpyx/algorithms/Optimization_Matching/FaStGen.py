"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
    Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from FaSt import Demote
import logging
logger = logging.getLogger(__name__)

def FaStGen(alloc: AllocationBuilder, agents_valuations:dict, items_valuations:dict)->dict:
    """
    Algorithem 3-FaSt-Gen: finding a match for the general ranked valuations setting.
    

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param agents_valuations: a dictionary represents how agents valuates the items
    :param items_valuations: a dictionary represents how items valuates the agents
    
    >>> from fairpyx.adaptors import divide
    >>> S = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"}
    >>> C = {"c1", "c2", "c3"}
    >>> V = {
        "c1" : ["s1","s2","s3","s4","s5","s6","s7"], 
        "c2" : ["s2","s4","s1","s3","s5","s6","s7"], 
        "c3" : [s3,s5,s6,s1,s2,s4,s7]}           #the colleges valuations                    
    >>> U = {
        "s1" : ["c1","c3","c2"], 
        "s2" : ["c2","c1","c3"], 
        "s3" : ["c1","c3","c2"], 
        "s4" : ["c3","c2","c1"],
        "s5" : ["c2","c3","c1"], 
        "s6" :["c3","c1","c2"], 
        "s7" : ["c1","c2","c3"]}        #the students valuations                 
    >>> instance = Instance(agents=S, items=C)
    >>> divide(FaStGen, instance=instance, agents_valuations=U, items_valuations=V)
    {'c1' : ['s1','s2'], 'c2' : ['s3','s4','s5']. 'c3' : ['s6','s7']}
    """
    S = alloc.instance.agents
    C = alloc.instance.items
    match = []
    UpperFix = [C[1]]
    LowerFix = [C[len(C)]]
    SoftFix = []
    UnFixed = [item for item in C if item not in UpperFix]

    matching_valuations_sum = {
        colleague: sum(items_valuations[colleague][student] for student in students) 
        for colleague, students in match.items()
        }

    while len(LowerFix) + len([item for item in UpperFix if item not in LowerFix]) < len(C) - 1:
        up = min([j for j in C if j not in LowerFix])
        down = min(valuations_sum for valuations_sum  in matching_valuations_sum.values())
        SoftFix = [pair for pair in SoftFix if not (pair[1] <= up < pair[0])]
        
        if (len(match[up]) == 1) or ():
            LowerFix.append(up)
        else:
            _match = Demote(_match, up, down)
            if calcluate_leximin(_match) >= calcluate_leximin(match):
                match = _match
            elif sourceDec(_match, match) == up:
                LowerFix.append(up)
                UpperFix.append(up + 1)
            elif sourceDec(_match, match) in alloc.instance.agents:
                t = match[sourceDec(_match, match)]
                LowerFix.append(t)
                UpperFix.append(t+1)
                A = [j for j in UnFixed if (j > t + 1)]
                SoftFix.extend((j, t+1) for j in A)
            else:
                match, LowerFix, UpperFix, SoftFix = LookAheadRoutine(match, down, LowerFix, UpperFix, SoftFix)
    UnFixed = [
        j for j in alloc.instance.items 
        if (j not in UpperFix) or 
        any((j, _j) not in SoftFix for _j in alloc.instance.items if _j > j)
        ]

    return match


def LookAheadRoutine(I:tuple, match:dict, down:str, LowerFix:list, UpperFix:list, SoftFix:list)->tuple:
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
    *We will asume that in the colleges list in index 0 there is college 1 in index 1 there is coll
    

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
        "s5" : ["c1","c3","c2","c4"]}        #the students valuations                      
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
    agents, items, agents_valuations, items_valuations = I
    LF = LowerFix.copy()
    UF = UpperFix.copy()
    _match = match.copy()

    while len(LF) + len([item for item in UF if item not in LF]) < len(items) - 1:
        up = min([j for j in items if j not in LowerFix])
        if (len(match[up]) == 1) or ():
            LF.append(up)
        else:
            _match = Demote(_match, len(agents) - 1, up, down)
            if calcluate_leximin(_match) >= calcluate_leximin(match):
                match = _match
                LowerFix = LF
                UpperFix = UF
                break
            elif sourceDec(_match, match) == up:
                LF.append(up)
                UF.append(up + 1)
            elif sourceDec(_match, match) in agents:
                    t = match[sourceDec(_match, match)]
                    if t == down:
                        UpperFix.append(down)
                    else:
                        SoftFix.append((down, t))
                    break
    return (match, LowerFix, UpperFix, SoftFix)

def calcluate_leximin(match:dict)->int:
    return 0

def sourceDec(new_match:dict, old_match:dict):
    """
    Determines the agent causing the leximin decrease between two matchings mu1 and mu2.
    
    Parameters:
    - new_match: First matching (dict of colleges to students)
    - old_match: Second matching (dict of colleges to students)
    
    Returns:
    - The agent (student) causing the leximin decrease.
    """
    for agent in new_match:
        if new_match[agent] != old_match[agent]:
            return agent
    return None

if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())
