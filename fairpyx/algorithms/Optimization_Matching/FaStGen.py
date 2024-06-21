"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
    Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from FaSt import Demote
import logging
import sys
logger = logging.getLogger(__name__)

def FaStGen(alloc: AllocationBuilder, agents_valuations:dict, items_valuations:dict)->dict:
    """
    Algorithem 3-FaSt-Gen: finding a match for the general ranked valuations setting.
    

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param agents_valuations: a dictionary represents how agents valuates the items
    :param items_valuations: a dictionary represents how items valuates the agents
    
    >>> from fairpyx.adaptors import divide
    >>> S = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    >>> C = ["c1", "c2", "c3", "c4"]
    >>> V = {       #the colleges valuations
        "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, 
        "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, 
        "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
        "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
        }
    >>> U = {       #the students valuations   
        "s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, 
        "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, 
        "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, 
        "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},
        "s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, 
        "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, 
        "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}
        }
    >>> instance = Instance(agents=S, items=C)
    >>> divide(FaStGen, instance=instance, agents_valuations=U, items_valuations=V)
    {"c1" : ["s1","s2","s3","s4"], "c2" : ["s5"], "c3" : ["s6"], "c4" : ["s7"]}
    """
    S = alloc.instance.agents
    C = alloc.instance.items
    match = []
    UpperFix = [C[1]]
    LowerFix = [C[len(C)]]
    SoftFix = []
    UnFixed = [item for item in C if item not in UpperFix]

    matching_valuations_sum = update_matching_valuations_sum(match=match,items_valuations=items_valuations, agents=S, items=C)

    while len(LowerFix) + len([item for item in UpperFix if item not in LowerFix]) < len(C):
        up = min([j for j in C if j not in LowerFix])
        down = min(valuations_sum for key, valuations_sum in matching_valuations_sum.items() if key in UnFixed)
        SoftFix = [pair for pair in SoftFix if not (pair[1] <= up < pair[0])]
        
        if (len(match[up]) == 1) or (matching_valuations_sum[up] <= matching_valuations_sum[down]):
            LowerFix.append(up)
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})
            agant_to_demote = get_lowest_ranked_student(down-1, match, items_valuations)
            _match = Demote(_match, agant_to_demote, up, down)
            _match_leximin_tuple = create_leximin_tuple(match=_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            match_leximin_tuple = create_leximin_tuple(match=match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            if compare_leximin(match_leximin_tuple, _match_leximin_tuple):
                match = _match  
                matching_valuations_sum = update_matching_valuations_sum(match=match,items_valuations=items_valuations, agents=S, items=C)
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
                match, LowerFix, UpperFix, SoftFix = LookAheadRoutine((S, C, agents_valuations, items_valuations), match, down, LowerFix, UpperFix, SoftFix)
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
    >>> S = ["s1", "s2", "s3", "s4", "s5"]
    >>> C = ["c1", "c2", "c3", "c4"]
    >>> V = {       #the colleges valuations
        "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10}, 
        "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26}, 
        "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28}, 
        "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15}}                               
    >>> U = {       #the students valuations
        "s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, 
        "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, 
        "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, 
        "s4" : {"c1":41,"c2":24,"c3":5,"c4":3}, 
        "s5" : {"c1":36,"c2":19,"c3":9,"c4":6}}                              
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
    ({"c1": ["s1", "s2"], "c2": ["s5"], "c3" : ["s3"], "c4" : ["s4"]}, ["c1"], [], [])
    """
    agents, items, agents_valuations, items_valuations = I
    LF = LowerFix.copy()
    UF = UpperFix.copy()
    _match = match.copy()

    matching_valuations_sum = update_matching_valuations_sum(match=_match,items_valuations=items_valuations, agents=agents, items=items)

    while len(LF) + len([item for item in UF if item not in LF]) < len(items) - 1:
        up = min([j for j in items if j not in LowerFix])
        if (len(match[up]) == 1) or (matching_valuations_sum[up] <= matching_valuations_sum[down]):
            LF.append(up)
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})
            agant_to_demote = get_lowest_ranked_student(down-1, match)
            _match = Demote(_match, agant_to_demote, up, down)
            matching_valuations_sum = update_matching_valuations_sum(match=_match,items_valuations=items_valuations, agents=agents, items=items)
            _match_leximin_tuple = create_leximin_tuple(match=_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            match_leximin_tuple = create_leximin_tuple(match=match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            if compare_leximin(match_leximin_tuple, _match_leximin_tuple):
                match = _match
                LowerFix = LF
                UpperFix = UF
                break
            elif sourceDec(_match, match) == up:
                LF.append(up)
                UF.append(up + 1)
            elif sourceDec(_match, match) in agents:
                    t = _match[sourceDec(_match, match)]
                    if t == down:
                        UpperFix.append(down)
                    else:
                        SoftFix.append((down, t))
                    break
    return (match, LowerFix, UpperFix, SoftFix)

def create_leximin_tuple(match:dict, agents_valuations:dict, items_valuations:dict):
    leximin_tuple = []
    for item in match.keys():
        for agent in match[item]:
            leximin_tuple.append((agent,items_valuations[item][agent]))
            leximin_tuple.append((item, agents_valuations[agent][item]))
    leximin_tuple.sort(key = lambda x: x[1]) 
    return leximin_tuple

def compare_leximin(new_match_leximin_tuple:list, old_match_leximin_tuple:list)->bool:
    """
    Determain whethere the leximin tuple of the new match is grater or equal to the leximin tuple of the old match
    
    Parameters:
    - new_match_leximin_tuple: the leximin tuple of the new matching
    - old_match_leximin_tuple: the leximin tuple of the old matching

    Return:
    - true or false if new_match_leximin_tuple >= old_match_leximin_tuple
    """
    for k in range(0, len(new_match_leximin_tuple)):
        if new_match_leximin_tuple[k][1] == old_match_leximin_tuple[k][1]:
            continue
        elif new_match_leximin_tuple[k][1] > old_match_leximin_tuple[k][1]:
            return True
        else:
            return False

def sourceDec(new_match_leximin_tuple:list, old_match_leximin_tuple:list)->str:
    """
    Determines the agent causing the leximin decrease between two matchings mu1 and mu2.
    
    Parameters:
    - new_match_leximin_tuple: the leximin tuple of the new matching
    - old_match_leximin_tuple: the leximin tuple of the old matching
    
    Returns:
    - The agent (student) causing the leximin decrease.
    """
    for k in range(0, len(new_match_leximin_tuple)):
        if new_match_leximin_tuple[k][1] < old_match_leximin_tuple[k][1]:
            return new_match_leximin_tuple[k][0]  
    return ""

def get_lowest_ranked_student(item:int, match:dict, items_valuations:dict)->str:
    # min = sys.maxsize
    # lowest_ranked_student = 0
    # for agant in match[item]:
    #     minTemp = items_valuations[item][agant]
    #     if minTemp < min:
    #         min = minTemp
    #         lowest_ranked_student = agant
    # return lowest_ranked_student
    return min(match[item], key=lambda agant: items_valuations[item][agant])

def update_matching_valuations_sum(match:dict, items_valuations:dict, agents:list, items:list)->dict:
    matching_valuations_sum = { #in the artical it looks like this: vj(mu)
        colleague: sum(items_valuations[colleague][student] for student in students) 
        for colleague, students in match.items()
        }
    return matching_valuations_sum

if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())
