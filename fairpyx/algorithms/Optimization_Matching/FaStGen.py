"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
    Date: 19.5.2024
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from FaSt import Demote
#import sys

import logging
logger = logging.getLogger(__name__)

def FaStGen(alloc: AllocationBuilder, items_valuations:dict)->dict:
    """
    Algorithem 3-FaSt-Gen: finding a match for the general ranked valuations setting.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param items_valuations: a dictionary represents how items valuates the agents
    
    >>> from fairpyx.adaptors import divide
    >>> S = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    >>> C = ["c1", "c2", "c3", "c4"]
    >>> V = {"c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},"c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}}
    >>> U = {   "s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},"s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}}
    >>> ins = Instance(agents=S, items=C, valuations=U)
    >>> alloc = AllocationBuilder(instance=ins)
    >>> FaStGen(alloc=alloc, items_valuations=V)
    {'c1': ['s1'], 'c2': ['s2'], 'c3': ['s5', 's4', 's3'], 'c4': ['s7', 's6']}   
    """
    logger.info("Starting FaStGen algorithm")
    
    S = alloc.instance.agents
    C = alloc.instance.items
    agents_valuations = alloc.instance._valuations
    #Convert the list of the agents and item to dictionary so that each agent\item will have its coresponding integer
    S_dict = generate_dictionary(S)
    C_dict = generate_dictionary(C)

    #Creating a match of the integers and the string coresponding one another to deal with the demote function and the leximin tuple as well
    integer_match = create_stable_matching(agents=S, items=C, agents_dict=S_dict, items_dict=C_dict)
    str_match = integer_to_str_matching(integer_match=integer_match, agent_dict=S_dict, items_dict=C_dict)

    logger.debug(f"Initial match: {str_match}")

    UpperFix = [C_dict[C[0]]]
    LowerFix = [C_dict[C[len(C)-1]]]
    SoftFix = []
    UnFixed = [item for item in C_dict.values() if item not in UpperFix]
   

    #creating a dictionary of vj(µ) = Pi∈µ(cj) for each j in C
    matching_valuations_sum = update_matching_valuations_sum(match=str_match,items_valuations=items_valuations)

    while len(LowerFix) + len([item for item in UpperFix if item not in LowerFix]) < len(C):
        up = min([j for j in C_dict.values() if j not in LowerFix])
        down = min(UnFixed, key=lambda j: matching_valuations_sum[get_key_by_value(value=j, items_dict=C_dict)])
        
        SoftFix = [pair for pair in SoftFix if not (pair[1] <= up < pair[0])]
        logger.debug(f"UpperFix: {UpperFix}, LowerFix: {LowerFix}, SoftFix: {SoftFix}, UnFixed: {UnFixed}")

        if (len(integer_match[up]) == 1) or (matching_valuations_sum[get_key_by_value(value=up, items_dict=C_dict)] <= matching_valuations_sum[get_key_by_value(value=down, items_dict=C_dict)]):
            LowerFix.append(up)
            logger.info(f"Added {up} to LowerFix")
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})
            agant_to_demote = get_lowest_ranked_student(down-1, integer_match, items_valuations, C_dict, S_dict)
            _match = Demote(integer_match, agant_to_demote, up_index=up, down_index=down)
            _match_str = integer_to_str_matching(integer_match=_match, agent_dict=S_dict, items_dict=C_dict)

            #Creating a leximin tuple for the new match from the demote and for the old match to compare
            _match_leximin_tuple = create_leximin_tuple(match=_match_str, agents_valuations=agents_valuations, items_valuations=items_valuations)
            match_leximin_tuple = create_leximin_tuple(match=str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            
            if compare_leximin(old_match_leximin_tuple=match_leximin_tuple, new_match_leximin_tuple=_match_leximin_tuple):
                integer_match = _match  
                str_match = integer_to_str_matching(integer_match=integer_match, agent_dict=S_dict, items_dict=C_dict)
                matching_valuations_sum = update_matching_valuations_sum(match=str_match,items_valuations=items_valuations)
                logger.debug(f"Match updated: {str_match}")
            
            elif sourceDec(_match_leximin_tuple, match_leximin_tuple) == up:
                LowerFix.append(up)
                UpperFix.append(up + 1)
                logger.info(f"Updated LowerFix and UpperFix with {up}")
            
            elif sourceDec(_match_leximin_tuple, match_leximin_tuple) in alloc.instance.agents:
                t = C_dict[get_match(match=str_match, value=sourceDec(_match_leximin_tuple, match_leximin_tuple))]
                LowerFix.append(t)
                UpperFix.append(t+1)
                A = [j for j in UnFixed if (j > t + 1)]
                SoftFix.extend((j, t+1) for j in A)
                logger.info(f"Updated LowerFix and UpperFix with {t}")
            
            else:
                str_match, LowerFix, UpperFix, SoftFix = LookAheadRoutine((S, C, agents_valuations, items_valuations), integer_match, down, LowerFix, UpperFix, SoftFix)
                logger.debug(f"LookAheadRoutine result: match={str_match}, LowerFix={LowerFix}, UpperFix={UpperFix}, SoftFix={SoftFix}")
        
        UnFixed = [
            j for j in C_dict.values() 
            if (j not in UpperFix) or 
            any((j, _j) not in SoftFix for _j in C_dict.values() if _j > j)
            ]

    logger.info("Finished FaStGen algorithm")
    return str_match    #We want to return the final march in his string form

def LookAheadRoutine(I:tuple, match:dict, down:int, LowerFix:list, UpperFix:list, SoftFix:list)->tuple:
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
    >>> V = {"c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10},"c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26},"c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28},"c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15}}                               
    >>> U = {"s1" : {"c1":16,"c2":10,"c3":6,"c4":5},"s2" : {"c1":36,"c2":20,"c3":10,"c4":1},"s3" : {"c1":29,"c2":24,"c3":12,"c4":10},"s4" : {"c1":41,"c2":24,"c3":5,"c4":3},"s5" : {"c1":36,"c2":19,"c3":9,"c4":6}}                              
    >>> match = {1 : [1,2],2 : [3,5],3 : [4],4 : []}
    >>> I = (S,C,U,V)
    >>> down = 4
    >>> LowerFix = [1]
    >>> UpperFix = []
    >>> SoftFix = []
    >>> LookAheadRoutine(I, match, down, LowerFix, UpperFix, SoftFix)
    ({'c1': ['s1', 's2'], 'c2': ['s5'], 'c3': ['s3'], 'c4': ['s4']}, [1], [], [])
    """    
    agents, items, agents_valuations, items_valuations = I
    agents_dict = generate_dictionary(agents)
    items_dict = generate_dictionary(items) 
    LF = LowerFix.copy()
    UF = UpperFix.copy()
    _match = match.copy()
    str_match = integer_to_str_matching(integer_match=_match, items_dict=items_dict, agent_dict=agents_dict)
    giving_str_match = integer_to_str_matching(integer_match=match, items_dict=items_dict, agent_dict=agents_dict)

    logger.info("Starting LookAheadRoutine")
    logger.debug(f"Initial parameters - match: {str_match}, down: {down}, LowerFix: {LowerFix}, UpperFix: {UpperFix}, SoftFix: {SoftFix}")
    matching_valuations_sum = update_matching_valuations_sum(match=str_match,items_valuations=items_valuations)
    while len(LF) + len([item for item in UF if item not in LF]) < len(items):
        up = min([j for j in items_dict.values() if j not in LowerFix])
        logger.debug(f"Selected 'up': {up}")
        if (len(_match[up]) == 1) or (matching_valuations_sum[get_key_by_value(value=up, items_dict=items_dict)] <= matching_valuations_sum[get_key_by_value(value=down, items_dict=items_dict)]):
            LF.append(up)
            logger.info(f"Appended {up} to LowerFix")
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})
            agant_to_demote = get_lowest_ranked_student(item=down-1, match=_match, items_valuations=items_valuations, items_dict=items_dict, agent_dict=agents_dict)
            logger.debug(f"Agent to demote: {agant_to_demote}")
            
            _match = Demote(_match, agant_to_demote, up_index=up, down_index=down)
            str_match = integer_to_str_matching(integer_match=_match, items_dict=items_dict, agent_dict=agents_dict)
            matching_valuations_sum = update_matching_valuations_sum(match=str_match,items_valuations=items_valuations)
            
            new_match_leximin_tuple = create_leximin_tuple(match=str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            old_match_leximin_tuple = create_leximin_tuple(match=giving_str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            if compare_leximin(old_match_leximin_tuple=old_match_leximin_tuple, new_match_leximin_tuple=new_match_leximin_tuple):
                match = _match
                LowerFix = LF
                UpperFix = UF
                logger.info("Updated match and fixed LowerFix and UpperFix")
                break
            elif sourceDec(new_match_leximin_tuple=new_match_leximin_tuple, old_match_leximin_tuple=old_match_leximin_tuple) == up:
                LF.append(up)
                UF.append(up + 1)
                logger.info(f"Appended {up} to LowerFix and {up+1} to UpperFix")
            elif sourceDec(new_match_leximin_tuple=new_match_leximin_tuple, old_match_leximin_tuple=old_match_leximin_tuple) in agents:
                    t = items_dict[get_match(match=str_match, value=sourceDec(new_match_leximin_tuple=new_match_leximin_tuple, old_match_leximin_tuple=old_match_leximin_tuple))]
                    if t == down:
                        UpperFix.append(down)
                    else:
                        SoftFix.append((down, t))
                        logger.info(f"Appended {down} to UpperFix or SoftFix")
                    break

    final_match = integer_to_str_matching(integer_match=match, items_dict=items_dict, agent_dict=agents_dict)
    logger.info("Completed LookAheadRoutine")
    logger.debug(f"Final result - match: {final_match}, LowerFix: {LowerFix}, UpperFix: {UpperFix}, SoftFix: {SoftFix}")
    return (final_match, LowerFix, UpperFix, SoftFix)

def create_leximin_tuple(match:dict, agents_valuations:dict, items_valuations:dict):
    # """
    # Create a leximin tuple from the given match, agents' valuations, and items' valuations.

    # Args:
    # - match (dict): A dictionary where keys are items and values are lists of agents.
    # - agents_valuations (dict): A dictionary where keys are agents and values are dictionaries of item valuations.
    # - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.

    # Returns:
    # - list: A sorted list of tuples representing the leximin tuple.

    # Example:
    # >>> match = {"c1":["s1","s2","s3"], "c2":["s4"], "c3":["s5"], "c4":["s7","s6"]}
    # >>> items_valuations = {       #the colleges valuations
    #     "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, 
    #     "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, 
    #     "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    #     "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    #     }
    # >>> agents_valuations = {       #the students valuations   
    #     "s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, 
    #     "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, 
    #     "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, 
    #     "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},
    #     "s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, 
    #     "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, 
    #     "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}
    #     }
    # >>> create_leximin_tuple(match, agents_valuations, items_valuations)
    # [("s7",1),("c4",1),("s6",6),("c4",7),("c3",9),("c1",16),("s3",21),("s2",23),("c2",24),("s4",29),("c1",29),("c1",36),("s1",50)]
    # """ 
    leximin_tuple = []
    for item in match.keys():
        if len(match[item]) == 0:
            leximin_tuple.append((item, 0))
        for agent in match[item]:
            leximin_tuple.append((agent,items_valuations[item][agent]))
            leximin_tuple.append((item, agents_valuations[agent][item]))
    leximin_tuple.sort(key = lambda x: x[1]) 
    return leximin_tuple

def compare_leximin(new_match_leximin_tuple:list, old_match_leximin_tuple:list)->bool:
    # """
    # Determine whether the leximin tuple of the new match is greater or equal to the leximin tuple of the old match.

    # Args:
    # - new_match_leximin_tuple (list): The leximin tuple of the new matching.
    # - old_match_leximin_tuple (list): The leximin tuple of the old matching.

    # Returns:
    # - bool: True if new_match_leximin_tuple >= old_match_leximin_tuple, otherwise False.

    # Example:
    # >>> new_match = [("s7",1),("c4",1),("s6",6),("c4",7),("c3",9),("c1",16),("s3",21),("s2",23),("c2",24),("s4",29),("c1",29),("c1",36),("s1",50)]
    # >>> old_match = [("s7",1),("c4",1),("s4",13),("c1",16),("c3",18),("c2",19),("s6",20),("s3",21),("s2",23),("s5",26),("c1",29),("c1",36),("c1",41),("s1",50)]
    # >>> compare_leximin(new_match, old_match)
    # False

    # >>> new_match = [("c4",0),("c3",5),("c1",16),("c2",19),("s2",23),("c2",24),("s5",26),("s3",32),("s4",35),("c1",36),("s1",50)]
    # >>> old_match = [("c4",3),("c3",12),("c1",16),("c2",19),("s2",23),("s5",26),("s4",29),("c1",36),("s1",50),("s3",60)]
    # >>> compare_leximin(new_match, old_match)
    # True
    # """
    for k in range(0, len(new_match_leximin_tuple)):
        if new_match_leximin_tuple[k][1] == old_match_leximin_tuple[k][1]:
            continue
        elif new_match_leximin_tuple[k][1] > old_match_leximin_tuple[k][1]:
            return True
        else:
            return False

def sourceDec(new_match_leximin_tuple:list, old_match_leximin_tuple:list)->str:
    # """
    # Determine the agent causing the leximin decrease between two matchings.

    # Args:
    # - new_match_leximin_tuple (list): The leximin tuple of the new matching.
    # - old_match_leximin_tuple (list): The leximin tuple of the old matching.

    # Returns:
    # - str: The agent (student) causing the leximin decrease.

    # Example:
    # >>> new_match = [("s7",1),("c4",1),("s6",6),("c4",7),("c3",9),("c1",16),("s3",21),("s2",23),("c2",24),("s4",29),("c1",29),("c1",36),("s1",50)]
    # >>> old_match = [("s7",1),("c4",1),("s4",13),("c1",16),("c3",18),("c2",19),("s6",20),("s3",21),("s2",23),("s5",26),("c1",29),("c1",36),("c1",41),("s1",50)]
    # >>> sourceDec(new_match, old_match)
    # 's6'

    # >>> new_match = [("c4",3),("c3",5),("c1",16),("c2",19),("s2",23),("c2",24),("s5",26),("s3",32),("s4",35),("c1",36),("s1",50)]
    # >>> old_match = [("c4",3),("c3",12),("c1",16),("c2",19),("s2",23),("s5",26),("s4",29),("c1",36),("s1",50),("s3",60)]
    # >>> sourceDec(new_match, old_match)
    # 'c3'
    # """
    for k in range(0, len(new_match_leximin_tuple)):
        if new_match_leximin_tuple[k][1] < old_match_leximin_tuple[k][1]:
            return new_match_leximin_tuple[k][0]  
    return ""

def get_lowest_ranked_student(item, match:dict, items_valuations:dict, items_dict:dict, agent_dict:dict):
    # """
    # Get the lowest ranked student for a given item.

    # Args:
    # - item: The item for which the lowest ranked student is to be found.
    # - match (dict): A dictionary where keys are items and values are lists of agents.
    # - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.

    # Returns:
    # - str: The lowest ranked student for the given item.

    # Example:
    # >>> match = {"c1":["s1","s2","s3","s4"], "c2":["s5"], "c3":["s6"], "c4":["s7"]}
    # >>> items_valuations = {       #the colleges valuations
    #     "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, 
    #     "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, 
    #     "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    #     "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    #     }
    # >>> get_lowest_ranked_student("c3", match, items_valuations)
    # 's6'
    # """
    return min(match[item], key=lambda agant: items_valuations[get_key_by_value(value=item, items_dict=items_dict)][get_key_by_value(value=agant, items_dict=agent_dict)])

def update_matching_valuations_sum(match:dict, items_valuations:dict)->dict:
    # """
    # Update the sum of valuations for each item in the matching.

    # Args:
    # - match (dict): A dictionary where keys are items and values are lists of agents.
    # - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.
    # - agents (list): List of agents.
    # - items (list): List of items.

    # Returns:
    # - dict: A dictionary with the sum of valuations for each item.

    # Example:
    # >>> match = {c1:[s1,s2,s3,s4], c2:[s5], c3:[s6], c4:[s7]}
    # >>> items_valuations = {       #the colleges valuations
    #     "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5}, 
    #     "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4}, 
    #     "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    #     "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    #     }
    # >>> agents = ["s1","s2","s3","s4","s5","s6","s7"]
    # >>> items = ["c1","c2","c3","c4"]
    # >>> update_matching_valuations_sum(match, items_valuations, agents, items)
    # {"c1": 107, "c2": 26, "c3": 20, "c4": 1}
    # """
    matching_valuations_sum = { #in the artical it looks like this: vj(mu)
        colleague: sum(items_valuations[colleague][student] for student in students) 
        for colleague, students in match.items()
        }
    return matching_valuations_sum

def create_stable_matching(agents, agents_dict, items, items_dict):
    # Initialize the matching dictionary
    matching = {}

    # Assign the first m-1 students to c1
    matching[items_dict[items[0]]] = [agents_dict[agents[i]] for i in range(0, len(agents) - len(items) + 1)]

    # Assign the remaining students to cj for j >= 2
    for j in range(1, len(items)):
        matching[items_dict[items[j]]] = [agents_dict[agents[len(agents) - (len(items) - j)]]]
    
    return matching

def generate_dictionary(input_list:list)->dict:
    return {item: index + 1 for index, item in enumerate(input_list)}
    
def get_key_by_value(value, items_dict):
    return next(key for key, val in items_dict.items() if val == value)

def integer_to_str_matching(integer_match:dict, agent_dict:dict, items_dict:dict)->dict:
    # Reverse the s_dict and c_dict to map integer values back to their string keys
    s_reverse_dict = {v: k for k, v in agent_dict.items()}
    c_reverse_dict = {v: k for k, v in items_dict.items()}
    
    # Create the new dictionary with string keys and lists of string values
    return {c_reverse_dict[c_key]: [s_reverse_dict[s_val] for s_val in s_values] for c_key, s_values in integer_match.items()}

def get_match(match:dict, value:str)->any:
    if value in match.keys():
        return match[value]
    else:
        return next((key for key, values_list in match.items() if value in values_list), None)

if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())