"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
    Date: 19.5.2024
"""

import random
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from copy import deepcopy


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
    >>> U = {"s1" : {"c1":16,"c2":10,"c3":6,"c4":5}, "s2" : {"c1":36,"c2":20,"c3":10,"c4":1}, "s3" : {"c1":29,"c2":24,"c3":12,"c4":10}, "s4" : {"c1":41,"c2":24,"c3":5,"c4":3},"s5" : {"c1":36,"c2":19,"c3":9,"c4":6}, "s6" :{"c1":39,"c2":30,"c3":18,"c4":7}, "s7" : {"c1":40,"c2":29,"c3":6,"c4":1}}
    >>> ins = Instance(agents=S, items=C, valuations=U)
    >>> alloc = AllocationBuilder(instance=ins)
    >>> FaStGen(alloc=alloc, items_valuations=V)
    {'c1': ['s1', 's2', 's3'], 'c2': ['s4'], 'c3': ['s5'], 'c4': ['s7', 's6']}
    """
    logger.info("Starting FaStGen algorithm")

    agents = alloc.instance.agents
    items = alloc.instance.items
    agents_valuations = alloc.instance._valuations

    if agents == []:
        raise ValueError(f"Agents list can't be empty")
    if items == []:
        raise ValueError(f"Items list can't be empty")
    if items_valuations == {}:
        raise ValueError(f"Items valuations list can't be empty")

    #Convert the list of the agents and item to dictionary so that each agent\item will have its coresponding integer
    student_name_to_int = generate_dict_from_str_to_int(agents)
    college_name_to_int = generate_dict_from_str_to_int(items)
    int_to_student_name = generate_dict_from_int_to_str(agents)
    int_to_college_name = generate_dict_from_int_to_str(items)

    #Creating a match of the integers and the string coresponding one another to deal with the demote function and the leximin tuple as well
    integer_match = create_stable_matching(agents=agents, items=items, agents_dict=student_name_to_int, items_dict=college_name_to_int)
    str_match = integer_to_str_matching(integer_match=integer_match, agent_dict=student_name_to_int, items_dict=college_name_to_int)

    logger.debug(f"Initial match: {str_match}")

    UpperFix = [college_name_to_int[items[0]]]              # Inidces of colleges to which we cannot add any more students.
    LowerFix = [college_name_to_int[items[len(items)-1]]]       # Inidces of colleges from which we cannot remove any more students.
    SoftFix = []
    UnFixed = [item for item in college_name_to_int.values() if item not in UpperFix]


    #creating a dictionary of vj(µ) = Pi∈µ(cj) for each j in C
    matching_college_valuations = update_matching_valuations_sum(match=str_match, items_valuations=items_valuations)
    logger.debug(f"matching_college_valuations: {matching_college_valuations}")

    while len(LowerFix) + len([item for item in UpperFix if item not in LowerFix]) < len(items):
        logger.debug(f"\nstr_match: {str_match}, integer_match: {integer_match}, UpperFix: {UpperFix}, LowerFix: {LowerFix}, SoftFix: {SoftFix}, UnFixed: {UnFixed}")
        up = min([j for j in college_name_to_int.values() if j not in LowerFix])
        down = min(UnFixed, key=lambda j: matching_college_valuations[int_to_college_name[j]])
        logger.debug(f"up: {up}, down: {down}")

        SoftFix = [pair for pair in SoftFix if not (pair[1] <= up < pair[0])]
        logger.debug(f"Updating SoftFix to {SoftFix}")

        logger.debug(f"vup(mu)={matching_college_valuations[int_to_college_name[up]]}, vdown(mu)={matching_college_valuations[int_to_college_name[down]]}")
        if (len(integer_match[up]) == 1) or (matching_college_valuations[int_to_college_name[up]] <= matching_college_valuations[int_to_college_name[down]]):
            LowerFix.append(up)
            logger.info(f"Cannot remove any more students from c_{up}: Added c_{up} to LowerFix")
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})
            agant_to_demote = get_lowest_ranked_student(down-1, integer_match, items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
            logger.info(f"Demoting from {up} to {down}, starting at student {agant_to_demote}.")
            # new_integer_match = Demote(deepcopy(integer_match), agant_to_demote, up_index=up, down_index=down)
            new_integer_match = Demote(deepcopy(integer_match), agant_to_demote, up_index=up, down_index=down, item_valuations=items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
            logger.info(f"New match: {new_integer_match}")
            new_match_str = integer_to_str_matching(integer_match=new_integer_match, agent_dict=student_name_to_int, items_dict=college_name_to_int)

            #Creating a leximin tuple for the new match from the demote and for the old match to compare
            match_leximin_tuple = create_leximin_tuple(match=str_match  , agents_valuations=agents_valuations, items_valuations=items_valuations)
            logger.info(f"Old match leximin tuple: {match_leximin_tuple}")
            new_match_leximin_tuple = create_leximin_tuple(match=new_match_str, agents_valuations=agents_valuations, items_valuations=items_valuations)
            logger.info(f"New match leximin tuple: {new_match_leximin_tuple}")

            #Extarcting from the SourceDec function the problematic item or agent, if there isn't one then it will be ""
            problematic_component = sourceDec(new_match_leximin_tuple=new_match_leximin_tuple, old_match_leximin_tuple=match_leximin_tuple)
            logger.info(f"   problematic component: {problematic_component}")

            if problematic_component == "":
                logger.debug(f"New match is leximin-better than old match:")
                integer_match = new_integer_match
                str_match = new_match_str
                matching_college_valuations = update_matching_valuations_sum(match=str_match,items_valuations=items_valuations)
                logger.debug(f"   Match updated to {str_match}")
                logger.info(f"    matching_college_valuations: {matching_college_valuations}")

            elif problematic_component == int_to_college_name[up]:
                logger.debug(f"New match is leximin-worse because of c_up = c_{up}:")
                LowerFix.append(up)
                UpperFix.append(up + 1)
                logger.info(f"   Updated LowerFix and UpperFix with {up}")

            elif problematic_component in alloc.instance.agents:
                sd = problematic_component
                logger.debug(f"New match is leximin-worse because of student {sd}: ")
                t = college_name_to_int[get_match(match=str_match, value=sd)]
                LowerFix.append(t)
                UpperFix.append(t+1)
                logger.debug(f"   sourceDec student {sd} is matched to c_t = c_{t}: adding c_{t} to LowerFix and c_{t+1} to UpperFix.")
                A = [j for j in UnFixed if (j > t + 1)]
                SoftFix.extend((j, t+1) for j in A)
                logger.debug(f"   Updating SoftFix to {SoftFix}")

            else:
                logger.debug(f"New match is leximin-worse because of college {sourceDec(new_match_leximin_tuple, match_leximin_tuple)}: ")
                str_match, LowerFix, UpperFix, SoftFix = LookAheadRoutine((agents, items, agents_valuations, items_valuations), integer_match, down, LowerFix, UpperFix, SoftFix)
                logger.debug(f"   LookAheadRoutine result: match={str_match}, LowerFix={LowerFix}, UpperFix={UpperFix}, SoftFix={SoftFix}")

        UnFixed = [
            j for j in college_name_to_int.values()
            if (j not in UpperFix) or
            any((j, _j) not in SoftFix for _j in college_name_to_int.values() if _j > j)
            ]
        logger.debug(f"   Updating UnFixed to {UnFixed}")

    logger.info(f"Finished FaStGen algorithm, final result: {str_match}")
    return str_match    #We want to return the final match in his string form

def LookAheadRoutine(I:tuple, integer_match:dict, down:int, LowerFix:list, UpperFix:list, SoftFix:list)->tuple:
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
    >>> match = {1: [1,2],  2: [3], 3: [4], 4: [5]}
    >>> I = (S,C,U,V)
    >>> down = 4
    >>> LowerFix = [4]
    >>> UpperFix = [1]
    >>> SoftFix = []
    >>> LookAheadRoutine(I, match, down, LowerFix, UpperFix, SoftFix)
    ({'c1': ['s1', 's2'], 'c2': ['s3'], 'c3': ['s4'], 'c4': ['s5']}, [4], [1, 4], [])
    """
    agents, items, agents_valuations, items_valuations = I

    if agents == []:
        raise ValueError(f"Agents list can't be empty")
    if items == []:
        raise ValueError(f"Items list can't be empty")
    if agents_valuations == {}:
        raise ValueError(f"Agents valuations list can't be empty")
    if items_valuations == {}:
        raise ValueError(f"Items valuations list can't be empty")

    student_name_to_int = generate_dict_from_str_to_int(agents)
    college_name_to_int = generate_dict_from_str_to_int(items)
    int_to_student_name = generate_dict_from_int_to_str(agents)
    int_to_college_name = generate_dict_from_int_to_str(items)

    LF = LowerFix.copy()
    UF = UpperFix.copy()
    given_str_match = integer_to_str_matching(integer_match=integer_match, items_dict=college_name_to_int, agent_dict=student_name_to_int)
    new_integer_match = deepcopy(integer_match)
    new_str_match = integer_to_str_matching(integer_match=new_integer_match, items_dict=college_name_to_int, agent_dict=student_name_to_int)

    logger.info(f"Starting LookAheadRoutine. Initial parameters - match: {new_str_match}, down: {down}, LowerFix: {LowerFix}, UpperFix: {UpperFix}, SoftFix: {SoftFix}")
    matching_college_valuations = update_matching_valuations_sum(match=new_str_match,items_valuations=items_valuations)
    while len(LF) + len([item for item in UF if item not in LF]) < len(items):
        up = min([j for j in college_name_to_int.values() if j not in LF])
        logger.debug(f"   Selected 'up': {up}")
        if (len(integer_match[up]) == 1) or (matching_college_valuations[int_to_college_name[up]] <= matching_college_valuations[int_to_college_name[down]]):
            LF.append(up)
            logger.info(f"   Cannot remove any more students from c_{up}: appended c_{up} to LF")
        else:
            #check the lowest-rank student who currently belongs to mu(c_{down-1})d
            agant_to_demote = get_lowest_ranked_student(item_int=down-1, match_int=new_integer_match, items_valuations=items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
            new_integer_match = Demote(deepcopy(integer_match), agant_to_demote, up_index=up, down_index=down, item_valuations=items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
            logger.info(f"   Demoting from {up} to {down}, starting at student {agant_to_demote}. New match: {new_integer_match}")

            new_str_match = integer_to_str_matching(integer_match=new_integer_match, items_dict=college_name_to_int, agent_dict=student_name_to_int)
            matching_college_valuations = update_matching_valuations_sum(match=new_str_match,items_valuations=items_valuations)

            old_match_leximin_tuple = create_leximin_tuple(match=given_str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            new_match_leximin_tuple = create_leximin_tuple(match=new_str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            logger.info(f"   Old match leximin tuple: {old_match_leximin_tuple}")
            new_match_leximin_tuple = create_leximin_tuple(match=new_str_match, agents_valuations=agents_valuations, items_valuations=items_valuations)
            logger.info(f"   New match leximin tuple: {new_match_leximin_tuple}")

                #Extarcting from the SourceDec function the problematic item or agent, if there isn't one then it will be ""
            problematic_component = sourceDec(new_match_leximin_tuple=new_match_leximin_tuple, old_match_leximin_tuple=old_match_leximin_tuple)
            logger.info(f"   problematic component: {problematic_component}")

            if problematic_component == "":
                logger.debug(f"   New match is leximin-better than old match:")
                integer_match = new_integer_match
                LowerFix = LF
                UpperFix = UF
                logger.info("       Updated match and fixed LowerFix and UpperFix")
                break

            elif problematic_component == int_to_college_name[up]:
                logger.debug(f"   New match is leximin-worse because of c_up = c_{up}:")
                LF.append(up)
                UF.append(up + 1)
                logger.info(f"      Appended {up} to LF and {up+1} to UF")

            elif problematic_component in agents:
                sd = problematic_component
                logger.debug(f"   New match is leximin-worse because of student {sd}: ")
                t = college_name_to_int[get_match(match=new_str_match, value=sd)]
                logger.debug(f"      sourceDec student {sd} is matched to c_t = c_{t}.")
                if t == down:
                    logger.debug(f"      t=down={down}: adding c_{down} to UpperFix")  # UF?
                    UpperFix.append(down)
                else:
                    logger.info(f"       t!=down: adding ({down},{t}) to SoftFix")
                    SoftFix.append((down, t))
                break

    final_match_str = integer_to_str_matching(integer_match=integer_match, items_dict=college_name_to_int, agent_dict=student_name_to_int)
    logger.info(f"Completed LookAheadRoutine. Final result - match: {final_match_str}, LowerFix: {LowerFix}, UpperFix: {UpperFix}, SoftFix: {SoftFix}")
    return (final_match_str, LowerFix, UpperFix, SoftFix)

def create_leximin_tuple(match:dict, agents_valuations:dict, items_valuations:dict):
    """
    Create a leximin tuple from the given match, agents' valuations, and items' valuations.

    Args:
    - match (dict): A dictionary where keys are items and values are lists of agents.
    - agents_valuations (dict): A dictionary where keys are agents and values are dictionaries of item valuations.
    - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.

    Returns:
    - list: A sorted list of tuples representing the leximin tuple.

    Example:
    >>> match = {"c1":["s1","s2","s3","s4"], "c2":["s5"], "c3":["s6"], "c4":["s7"]}
    >>> items_valuations = {       #the colleges valuations
    ... "c1":{"s1":50, "s2":23, "s3":21, "s4":13, "s5":10, "s6":6, "s7":5},
    ... "c2":{"s1":45, "s2":40, "s3":32, "s4":29, "s5":26, "s6":11, "s7":4},
    ... "c3":{"s1":90, "s2":79, "s3":60, "s4":35, "s5":28, "s6":20, "s7":15},
    ... "c4":{"s1":80, "s2":48, "s3":36, "s4":29, "s5":15, "s6":7, "s7":1},
    ... }
    >>> agents_valuations = {       #the students valuations
    ... "s1":{"c1":16, "c2":10, "c3":6, "c4":5},
    ... "s2":{"c1":36, "c2":20, "c3":10, "c4":1},
    ... "s3":{"c1":29, "c2":24, "c3":12, "c4":10},
    ... "s4":{"c1":41, "c2":24, "c3":5, "c4":3},
    ... "s5":{"c1":36, "c2":19, "c3":9, "c4":6},
    ... "s6":{"c1":39, "c2":30, "c3":18, "c4":7},
    ... "s7":{"c1":40, "c2":29, "c3":6, "c4":1}
    ... }
    >>> create_leximin_tuple(match, agents_valuations, items_valuations)
    [('c4', 1), ('s7', 1), ('s1', 16), ('s6', 18), ('s5', 19), ('c3', 20), ('c2', 26), ('s3', 29), ('s2', 36), ('s4', 41), ('c1', 107)]

    >>> match = {"c1":["s1","s2","s3"], "c2":["s4"], "c3":["s5"], "c4":["s7","s6"]}
    >>> create_leximin_tuple(match, agents_valuations, items_valuations)
    [('s7', 1), ('s6', 7), ('c4', 8), ('s5', 9), ('s1', 16), ('s4', 24), ('c3', 28), ('c2', 29), ('s3', 29), ('s2', 36), ('c1', 94)]
    """
    leximin_tuple = []
    matching_college_valuations = update_matching_valuations_sum(match=match, items_valuations=items_valuations)
    logger.debug(f"matching_college_valuations: {matching_college_valuations}")
    for item in match.keys():
        if len(match[item]) == 0:
            leximin_tuple.append((item, 0))
        else:
            leximin_tuple.append((item, matching_college_valuations[item]))
        for agent in match[item]:
            leximin_tuple.append((agent,agents_valuations[agent][item]))

    leximin_tuple = sorted(leximin_tuple, key=lambda x: (x[1], x[0]))
    return leximin_tuple

def sourceDec(new_match_leximin_tuple:list, old_match_leximin_tuple:list)->str:
    """
    Determine the agent causing the leximin decrease between two matchings.

    Args:
    - new_match_leximin_tuple (list): The leximin tuple of the new matching.
    - old_match_leximin_tuple (list): The leximin tuple of the old matching.

    Returns:
    - str: The agent (student) causing the leximin decrease.

    Example:
    >>> new_match = [("s7",1),("s4",5),("s5",6),("s6",7),("c4",14),("s1",16),("s3",24),("c2",32),("c3",35),("s2",36),("c1",52)]
    >>> old_match = [("s7",1),("s6",7),("c4",8),("s5",9),("s1",16),("s4",24),("c3",28),("c2",29),("s3",29),("s2",36),("c1",94)]
    >>> sourceDec(new_match, old_match)
    's4'

    >>> new_match = [("s7",1),("s4",7),("s5",8),("s6",9),("c4",14),("s1",16),("s3",24),("c2",32),("c3",35),("s2",36),("c1",52)]
    >>> old_match = [("s7",1),("s6",7),("c4",8),("s5",9),("s1",16),("s4",24),("c3",28),("c2",29),("s3",29),("s2",36),("c1",94)]
    >>> sourceDec(new_match, old_match)
    'c4'
    """
    for k in range(0, len(new_match_leximin_tuple)):
        if new_match_leximin_tuple[k][1] == old_match_leximin_tuple[k][1]:
            continue
        elif new_match_leximin_tuple[k][1] > old_match_leximin_tuple[k][1]:
            return ""
        else:   #new_match_leximin_tuple[k][1] < old_match_leximin_tuple[k][1]
            return new_match_leximin_tuple[k][0]
    return ""

def get_lowest_ranked_student(item_int:int, match_int:dict, items_valuations:dict, int_to_college_name:dict, int_to_student_name:dict):
    """
    Get the lowest ranked student that is matched to the item with the given index.

    # Args:
    # - item: The item for which the lowest ranked student is to be found.
    # - match (dict): A dictionary where keys are items and values are lists of agents.
    # - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.

    # Returns:
    # - str: The lowest ranked student for the given item.

    # Example:
    >>> match = {1:[1,2,3,4], 2:[5], 3:[6], 4:[7]}
    >>> items_valuations = {       #the colleges valuations
    ... "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5},
    ... "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4},
    ... "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    ... "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    ... }
    >>> int_to_college_name = {i: f"c{i}" for i in [1,2,3,4]}
    >>> int_to_student_name = {i: f"s{i}" for i in [1,2,3,4,5,6,7]}
    >>> get_lowest_ranked_student(1, match, items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
    4
    >>> get_lowest_ranked_student(2, match, items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
    5
    >>> get_lowest_ranked_student(3, match, items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
    6
    """
    return min(match_int[item_int], key=lambda agent: items_valuations[int_to_college_name[item_int]][int_to_student_name[agent]])

def update_matching_valuations_sum(match:dict, items_valuations:dict)->dict:
    """
    Update the sum of valuations for each item in the matching.

    Args:
    - match (dict): A dictionary where keys are items and values are lists of agents.
    - items_valuations (dict): A dictionary where keys are items and values are dictionaries of agent valuations.
    - agents (list): List of agents.
    - items (list): List of items.

    Returns:
    - dict: A dictionary with the sum of valuations for each item.

    Example:
    >>> match = {"c1":["s1","s2","s3","s4"], "c2":["s5"], "c3":["s6"], "c4":["s7"]}
    >>> items_valuations = {       #the colleges valuations
    ... "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5},
    ... "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4},
    ... "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    ... "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":7,"s7":1}
    ... }
    >>> update_matching_valuations_sum(match, items_valuations)
    {'c1': 107, 'c2': 26, 'c3': 20, 'c4': 1}

    >>> match = {"c1":["s1","s2","s3"], "c2":["s4"], "c3":["s5"], "c4":["s7","s6"]}
    >>> update_matching_valuations_sum(match, items_valuations)
    {'c1': 94, 'c2': 29, 'c3': 28, 'c4': 8}
    """
    matching_valuations_sum = { #in the artical it looks like this: vj(mu)
        colleague: sum(items_valuations[colleague][student] for student in students)
        for colleague, students in match.items()
        }
    return matching_valuations_sum

def create_stable_matching(agents, agents_dict, items, items_dict):
    """
    Creating a stable matching according to this:
        the first collage get the first n-m+1 students
        each collage in deacrising order get the n-(m-j)th student

    Args:
    - items_dict: A dictionary of all the items and there indexes like this: ("c1":1).
    - agents_dict: A dictionary of all the agents and there indexes like this: ("s1":1).
    - agents (list): List of agents.
    - items (list): List of items.

    Returns:
    - dict: A stable matching of integers

    Example:
    >>> agents = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    >>> items = ["c1", "c2", "c3", "c4"]
    >>> agents_dict = {"s1":1, "s2":2, "s3":3, "s4":4, "s5":5, "s6":6, "s7":7}
    >>> items_dict = {"c1":1, "c2":2, "c3":3, "c4":4}
    >>> create_stable_matching(agents, agents_dict, items, items_dict)
    {1: [1, 2, 3, 4], 2: [5], 3: [6], 4: [7]}
    """
    # Initialize the matching dictionary
    matching = {}

    # Assign the first m-1 students to c1
    matching[items_dict[items[0]]] = [agents_dict[agents[i]] for i in range(0, len(agents) - len(items) + 1)]

    # Assign the remaining students to cj for j >= 2
    for j in range(1, len(items)):
        matching[items_dict[items[j]]] = [agents_dict[agents[len(agents) - (len(items) - j)]]]

    return matching

def generate_dict_from_str_to_int(input_list:list)->dict:
    """
    Creating a dictionary that includes for each string item in the list an index representing it, key=string.

    Args:
    - input_list: A list of strings

    Returns:
    - dict: a dictionary of strings ang indexes

    Example:
    >>> agents = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    >>> items = ["c1", "c2", "c3", "c4"]

    >>> generate_dict_from_str_to_int(agents)
    {'s1': 1, 's2': 2, 's3': 3, 's4': 4, 's5': 5, 's6': 6, 's7': 7}

    >>> generate_dict_from_str_to_int(items)
    {'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4}
    """
    return {item: index + 1 for index, item in enumerate(input_list)}

def generate_dict_from_int_to_str(input_list:list)->dict:
    """
    Creating a dictionary that includes for each string item in the list an index representing it, key=integer.

    Args:
    - input_list: A list of strings

    Returns:
    - dict: a dictionary of strings ang indexes

    Example:
    >>> agents = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    >>> items = ["c1", "c2", "c3", "c4"]

    >>> generate_dict_from_int_to_str(agents)
    {1: 's1', 2: 's2', 3: 's3', 4: 's4', 5: 's5', 6: 's6', 7: 's7'}

    >>> generate_dict_from_int_to_str(items)
    {1: 'c1', 2: 'c2', 3: 'c3', 4: 'c4'}
    """
    return {index + 1: item for index, item in enumerate(input_list)}

def integer_to_str_matching(integer_match:dict, agent_dict:dict, items_dict:dict)->dict:
    """
    Converting an integer match to a string match.

    Args:
    - integer_match: A matching of agents to items out of numbers.
    - items_dict: A dictionary of all the items and there indexes like this: ("c1":1).
    - agents_dict: A dictionary of all the agents and there indexes like this: ("s1":1).

    Returns:
    - dict: A string matching.

    Example:
    >>> agents_dict = {"s1":1, "s2":2, "s3":3, "s4":4, "s5":5, "s6":6, "s7":7}
    >>> items_dict = {"c1":1, "c2":2, "c3":3, "c4":4}
    >>> integer_match = {1: [1, 2, 3, 4], 2: [5], 3: [6], 4: [7]}
    >>> integer_to_str_matching(integer_match, agents_dict, items_dict)
    {'c1': ['s1', 's2', 's3', 's4'], 'c2': ['s5'], 'c3': ['s6'], 'c4': ['s7']}
    """
    # Reverse the s_dict and c_dict to map integer values back to their string keys
    s_reverse_dict = {v: k for k, v in agent_dict.items()}
    c_reverse_dict = {v: k for k, v in items_dict.items()}

    # Create the new dictionary with string keys and lists of string values
    return {c_reverse_dict[c_key]: [s_reverse_dict[s_val] for s_val in s_values] for c_key, s_values in integer_match.items()}

def get_match(match:dict, value:str)->any:
    """
    Giving a match and an agent or an item the function will produce its match

    Args:
    - match: A matching of agents to items.
    - value: An agent or an item.

    Returns:
    - any: An agent or an item.

    Example:
    >>> match = {"c1":["s1","s2","s3","s4"], "c2":["s5"], "c3":["s6"], "c4":["s7"]}

    >>> value = "c1"
    >>> get_match(match, value)
    ['s1', 's2', 's3', 's4']

    >>> value = "s4"
    >>> get_match(match, value)
    'c1'

    >>> value = "c4"
    >>> get_match(match, value)
    ['s7']
    """
    if value in match.keys():
        return match[value]
    else:
        return next((key for key, values_list in match.items() if value in values_list), None)

def Demote(matching:dict, student_index:int, down_index:int, up_index:int, item_valuations:dict, int_to_college_name:dict, int_to_student_name:dict)-> dict:
    """
    Demote algorithm: Adjust the matching by moving a student to a lower-ranked college
    while maintaining the invariant of a complete stable matching.
    The Demote algorithm is a helper function used within the FaSt algorithm to adjust the matching while maintaining stability.

    :param matching: the matchinf of the students with colleges.
    :param student_index: Index of the student to move.
    :param down_index: Index of the college to move the student to.
    :param up_index: Index of the upper bound college.

        #* The test is the same as the running example we gave in Ex2.*
#    ...   valuations       = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}},

    >>> matching = {1: [1, 2, 3, 4], 2: [5], 3: [6], 4: [7]}
    >>> UP = 1
    >>> DOWN = 4
    >>> student_index = 6
    >>> items_valuations = {       #the colleges valuations
    ... "c1" : {"s1":50,"s2":23,"s3":21,"s4":13,"s5":10,"s6":6,"s7":5},
    ... "c2" : {"s1":45,"s2":40,"s3":32,"s4":29,"s5":26,"s6":11,"s7":4},
    ... "c3" : {"s1":90,"s2":79,"s3":60,"s4":35,"s5":28,"s6":20,"s7":15},
    ... "c4" : {"s1":80,"s2":48,"s3":36,"s4":29,"s5":15,"s6":6,"s7":1}
    ... }
    >>> int_to_college_name = {i: f"c{i}" for i in [1,2,3,4]}
    >>> int_to_student_name = {i: f"s{i}" for i in [1,2,3,4,5,6,7]}
    >>> Demote(matching=matching, student_index=student_index, down_index=DOWN, up_index=UP, item_valuations=items_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)
    {1: [1, 2, 3], 2: [4], 3: [5], 4: [7, 6]}
    """
    # Move student to college 'down' while reducing the number of students in 'up'
    # Set t to student_index
    t = student_index
    # Set p to 'down'
    p = down_index
    logger.info(f"matching: {matching}")
    # Check if student 't' is in college 'Cp-1'
    if t not in matching[p - 1]:
        raise ValueError(f"Student {t} should be in matching to college {p - 1}")
    # Check that all colleges have at least one students
    for college, students in matching.items():
        if len(students) < 1:
            raise ValueError(f"All colleges must contain at least 1 student. College number {college} has only {len(students)} students.")

    # While p > up
    while p > up_index:
        # Remove student 't' from college 'cp-1'
        matching[p - 1].remove(t)
        # Add student 't' to college 'cp'
        matching[p].append(t)
        # Decrement t and p
        p -= 1
        if p > 1: t = get_lowest_ranked_student(item_int=p-1, match_int=matching, items_valuations=item_valuations, int_to_college_name=int_to_college_name, int_to_student_name=int_to_student_name)

    return matching #Return the matching after the change

if __name__ == "_main_":
    import doctest, sys
    console=logging.StreamHandler() #writes to stderr (= cerr)
    logger.handlers=[console] # we want the logs to be written to console
    # Change logger level
    logger.setLevel(logging.DEBUG) # Set logger level to DEBUG

    doctest.testmod()
    sys.exit(0)