"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

from fairpyx.algorithms.course_match import A_CEEI
from fairpyx.algorithms.course_match import remove_oversubscription
from fairpyx.algorithms.course_match import reduce_undersubscription
import logging
# logging.basicConfig(level=logging.INFO)

def course_match_algorithm(alloc: AllocationBuilder, budget: dict, priorities_student_list: list = [], time : int = 60):
    """
    Perform the Course Match algorithm to find the best course allocations.
    
    :param alloc: (AllocationBuilder) an allocation builder object

    :return: (dict) course allocations

    """
    price_vector = A_CEEI.A_CEEI(alloc,budget,time)
    price_vector = remove_oversubscription.remove_oversubscription(alloc, price_vector, budget)
    reduce_undersubscription.reduce_undersubscription(alloc, price_vector, budget, priorities_student_list)
    return alloc
   
def check_envy(res, instance : Instance):
    alloc = AllocationBuilder(instance)
    my_valuations = {agent: sum([alloc.instance._valuations[agent][item] for item in res[agent]]) for agent in res.keys()}
    # print(my_valuations)
    envy_agent = {agent : {} for agent in res.keys()}
    for agent in res.keys():
        for agent2 in res.keys():
            if agent != agent2:
                agent_val_sum_for_agent2_baskets = sum([alloc.instance._valuations[agent][item] for item in res[agent2]])
                if agent_val_sum_for_agent2_baskets > my_valuations[agent]:
                    envy_agent[agent][agent2] = agent_val_sum_for_agent2_baskets - my_valuations[agent]

    for agent in envy_agent.keys():
        for agent2 in envy_agent[agent].keys():
            agent_val_for_agent2_baskets= [alloc.instance._valuations[agent][item] for item in res[agent2]]
            sum_val = sum(agent_val_for_agent2_baskets)
            boolian_check_envy = False
            for val in agent_val_for_agent2_baskets:
                if  sum_val - val <= my_valuations[agent]:
                    boolian_check_envy=True
                    break
            if not boolian_check_envy:
                raise Exception(f"EF?!?! agent1 {agent} envies agent2 {agent2} by {envy_agent[agent][agent2]}")
    print("check_envy done")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    from fairpyx import divide
    instance = Instance(
      agent_conflicts = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [], "G": [], "H": [], "I": [], "J": []},
      item_conflicts = {"c1": [], "c2": [], "c3": [], "c4": [], "c5": [], "c6": [], "c7": [], "c8": [], "c9": [], "c10": []},
      agent_capacities = {"A": 2, "B": 2, "C": 2, "D": 2, "E": 2, "F": 2, "G": 2, "H": 2, "I": 2, "J": 2},
      item_capacities  = {"c1": 2, "c2": 2, "c3": 2, "c4": 2, "c5": 2, "c6": 2,"c7": 2,"c8": 2,"c9": 2,"c10": 2},
      valuations = {"A": {"c1": 100, "c2": 100, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "B": {"c1": 100, "c2": 100, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "C": {"c1": 0, "c2": 0, "c3": 100, "c4":100, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "D": {"c1": 0, "c2": 0, "c3": 100, "c4":100, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "E": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 100, "c6": 100, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "F": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 100, "c6": 100, "c7": 0, "c8": 0, "c9": 0, "c10": 0},
                    "G": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 100, "c8": 100, "c9": 0, "c10": 0},
                    "H": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 100, "c8": 100, "c9": 0, "c10": 0},
                    "I": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 100, "c10": 100},
                    "J": {"c1": 0, "c2": 0, "c3": 0, "c4":0, "c5": 0, "c6": 0, "c7": 0, "c8": 0, "c9": 100, "c10": 100},             
    })
    budget = {"A": 100.1, "B": 100.2, "C": 100.3, "D": 100.4, "E": 100.5, "F": 100.6, "G": 100.7, "H": 100.8, "I": 100.9, "J": 100.11}    
    
    print(divide(course_match_algorithm, instance, budget=budget))
    # print(res)

    # check_envy(res,instance)