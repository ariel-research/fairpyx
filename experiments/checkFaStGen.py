import inspect
import logging
from fairpyx.algorithms.Optimization_Matching import FaSt, FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import experiments_csv
from matplotlib import pyplot as plt
import random
import numpy as np
random.seed(1)

# def create_matrix(n, m):
#     v = np.zeros((n, m), dtype=int)
    
#     #Initialize the n,m squre
#     v[n-1,m-1] = np.random.randint(1, 11)

#     #Settin last column
#     for j in range(n-1, 0, -1):
#         v[j-1, m-1] = v[j, m-1] + np.random.randint(1, 11)

#     #Setting last row
#     for i in range(m-1, 0, -1):
#         v[n-1, i-1] =  v[n-1, i] + np.random.randint(1, 11)

#     #Setting the rest of the matrix
#     for i in range(n-2, -1, -1):
#         for j in range(m-2, -1, -1):
#             v[i, j] = max(v[i+1, j], v[i, j+1]) + np.random.randint(1, 11)

#     return v


# def generate_isometric_data(num_of_agents, num_of_items):
#     #agents dict creation > agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
#     agents = [f"s{i}" for i in range (1,num_of_agents + 1)]
#     #items dict creation > items = {"c1", "c2", "c3"} #College set=C
#     items = [f"c{i}" for i in range (1,num_of_items + 1)]

#     matrix = create_matrix(n=num_of_agents, m=num_of_items)

#     # Extract row-based dictionary (agents to items)
#     valuations = {f"s{i+1}": {f"c{j+1}": int(matrix[i, j]) for j in range(matrix.shape[1])} for i in range(matrix.shape[0])}

#     # Extract column-based dictionary (items to agents)
#     items_valuation = {f"c{j+1}": {f"s{i+1}": int(matrix[i, j]) for i in range(matrix.shape[0])} for j in range(matrix.shape[1])}

#     return agents, items, valuations, items_valuation


# agents, items, valuations, items_valuation = generate_isometric_data(num_of_agents=100, num_of_items=25)
# ins = Instance(agents=agents, items=items, valuations=valuations)
# allocation = AllocationBuilder(instance=ins)
# matching = FaStGen.FaStGen(allocation, items_valuations=items_valuation)

# print(matching)


def generate_isometric_data(num_of_agents, num_of_items):
    #agents dict creation > agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
    agents = [f"s{i}" for i in range (1,num_of_agents + 1)]
    #items dict creation > items = {"c1", "c2", "c3"} #College set=C
    items = [f"c{i}" for i in range (1,num_of_items + 1)]

    valuations = {}
    for student in agents:
        valuations_for_items = sorted([random.randint(1, 1000) for _ in items], reverse=True)
        valuations[student] = {college: valuations_for_items[i] for i, college in enumerate(items)}    

    return agents, items, valuations

# def evaluate_algorithm_output(matching:dict, valuation:dict, agents:list, items:list):
#     """
#     Evaluate the algorithm output according to the following keys:
#     - sum of the items values in the final matching.
#     - sum of the agents values in the final matching.
#     - the minimum value of item.
#     - the maximum value of item.
#     - the minimum value of agent.
#     - the maximum value of agent.
#     """
#     # Calculate sums of item values in the final matching
#     valuation_sums = {item: sum(valuation[agent][item] for agent in agents) for item, agents in matching.items()}
    
#     # Print the valuation sums for debugging
#     print("Valuation Sums:", valuation_sums)
    
#     # Sum of all item values in the final matching
#     sum_item_values = sum(valuation_sums.values())
    
#     # Sum of all agent values in the final matching
#     sum_agent_values = sum(valuation[agent][item] for item, agents in matching.items() for agent in agents)
    
#     # Extract the minimum and maximum values of items
#     min_item = min(valuation_sums.values())
#     max_item = max(valuation_sums.values())
    
#     # Extract the minimum and maximum values of agents
#     min_agent = min(valuation[agent][item] for item, agent in matching.items() for agent in agents)
#     max_agent = max(valuation[agent][item] for item, agent in matching.items() for agent in agents)
    
#     # Print extracted min and max for debugging
#     print("Min Item:", min_item)
#     print("Max Item:", max_item)
    
#     return {
#         "sum_item_values" : sum_item_values,
#         "sum_agent_values" : sum_agent_values,
#         "min_item" : min_item,
#         "max_item" : max_item,
#         "min_agent" : min_agent,
#         "max_agent" : max_agent
#     }

def evaluate_algorithm_output(matching:dict, valuation:dict, agents:list, items:list):
    """
        Evaluate the algorithm output according to the following keys:
        - sum of the items values in the final matching.
        - sum of the agents values in the final matching.
        - the minimum value of item.
        - the maximum value of item.
        - the minimum value of agent.
        - the maximum value of agent.
    """
    valuation_sums = {item: sum(valuation[agent][item] for agent in agents) for item, agents in matching.items()}
    agents = [int(agent[1:]) for agent in agents]
    print(valuation_sums)
    sum_item_values = sum(key for key in valuation_sums.keys())
    sum_agent_values = sum(valuation[agent][item] for item, agents in matching.items() for agent in agents)
    min_item = min(valuation_sums.items(), key=lambda x: x[1])[1]
    max_item = max(valuation_sums.items(), key=lambda x: x[1])[1]
    min_agent = min(valuation[agent][item] for item, agent in matching.items() for agent in agents)
    max_agent = max(valuation[agent][item] for item, agent in matching.items() for agent in agents)
    return {
        "sum_item_values" : sum_item_values,
        "sum_agent_values" : sum_agent_values,
        "min_item" : min_item,
        "max_item" : max_item,
        "min_agent" : min_agent,
        "max_agent" : max_agent
    }

agents, items, valuations = generate_isometric_data(num_of_agents=100, num_of_items=25)
ins = Instance(agents=agents, items=items, valuations=valuations)
valuations_int = {int(s_key[1:]): {int(c_key[1:]): value for c_key, value in inner_dict.items()} for s_key, inner_dict in valuations.items()}
allocation = AllocationBuilder(instance=ins)
matching = FaSt.FaSt(allocation)
print(evaluate_algorithm_output(matching=matching, valuation=valuations_int, agents=agents, items=items))

