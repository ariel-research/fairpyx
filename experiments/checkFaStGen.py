import inspect
import logging
from fairpyx.algorithms.Optimization_Matching import FaSt, FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import experiments_csv
from matplotlib import pyplot as plt
import random
random.seed(1)

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


agents, items, valuations = generate_isometric_data(num_of_agents=100, num_of_items=25)
items_valuation = {item: {agent: value for agent, items in valuations.items() 
                            for item, value in items.items()} 
                            for item in {item for items in valuations.values() for item in items}}
ins = Instance(agents=agents, items=items, valuations=valuations)
print(valuations)
print()
print(items_valuation)
allocation = AllocationBuilder(instance=ins)
matching = FaStGen.FaStGen(allocation, items_valuations=items_valuation)


