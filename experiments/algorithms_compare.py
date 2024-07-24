
from fairpyx.algorithms.Optimization_Matching import FaSt, FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import experiments_csv
import time
import random

def generate_isometric_data():
    """valuation= {"S1": {"c1": 9, "c2": 8, "c3": 7},
    ... "S2": {"c1": 8, "c2": 7, "c3":6},
    ... "S3": {"c1": 7, "c2": 6, "c3":5},
    ... "S4": {"c1": 6, "c2": 5, "c3":4},
    ... "S5": {"c1": 5, "c2": 4, "c3":3},
    ... "S6": {"c1": 4, "c2": 3, "c3":2},
    ... "S7": {"c1": 3, "c2": 2, "c3":1}}# V[i][j] is the valuation of Si for matching with Cj""" 
    # Generate the number of agents
    num_of_agents = random.randint(100, 400)

    # Generate the number of items, ensuring it is not greater than the number of agents
    num_of_items = random.randint(1, num_of_agents)

    random_integers = random.sample(range(1, 1001), num_of_agents)
    random_integers.sort()
    #agents dict creation > agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
    agents = [f"s{i}" for i in range (1,num_of_agents + 1)]
    #items dict creation > items = {"c1", "c2", "c3"} #College set=C
    items = [f"c{i}" for i in range (1,num_of_items + 1)]

    valuations = {}
    for student in agents:
        valuations_for_items = sorted([random.randint(1, 1000) for _ in items], reverse=True)
        valuations[student] = {college: valuations_for_items[i] for i, college in enumerate(items)}    

    return agents, items, valuations

def generate_regular_data():
    # Generate the number of agents
    num_of_agents = random.randint(100, 400)

    # Generate the number of items, ensuring it is not greater than the number of agents
    num_of_items = random.randint(1, num_of_agents)

    random_integers = random.sample(range(1, 1001), num_of_agents)
    random_integers.sort()
    #agents dict creation > agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
    agents = [f"s{i}" for i in range (1,num_of_agents + 1)]
    #items dict creation > items = {"c1", "c2", "c3"} #College set=C
    items = [f"c{i}" for i in range (1,num_of_items + 1)]

    agents_valuations = {}
    for agent in agents:
        valuations_for_items = sorted([random.randint(1, 1000) for _ in items], reverse=True)
        agents_valuations[agent] = {item: valuations_for_items[i] for i, item in enumerate(items)}    

    items_valuations = {}
    for item in items:
        valuations_for_agents = sorted([random.randint(1, 1000) for _ in agents], reverse=True)
        items_valuations[item] = {agent: valuations_for_agents[i] for i, agent in enumerate(agents)}    
    
    return agents, items, agents_valuations, items_valuations


if __name__ == "main":
    ex = experiments_csv.Experiment("results/", "FaStVsFaStGen.csv")

    agents, items, valuation = generate_isometric_data()
    ins = Instance(agents=agents, items=items, valuations=valuation)
    alloc = AllocationBuilder(instance=ins)

    start_time = time.time() # Measure time for improvements
    FaSt(alloc=alloc)
    end_time = time.time()# Measure time for improvements
    print("Time for FaSt: ", end_time - start_time)

    # ex.clear_previous_results()
    # ex.run(add_three_numbers, input_ranges)
        
    start_time = time.time() # Measure time for improvements
    ins = Instance(agents=agents, items=items, valuations=valuation)
    alloc = AllocationBuilder(instance=ins)
    FaStGen(alloc=alloc, items_valuations=valuation)
    end_time = time.time()# Measure time for improvements
    print("Time for FaStGen: ", end_time - start_time)


    # agents, items, agents_valuations, items_valuation  = generate_isometric_data()

    # start_time = time.time() # Measure time for improvements
    # ins = Instance(agents=agents, items=items, valuations=agents_valuations)
    # alloc = AllocationBuilder(instance=ins)
    # FaStGen(alloc=alloc, items_valuations=items_valuation)
    # end_time = time.time()# Measure time for improvements
    # print("Time for FaStGen over 1st data: ", end_time - start_time)

    # agents, items, items_valuation, agents_valuations = generate_isometric_data()

    # start_time = time.time() # Measure time for improvements
    # ins = Instance(agents=agents, items=items, valuations=agents_valuations)
    # alloc = AllocationBuilder(instance=ins)
    # FaStGen(alloc=alloc, items_valuations=items_valuation)
    # end_time = time.time()# Measure time for improvements
    # print("Time for FaStGen over 2nd data: ", end_time - start_time)