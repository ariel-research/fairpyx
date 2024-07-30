
import inspect
import logging
from fairpyx.algorithms.Optimization_Matching import FaSt, FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import experiments_csv
from matplotlib import pyplot as plt
import random

TIME_LIMIT = 100

algorithms_to_check = [
    FaSt,
    FaStGen
    ]



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

def generate_regular_data(num_of_agents, num_of_items):
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

def evaluate_algorithm_output(matching, valuation):

    return {}


def run_algorithm_with_random_instance_uniform(num_of_agents, num_of_items, algorithm):
    matchingFaSt, matchingFaStGen = {}
    agents, items, valuations = generate_isometric_data(num_of_agents=num_of_agents, num_of_items=num_of_items)
    allocation = AllocationBuilder(agents, items, valuations)
    if inspect.getsource(algorithm) == inspect.getsource(FaSt):
        matchingFaSt = FaSt(allocation)
    if inspect.getsource(algorithm) == inspect.getsource(FaStGen):
        matchingFaStGen = FaSt(allocation, valuations)
    return {
        "FaSt" : evaluate_algorithm_output(matching=matchingFaSt, valuation=valuations),
        "FaStGen" : evaluate_algorithm_output(matching=matchingFaStGen, valuation=valuations)
    }

def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "FaStVsFaStGen.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [100,200,300],         
        "num_of_items":  [25],                   
        "algorithm": algorithms_to_check,
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(run_algorithm_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)

if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    run_uniform_experiment()