
import csv
import inspect
import logging
from fairpyx.algorithms.Optimization_Matching import FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from experiments_csv import single_plot_results, multi_plot_results
import experiments_csv
from matplotlib import pyplot as plt
import random

TIME_LIMIT = 100

algorithms_to_check = [
    FaStGen
    ]



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

def evaluate_algorithm_output(matching:dict, agentValuations:dict, itemValuations:dict, agents:list, items:list):
    """
        Evaluate the algorithm output according to the following keys:
        - sum of the items values in the final matching.
        - sum of the agents values in the final matching.
        - the minimum value of item.
        - the maximum value of item.
        - the minimum value of agent.
        - the maximum value of agent.
    """
    matching_college_valuations = FaStGen.update_matching_valuations_sum(match=matching, items_valuations=itemValuations)
    sum_item_values = sum(value for value in matching_college_valuations.values())
    sum_agent_values = sum(agentValuations[agent][item] for item, agent in matching.items() for agent in agents)
    min_item = min(matching_college_valuations.items(), key=lambda x: x[1])[1]
    max_item = max(matching_college_valuations.items(), key=lambda x: x[1])[1]
    min_agent = min(agentValuations[agent][item] for item, agent in matching.items() for agent in agents)
    max_agent = max(agentValuations[agent][item] for item, agent in matching.items() for agent in agents)
    return {
        "sum_item_values" : sum_item_values,
        "sum_agent_values" : sum_agent_values,
        "min_item" : min_item,
        "max_item" : max_item,
        "min_agent" : min_agent,
        "max_agent" : max_agent
    }

def run_algorithm_with_random_instance_uniform(num_of_agents, num_of_items, algorithm):
    matchingFaStGen = {}
    agents, items, agentValuations, itemValuations = generate_regular_data(num_of_agents=num_of_agents, num_of_items=num_of_items)
    ins = Instance(agents=agents, items=items, valuations=agentValuations)
    allocation = AllocationBuilder(instance=ins)
    if inspect.getsource(algorithm) == inspect.getsource(FaStGen):
        matchingFaStGen = FaStGen.FaStGen(allocation, itemValuations)
    return evaluate_algorithm_output(matching=matchingFaStGen, agentValuations=agentValuations, itemValuations=itemValuations, agents=agents, items=items)

def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "FaStGenEXP.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [100,200,300],         
        "num_of_items":  [25],                   
        "algorithm": algorithms_to_check,
        # "random_seed": range(5),
    }
    experiment.run_with_time_limit(run_algorithm_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)

def multi_multi_plot_results(results_csv_file:str, save_to_file_template:str, filter:dict, 
     x_field:str, y_fields:list[str], z_field:str, mean:bool, 
     legend_properties:dict):
     for y_field in y_fields:
        plt.clf()
        save_to_file = save_to_file_template.format(y_field)    # Path where the plot will be saved
        single_plot_results(
            results_csv_file=results_csv_file,
            save_to_file=save_to_file,
            filter=filter,
            x_field=x_field,
            y_field=y_field,
            z_field=z_field,
            mean=mean,
            legend_properties=legend_properties
        )

def plot_course_allocation_results():
    y_fields=["sum_item_values","sum_agent_values", "min_item", "max_item",  "min_agent", "max_agent"]
    multi_multi_plot_results(
        results_csv_file="results/FaStGenEXP.csv", 
        save_to_file_template="results/FaStGenEXP_{}.png",
        filter={}, 
        x_field="num_of_agents", y_fields=y_fields, z_field="algorithm", mean=True,
        legend_properties={"size":6}, 
        )

if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    run_uniform_experiment()
    plot_course_allocation_results()