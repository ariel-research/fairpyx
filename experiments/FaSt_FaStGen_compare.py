
import inspect
import logging
from fairpyx.algorithms.Optimization_Matching import FaSt, FaStGen
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import experiments_csv
from experiments_csv import single_plot_results, multi_plot_results
from matplotlib import pyplot as plt
import random
import numpy as np
random.seed(1)


TIME_LIMIT = 100

algorithms_to_check = [
    FaSt,
    FaStGen
    ]



def create_matrix(n, m):
    v = np.zeros((n, m), dtype=int)
    
    #Initialize the n,m squre
    v[n-1,m-1] = np.random.randint(1, 11)

    #Settin last column
    for j in range(n-1, 0, -1):
        v[j-1, m-1] = v[j, m-1] + np.random.randint(1, 11)

    #Setting last row
    for i in range(m-1, 0, -1):
        v[n-1, i-1] =  v[n-1, i] + np.random.randint(1, 11)

    #Setting the rest of the matrix
    for i in range(n-2, -1, -1):
        for j in range(m-2, -1, -1):
            v[i, j] = max(v[i+1, j], v[i, j+1]) + np.random.randint(1, 11)

    return v

def generate_isometric_data(num_of_agents, num_of_items):
    #agents dict creation > agents = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"} #Student set=S
    agents = [f"s{i}" for i in range (1,num_of_agents + 1)]
    #items dict creation > items = {"c1", "c2", "c3"} #College set=C
    items = [f"c{i}" for i in range (1,num_of_items + 1)]

    matrix = create_matrix(n=num_of_agents, m=num_of_items)

    # Extract row-based dictionary (agents to items)
    valuations = {f"s{i+1}": {f"c{j+1}": int(matrix[i, j]) for j in range(matrix.shape[1])} for i in range(matrix.shape[0])}

    # Extract column-based dictionary (items to agents)
    items_valuation = {f"c{j+1}": {f"s{i+1}": int(matrix[i, j]) for i in range(matrix.shape[0])} for j in range(matrix.shape[1])}

    return agents, items, valuations, items_valuation

def evaluate_algorithm_output(matching, valuation, agents, items):
    """
        Evaluate the algorithm output according to the following keys:
        - sum of the items values in the final matching.
        - sum of the agents values in the final matching.
        - the minimum value of item.
        - the maximum value of item.
        - the minimum value of agent.
        - the maximum value of agent.
    """
    valuation = {int(agent[1:]): {int(item[1:]): value for item, value in items.items()} for agent, items in valuation.items()}
    valuation_sums = {item: sum(valuation[agent][item] for agent in agents) for item, agents in matching.items()}
    agents = [int(agent[1:]) for agent in agents]
    sum_item_values = sum(value for value in valuation_sums.values())
    sum_agent_values = sum(valuation[agent][item] for item, agent in matching.items() for agent in agents)
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

def run_algorithm_with_random_instance_uniform(num_of_agents, num_of_items, algorithm):
    agents, items, valuations, items_valuation = generate_isometric_data(num_of_agents=num_of_agents, num_of_items=num_of_items)
    ins = Instance(agents=agents, items=items, valuations=valuations)
    allocation = AllocationBuilder(instance=ins)
    if inspect.getsource(algorithm) == inspect.getsource(FaSt):
        matching = FaSt.FaSt(allocation)
    elif inspect.getsource(algorithm) == inspect.getsource(FaStGen):
        matching = FaStGen.FaStGen(allocation, items_valuations=items_valuation)
        matching = {int(college[1:]): [int(student[1:]) for student in students] for college, students in matching.items()}
    return evaluate_algorithm_output(matching=matching, valuation=valuations, agents=agents, items=items)

def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "FaStVsFaStGenEXP.csv", backup_folder="results/backup/")
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
        save_to_file=save_to_file_template.format(y_field)
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
        results_csv_file="results/FaStVsFaStGenEXP.csv", 
        save_to_file_template="results/FaStFaStVSFaStGenEXP_{}.png",
        filter={}, 
        x_field="num_of_agents", y_fields=y_fields, z_field="algorithm", mean=True,
        legend_properties={"size":6}, 
        )
    
if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    run_uniform_experiment()
    plot_course_allocation_results()