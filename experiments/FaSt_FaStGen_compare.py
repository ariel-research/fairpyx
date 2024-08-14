
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
    sum_item_values = sum(key for key in valuation_sums.keys())
    sum_agent_values = sum(valuation[agent][item] for item, agent in matching.items() for agent in agents)
    min_item = min(valuation_sums.items(), key=lambda x: x[1])
    max_item = max(valuation_sums.items(), key=lambda x: x[1])
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
    agents, items, valuations = generate_isometric_data(num_of_agents=num_of_agents, num_of_items=num_of_items)
    items_valuation = {item: {agent: value for agent, items in valuations.items() 
                              for item, value in items.items()} 
                              for item in {item for items in valuations.values() for item in items}}
    ins = Instance(agents=agents, items=items, valuations=valuations)
    allocation = AllocationBuilder(instance=ins)
    if inspect.getsource(algorithm) == inspect.getsource(FaSt):
        matching = FaSt.FaSt(allocation)
    elif inspect.getsource(algorithm) == inspect.getsource(FaStGen):
        matching = FaStGen.FaStGen(allocation, items_valuations=items_valuation)
    return evaluate_algorithm_output(matching=matching, valuation=valuations, agents=agents, items=items)

def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "FaStVsFaStGen.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [100,200,300],         
        "num_of_items":  [25],                   
        "algorithm": algorithms_to_check,
        # "random_seed": range(5),
    }
    experiment.run_with_time_limit(run_algorithm_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)

# def plot_experiment_results_from_csv(file_path):
#     algorithms = {}
#     with open(file_path, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             algorithm = row['algorithm']
#             if algorithm not in algorithms:
#                 algorithms[algorithm] = {
#                     'num_of_agents': [],
#                     'sum_item_values': [],
#                     'sum_agent_values': [],
#                     'min_item': [],
#                     'max_item': [],
#                     'min_agent': [],
#                     'max_agent': []
#                 }
#             algorithms[algorithm]['num_of_agents'].append(int(row['num_of_agents']))
#             algorithms[algorithm]['sum_item_values'].append(float(row['sum_item_values']))
#             algorithms[algorithm]['sum_agent_values'].append(float(row['sum_agent_values']))
#             algorithms[algorithm]['min_item'].append(float(row['min_item']))
#             algorithms[algorithm]['max_item'].append(float(row['max_item']))
#             algorithms[algorithm]['min_agent'].append(float(row['min_agent']))
#             algorithms[algorithm]['max_agent'].append(float(row['max_agent']))

#     plt.figure(figsize=(10, 6))

#     for algorithm, data in algorithms.items():
#         plt.plot(data['num_of_agents'], data['sum_item_values'], marker='o', label=f"{algorithm} - Sum Item Values")
#         plt.plot(data['num_of_agents'], data['sum_agent_values'], marker='x', label=f"{algorithm} - Sum Agent Values")
#         plt.plot(data['num_of_agents'], data['min_item'], marker='^', label=f"{algorithm} - Min Item Value")
#         plt.plot(data['num_of_agents'], data['max_item'], marker='v', label=f"{algorithm} - Max Item Value")
#         plt.plot(data['num_of_agents'], data['min_agent'], marker='<', label=f"{algorithm} - Min Agent Value")
#         plt.plot(data['num_of_agents'], data['max_agent'], marker='>', label=f"{algorithm} - Max Agent Value")

#     plt.xlabel("Number of Agents")
#     plt.ylabel("Value")
#     plt.title("FaSt vs FaStGen Algorithm Performance")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig("FaSt_vs_FaStGen_experiment_results.png")
#     plt.show()

if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    run_uniform_experiment()
    # plot_experiment_results_from_csv("results/FaStVsFaStGen.csv")