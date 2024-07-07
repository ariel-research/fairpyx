"""
Compare the performance of algorithms for fair course allocation.

Programmer: Erel Segal-Halevi
Since: 2023-07
"""
import os
######### COMMON VARIABLES AND ROUTINES ##########

import time
from fairpyx import divide, AgentBundleValueMatrix, Instance
import fairpyx.algorithms as crs
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns

from fairpyx.algorithms.ACEEI import EFTBStatus

max_value = 1000
normalized_sum_of_values = 1000
TIME_LIMIT = 100

algorithms_to_check = [
    crs.tabu_search,
    crs.find_ACEEI_with_EFTB,
    # crs.iterated_maximum_matching_adjusted,
    # crs.bidirectional_round_robin,
]


######### EXPERIMENT WITH UNIFORMLY-RANDOM DATA ##########

# def course_allocation_with_random_instance_uniform(
#         num_of_agents: int, num_of_items: int,
#         value_noise_ratio: float,
#         algorithm: Callable,
#         random_seed: int, ):
#     agent_capacity_bounds = [6, 6]
#     item_capacity_bounds = [40, 40]
#     np.random.seed(random_seed)
#     instance = Instance.random_uniform(
#         num_of_agents=num_of_agents, num_of_items=num_of_items,
#         normalized_sum_of_values=normalized_sum_of_values,
#         agent_capacity_bounds=agent_capacity_bounds,
#         item_capacity_bounds=item_capacity_bounds,
#         item_base_value_bounds=[1, max_value],
#         item_subjective_ratio_bounds=[1 - value_noise_ratio, 1 + value_noise_ratio]
#     )
#     return evaluate_algorithm_on_instance(algorithm, instance)


def create_initial_budgets(num_of_agents: int, beta: float = 100) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [1, 1 + beta]
    initial_budgets = np.random.uniform(1, 1 + beta, num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}


def evaluate_algorithm_on_instance(algorithm, instance, **kwargs):
    print(f"--------algorithm = {algorithm}")
    algorithm_name = algorithm.__name__ if callable(algorithm) else algorithm
    if algorithm_name == 'find_ACEEI_with_EFTB':
        if all(key in kwargs for key in ('delta', 'epsilon', 't')):
            initial_budgets = create_initial_budgets(instance.num_of_agents)
            allocation = divide(algorithm, instance, initial_budgets=initial_budgets, **kwargs)
        else:
            raise ValueError("Missing parameters for algorithm1. Required: delta, epsilon, t")
    elif algorithm_name == 'tabu_search':
        if all(key in kwargs for key in ('beta', 'delta')):
            beta = kwargs['beta']
            initial_budgets = create_initial_budgets(instance.num_of_agents, beta)
            allocation = divide(algorithm, instance, initial_budgets=initial_budgets, **kwargs)
        else:
            raise ValueError("Missing parameters for algorithm2. Required: beta, delta")
    else:
        raise ValueError("Unknown algorithm")

    matrix = AgentBundleValueMatrix(instance, allocation)
    matrix.use_normalized_values()
    return {
        "utilitarian_value": matrix.utilitarian_value(),
        "egalitarian_value": matrix.egalitarian_value(),
        "max_envy": matrix.max_envy(),
        "mean_envy": matrix.mean_envy(),
        "max_deficit": matrix.max_deficit(),
        "mean_deficit": matrix.mean_deficit(),
        "num_with_top_1": matrix.count_agents_with_top_rank(1),
        "num_with_top_2": matrix.count_agents_with_top_rank(2),
        "num_with_top_3": matrix.count_agents_with_top_rank(3),
    }


######### EXPERIMENT WITH UNIFORMLY-RANDOM DATA ##########

def course_allocation_with_random_instance_uniform(
        num_of_agents: int, num_of_items: int,
        value_noise_ratio: float,
        algorithm: Callable,
        random_seed: int, **kwargs):
    agent_capacity_bounds = [6, 6]
    item_capacity_bounds = [40, 40]
    np.random.seed(random_seed)
    instance = Instance.random_uniform(
        num_of_agents=num_of_agents, num_of_items=num_of_items,
        normalized_sum_of_values=normalized_sum_of_values,
        agent_capacity_bounds=agent_capacity_bounds,
        item_capacity_bounds=item_capacity_bounds,
        item_base_value_bounds=[1, max_value],
        item_subjective_ratio_bounds=[1 - value_noise_ratio, 1 + value_noise_ratio]
    )
    return evaluate_algorithm_on_instance(algorithm, instance, **kwargs)


######### COMPARING DELTA AND EPSILON PERFORMANCE - ACEEI ##########

RESULTS_BETA_DELTA_FILE_ACEEI = "results/compering_delta_epsilon_ACEEI.csv"


def run_delta_epsilon_experiment_ACEEI():
    # Remove existing results file if it exists
    if os.path.exists(RESULTS_BETA_DELTA_FILE_ACEEI):
        os.remove(RESULTS_BETA_DELTA_FILE_ACEEI)

    # Run on uniformly-random data with beta and delta parameters:
    experiment = experiments_csv.Experiment("results/", "compering_delta_epsilon_ACEEI.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [10, 20],
        "num_of_items": [4, 8],
        "value_noise_ratio": [0, 0.2, 0.4, 0.8, 1],
        "delta": [0.001, 0.34, 0.5, 0.8, 0.9],
        "epsilon": [0.3, 0.9, 1.2, 1.7, 3, 10],
        "t": [EFTBStatus.NO_EF_TB, EFTBStatus.EF_TB, EFTBStatus.CONTESTED_EF_TB],
        "algorithm": [crs.find_ACEEI_with_EFTB],
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)


def analyze_experiment_results_ACEEI():
    # Load the results from the CSV file
    df = pd.read_csv(RESULTS_BETA_DELTA_FILE_ACEEI)

    best_row = df.loc[df['runtime'].idxmin()]

    # Extract the best beta and delta values
    best_delta = best_row['delta']  # Assuming delta is already in the correct format
    best_epsilon = best_row['epsilon']

    print(f"Best delta: {best_delta}")
    print(f"Best epsilon: {best_epsilon}")

    return df


######### COMPARING BETA AND DELTA PERFORMANCE - TABU-SEARCH ##########

RESULTS_BETA_DELTA_FILE_TABU_SEARCH = "results/compering_beta_delta_tabu_search.csv"


def run_beta_delta_experiment_tabu_search():
    # Remove existing results file if it exists
    if os.path.exists(RESULTS_BETA_DELTA_FILE_TABU_SEARCH):
        os.remove(RESULTS_BETA_DELTA_FILE_TABU_SEARCH)

    # Run on uniformly-random data with beta and delta parameters:
    experiment = experiments_csv.Experiment("results/", "compering_beta_delta_tabu_search.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [10, 20],
        "num_of_items": [4, 8],
        "value_noise_ratio": [0, 0.2, 0.4, 0.8, 1],
        "beta": [0.001, 0.1, 0.3, 0.5, 3, 5],  # example values for beta
        "delta": [{0.001}, {0.34}, {0.5}, {0.8}, {0.9}],  # example values for delta
        "algorithm": [crs.tabu_search],  # only the tabu_search algorithm
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)


def analyze_experiment_results_tabu_search():
    # Load the results from the CSV file
    df = pd.read_csv(RESULTS_BETA_DELTA_FILE_TABU_SEARCH)

    best_row = df.loc[df['runtime'].idxmin()]

    # Extract the best beta and delta values
    best_beta = best_row['beta']
    best_delta = best_row['delta']  # Assuming delta is already in the correct format

    print(f"Best beta: {best_beta}")
    print(f"Best delta: {best_delta}")

    return df


def plot_speed_vs_param(df, param, algorithm_name):
    avg_runtime = df.groupby(param)['runtime'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_runtime, x=param, y='runtime', marker='o', err_style=None)
    plt.title(f'Algorithm Speed vs. {param.capitalize()} for {algorithm_name}')
    plt.xlabel(param.capitalize())
    plt.ylabel('Average Runtime (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_speed_vs_params(df, param1, param2, algorithm_name):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    param1_values = df[param1]
    param2_values = df[param2].apply(
        lambda x: float(list(ast.literal_eval(x))[0]) if isinstance(x, str) else x)  # Convert if necessary
    runtime_values = df['runtime']

    ax.scatter(param1_values, param2_values, runtime_values, c='b', marker='o')
    ax.set_title(f'Algorithm Speed vs. {param1.capitalize()} and {param2.capitalize()} for {algorithm_name}')
    ax.set_xlabel(param1.capitalize())
    ax.set_ylabel(param2.capitalize())
    ax.set_zlabel('Runtime (seconds)')

    plt.tight_layout()
    plt.show()


###########################


def run_uniform_experiment():
    # Run on uniformly-random data:
    experiment = experiments_csv.Experiment("results/", "course_allocation_uniform.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        # todo add more parameters : beta \ delta
        "num_of_agents": [100, 200, 300],
        "num_of_items": [25],
        "value_noise_ratio": [0, 0.2, 0.5, 0.8, 1],
        "algorithm": algorithms_to_check,
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)


######### EXPERIMENT WITH DATA GENERATED ACCORDING TO THE SZWS MODEL ##########

def course_allocation_with_random_instance_szws(
        num_of_agents: int, num_of_items: int,
        agent_capacity: int,
        supply_ratio: float,
        num_of_popular_items: int,
        mean_num_of_favorite_items: float,
        favorite_item_value_bounds: tuple[int, int],
        nonfavorite_item_value_bounds: tuple[int, int],
        algorithm: Callable,
        random_seed: int, ):
    np.random.seed(random_seed)
    instance = Instance.random_szws(
        num_of_agents=num_of_agents, num_of_items=num_of_items, normalized_sum_of_values=normalized_sum_of_values,
        agent_capacity=agent_capacity,
        supply_ratio=supply_ratio,
        num_of_popular_items=num_of_popular_items,
        mean_num_of_favorite_items=mean_num_of_favorite_items,
        favorite_item_value_bounds=favorite_item_value_bounds,
        nonfavorite_item_value_bounds=nonfavorite_item_value_bounds,
    )
    return evaluate_algorithm_on_instance(algorithm, instance)


def run_szws_experiment():
    # Run on SZWS simulated data:
    experiment = experiments_csv.Experiment("results/", "course_allocation_szws.csv", backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [100, 200, 300],
        "num_of_items": [25],  # in SZWS: 25
        "agent_capacity": [5],  # as in SZWS
        "supply_ratio": [1.1, 1.25, 1.5],  # as in SZWS
        "num_of_popular_items": [6, 9],  # as in SZWS
        "mean_num_of_favorite_items": [2.6, 3.85],
        # as in SZWS code https://github.com/marketdesignresearch/Course-Match-Preference-Simulator/blob/main/preference_generator_demo.ipynb
        "favorite_item_value_bounds": [(50, 100)],
        # as in SZWS code https://github.com/marketdesignresearch/Course-Match-Preference-Simulator/blob/main/preference_generator.py
        "nonfavorite_item_value_bounds": [(0, 50)],
        # as in SZWS code https://github.com/marketdesignresearch/Course-Match-Preference-Simulator/blob/main/preference_generator.py
        "algorithm": algorithms_to_check,
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_szws, input_ranges, time_limit=TIME_LIMIT)


######### EXPERIMENT WITH DATA SAMPLED FROM ARIEL 5783 DATA ##########

import json

filename = "data/ariel_5783_input.json"
with open(filename, "r", encoding="utf-8") as file:
    ariel_5783_input = json.load(file)


def course_allocation_with_random_instance_sample(
        max_total_agent_capacity: int,
        algorithm: Callable,
        random_seed: int, ):
    np.random.seed(random_seed)

    (valuations, agent_capacities, item_capacities, agent_conflicts, item_conflicts) = \
        (ariel_5783_input["valuations"], ariel_5783_input["agent_capacities"], ariel_5783_input["item_capacities"],
         ariel_5783_input["agent_conflicts"], ariel_5783_input["item_conflicts"])
    instance = Instance.random_sample(
        max_num_of_agents=max_total_agent_capacity,
        max_total_agent_capacity=max_total_agent_capacity,
        prototype_agent_conflicts=agent_conflicts,
        prototype_agent_capacities=agent_capacities,
        prototype_valuations=valuations,
        item_capacities=item_capacities,
        item_conflicts=item_conflicts)
    return evaluate_algorithm_on_instance(algorithm, instance)


def run_ariel_experiment():
    # Run on Ariel sample data:
    experiment = experiments_csv.Experiment("results/", "course_allocation_ariel.csv", backup_folder="results/backup/")
    input_ranges = {
        "max_total_agent_capacity": [1000, 1115, 1500, 2000],  # in reality: 1115
        "algorithm": algorithms_to_check,
        "random_seed": range(10),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_sample, input_ranges, time_limit=TIME_LIMIT)

######### Tabu Search - COMPARING USING THREADS IN student_best_bundle ##########

RESULTS_CACHE_FILE = "results/comparing_cache_Tabu_Search.csv"
def run_cache_experiment_Tabu():
    # Remove existing results file if it exists
    # if os.path.exists(RESULTS_CACHE_FILE):
    #     os.remove(RESULTS_CACHE_FILE)

    # Run on uniformly-random data with beta and delta parameters:
    experiment = experiments_csv.Experiment("results/", "comparing_cache_Tabu_Search.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": range(20,30),
        "num_of_items": [8, 10],
        "value_noise_ratio": [0, 0.2, 0.4, 0.8, 1],
        "beta": [0.001],
        "delta": [{0.34}],
        "use_cache": [False, True],
        "algorithm": [crs.tabu_search],
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=5)


def analyze_experiment_results_cache():
    # Load the results from the CSV file
    df = pd.read_csv(RESULTS_CACHE_FILE)

    best_row = df.loc[df['runtime'].idxmin()]

    # Extract relevant columns or parameters
    best_use_cache_value = best_row['use_cache']
    best_runtime = best_row['runtime']

    print(f"Best use_cache: {best_use_cache_value}")
    print(f"Corresponding Runtime: {best_runtime} seconds")

    return df

def plot_runtime_vs_cache(df, algorithm_name):
    plt.figure(figsize=(12, 8))

    # Plotting runtime vs. num_of_agents for each use_threads value
    sns.lineplot(data=df, x='num_of_agents', y='runtime', hue='use_cache', marker='o')

    plt.title(f'Runtime vs. Number of Agents for {algorithm_name}')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.legend(title='Use Cache', labels=['False: Average runtime', 'Markers False', 'True: Average runtime', 'Markers True'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import logging, experiments_csv

    experiments_csv.logger.setLevel(logging.INFO)
    # run_uniform_experiment()
    # run_szws_experiment()
    # run_ariel_experiment()
    # run_beta_delta_experiment()
    # max_agents_true, max_agents_false = run_check_performance_of_history()
    # print(f'Max number of agents handled in 60 seconds with check_history=True: {max_agents_true}')
    # print(f'Max number of agents handled in 60 seconds with check_history=False: {max_agents_false}')

    # ######### COMPARING BETA AND DELTA PERFORMANCE - TABU-SEARCH ##########
    # run_beta_delta_experiment_tabu_search()
    # df = analyze_experiment_results_tabu_search()
    # plot_speed_vs_param(df, 'beta', 'Tabu Search')
    # plot_speed_vs_param(df, 'delta', 'Tabu Search')
    # plot_speed_vs_params(df, 'beta', 'delta', 'Tabu Search')
    #
    # ######### COMPARING DELTA AND EPSILON PERFORMANCE - ACEEI ##########
    # run_delta_epsilon_experiment_ACEEI()
    # df = analyze_experiment_results_ACEEI()
    # plot_speed_vs_param(df, 'delta', 'ACEEI')
    # plot_speed_vs_param(df, 'epsilon', 'ACEEI')
    # plot_speed_vs_params(df, 'delta', 'epsilon', 'ACEEI')

    ######### Tabu Search - COMPARING USING THREADS IN student_best_bundle ##########
    run_cache_experiment_Tabu()
    df = analyze_experiment_results_cache()
    plot_runtime_vs_cache(df, 'Tabu Search')