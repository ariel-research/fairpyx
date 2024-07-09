"""
Compare the performance of algorithms for fair course allocation.

Programmer: Erel Segal-Halevi
Since: 2023-07
"""
import os
######### COMMON VARIABLES AND ROUTINES ##########

import random

from fairpyx import Instance, AgentBundleValueMatrix, divide
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
import fairpyx.algorithms as crs

max_value = 1000
normalized_sum_of_values = 1000
TIME_LIMIT = 100


def create_initial_budgets(num_of_agents: int, beta: float = 100) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [1, 1 + beta]
    initial_budgets = np.random.uniform(1, 1 + beta, num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}


def evaluate_algorithm_on_instance(algorithm, instance, **kwargs):
    beta = kwargs.get("beta", 100)
    initial_budgets = create_initial_budgets(instance.num_of_agents, beta)
    allocation = divide(algorithm, instance=instance, initial_budgets=initial_budgets, **kwargs)

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


######### COMPARING USING CACHE - TABU SEARCH ##########

RESULTS_CACHE_TABU_SEARCH = "results/compering_using_cache_tabu_search.csv"

def run_cache_experiment_tabu_search():
    # Run on uniformly-random data with beta and delta parameters:
    experiment = experiments_csv.Experiment("results/", "compering_using_cache_tabu_search.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [10, 20, 30],
        "num_of_items": [5, 10, 15],
        "value_noise_ratio": [0, 0.2],
        # "value_noise_ratio": [0, 0.2, 0.4, 0.8, 1],
        "beta": [3],
        "delta": [{0.5}],
        # "delta": [{0.001}, {0.1}, {0.3}, {0.5}, {0.9}],
        "use_cache": [False, True],
        "algorithm": [crs.tabu_search],
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)



def analyze_experiment_results_cache():
    # Load the results from the CSV file
    df = pd.read_csv(RESULTS_CACHE_TABU_SEARCH)

    best_row = df.loc[df['runtime'].idxmin()]

    # Extract relevant columns or parameters
    best_use_cache_value = best_row['use_cache']
    best_runtime = best_row['runtime']

    print(f"Best use_cache: {best_use_cache_value}")
    print(f"Corresponding Runtime: {best_runtime} seconds")

    return df


##### PLOT #######

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


def plot_runtime_vs_cache(df, algorithm_name):
    plt.figure(figsize=(12, 8))

    # Plotting runtime vs. num_of_agents for each use_cache value
    sns.lineplot(data=df[df['use_cache'] == True], x='num_of_agents', y='runtime', marker='o',
                 label='Use Cache = True')
    sns.lineplot(data=df[df['use_cache'] == False], x='num_of_agents', y='runtime', marker='o',
                 label='Use Cache = False')

    plt.title(f'Runtime vs. Number of Agents for {algorithm_name}')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)

    # Customizing the legend
    legend_labels = {
        'Use Cache = True': 'blue',  # Blue line and markers for Use Cache = True
        'Use Cache = False': 'orange'  # Orange line and markers for Use Cache = False
    }

    handles = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in legend_labels.values()]
    labels = legend_labels.keys()
    plt.legend(handles, labels, title='Legend')

    plt.tight_layout()
    plt.show()



###########################

if __name__ == "__main__":
    import logging, experiments_csv

    experiments_csv.logger.setLevel(logging.INFO)

    run_cache_experiment_tabu_search()
    df = analyze_experiment_results_cache()
    plot_runtime_vs_cache(df, 'Tabu Search')