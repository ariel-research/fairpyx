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

max_value = 1000
normalized_sum_of_values = 1000
TIME_LIMIT = 100

algorithms_to_check = [
    crs.tabu_search,
    crs.find_ACEEI_with_EFTB,
    # crs.iterated_maximum_matching_adjusted,
    # crs.bidirectional_round_robin,
]


def evaluate_algorithm_on_instance(algorithm, instance):
    allocation = divide(algorithm, instance)
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


def create_initial_budgets(num_of_agents: int, beta: float) -> dict:
    # Create initial budgets for each agent, uniformly distributed in the range [1, 1 + beta]
    initial_budgets = np.random.uniform(1, 1 + beta, num_of_agents)
    return {f's{agent + 1}': initial_budgets[agent] for agent in range(num_of_agents)}


def evaluate_algorithm_on_instance(algorithm, instance, beta, delta):
    initial_budgets = create_initial_budgets(instance.num_of_agents, beta)
    allocation = divide(algorithm, instance, initial_budgets=initial_budgets, beta=beta, delta=set(delta))
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
        beta: float, delta: List[float],
        algorithm: Callable,
        random_seed: int):
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
    return evaluate_algorithm_on_instance(algorithm, instance, beta, delta)


######### COMPARING BETA AND DELTA PERFORMANCE ##########

RESULTS_BETA_DELTA_FILE = "results/course_allocation_beta_delta.csv"


def run_beta_delta_experiment():
    # Remove existing results file if it exists
    if os.path.exists(RESULTS_BETA_DELTA_FILE):
        os.remove(RESULTS_BETA_DELTA_FILE)

    # Run on uniformly-random data with beta and delta parameters:
    experiment = experiments_csv.Experiment("results/", "course_allocation_beta_delta.csv",
                                            backup_folder="results/backup/")
    input_ranges = {
        "num_of_agents": [10],
        "num_of_items": [4],
        "value_noise_ratio": [0, 0.2, 0.4, 0.8, 1],
        "beta": [0.001, 0.1, 0.3, 0.5, 3, 5],  # example values for beta
        "delta": [{0.001}, {0.34}, {0.5}, {0.8}, {0.9}],
        # example values for delta
        "algorithm": [crs.tabu_search],  # only the tabu_search algorithm
        "random_seed": range(5),
    }
    experiment.run_with_time_limit(course_allocation_with_random_instance_uniform, input_ranges, time_limit=TIME_LIMIT)


def analyze_experiment_results():
    # Load the results from the CSV file
    df = pd.read_csv(RESULTS_BETA_DELTA_FILE)

    best_row = df.loc[df['runtime'].idxmin()]

    # Extract the best beta and delta values
    best_beta = best_row['beta']
    best_delta = best_row['delta']  # Assuming delta is already in the correct format

    print(f"Best beta: {best_beta}")
    print(f"Best delta: {best_delta}")

    return df


def plot_speed_vs_beta(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='beta', y='runtime', marker='o', err_style=None)
    plt.title('Algorithm Speed vs. Beta')
    plt.xlabel('Beta')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_speed_vs_delta(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='delta', y='runtime', marker='o', err_style=None)
    plt.title('Algorithm Speed vs. Delta')
    plt.xlabel('Delta')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_speed_vs_beta_and_delta(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    beta_values = df['beta']
    delta_values = df['delta'].apply(
        lambda x: float(list(ast.literal_eval(x))[0]))  # Assuming delta is a set and we take the first element
    runtime_values = df['runtime']

    ax.scatter(beta_values, delta_values, runtime_values, c='b', marker='o')
    ax.set_title('Algorithm Speed vs. Beta and Delta')
    ax.set_xlabel('Beta')
    ax.set_ylabel('Delta')
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


# def run_check_performance_of_history():
#     num_of_agents_values = [10, 20]
#     beta = 0.5
#     delta = [0.1, 0.2]
#     value_noise_ratio = 0.5
#     num_of_items = 25
#     random_seed = 42
#     algorithm = crs.tabu_search
#
#     times_true = []
#     times_false = []
#     max_agents_within_60s_true = 0
#     max_agents_within_60s_false = 0
#
#     for num_of_agents in num_of_agents_values:
#         start_time = time.time()
#         course_allocation_with_random_instance_uniform(num_of_agents, num_of_items, value_noise_ratio, beta, delta, algorithm, random_seed, check_history=True)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         times_true.append(elapsed_time)
#         if elapsed_time < 60:
#             max_agents_within_60s_true = num_of_agents
#
#         start_time = time.time()
#         course_allocation_with_random_instance_uniform(num_of_agents, num_of_items, value_noise_ratio, beta, delta, algorithm, random_seed, check_history=False)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         times_false.append(elapsed_time)
#         if elapsed_time < 60:
#             max_agents_within_60s_false = num_of_agents
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(num_of_agents_values, times_true, label='check_history=True', marker='o')
#     plt.plot(num_of_agents_values, times_false, label='check_history=False', marker='o')
#     plt.xlabel('Number of Agents')
#     plt.ylabel('Run Time (seconds)')
#     plt.title('Run Time vs Number of Agents for Tabu Search')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('check_history_performance.png')
#     plt.show()
#
#     return max_agents_within_60s_true, max_agents_within_60s_false


######### MAIN PROGRAM ##########

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
    run_beta_delta_experiment()
    df = analyze_experiment_results()
    plot_speed_vs_delta(df)
    plot_speed_vs_beta(df)
    plot_speed_vs_beta_and_delta(df)
