"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""

from fairpyx import Instance, AllocationBuilder, divide, AgentBundleValueMatrix
from typing import Dict, List, Set, Tuple,Any
import fairpyx.algorithms as crs
import experiments_csv,logging
import numpy as np
from typing import Callable
from experiments_csv import single_plot_results, multi_multi_plot_results

max_value = 1000
normalized_sum_of_values = 1000
TIME_LIMIT = 60

# # run the algorithm
# def run_santa_claus(instance) -> Dict[str, Set[str]]:
#     """
#     Applies the Santa Claus allocation algorithm to the given instance
#     and returns the resulting allocation mapping.
#     """
#     allocation_builder = AllocationBuilder(instance=instance)
#     return crs.santa_claus_main(allocation_builder)


# random instance for the Santa Claus problem
def allocation_with_random_instance(
    num_of_players: int,
    num_of_gifts: int,
    value_noise_ratio: float,
    algorithm: Callable,
    random_seed: int,
) -> Dict[str,Any]:
    """
    Creates a random instance for the Santa Claus problem where each player receives at least one gift,
    each gift is unique (capacity 1), and applies the given allocation algorithm to evaluate metrics.
    """
    # Ensure each player gets at least one gift (up to all gifts)
    agent_capacity_bounds = (1, 1)
    # Ensure each gift is available exactly once
    item_capacity_bounds = (1, 1)

    # Initialize the random seed for reproducibility
    np.random.seed(random_seed)

    instance = Instance.random_uniform(
        num_of_agents=num_of_players,
        num_of_items=num_of_gifts,
        normalized_sum_of_values=normalized_sum_of_values,
        agent_capacity_bounds=agent_capacity_bounds,
        item_capacity_bounds=item_capacity_bounds,
        item_base_value_bounds=(1, max_value),
        item_subjective_ratio_bounds=(1 - value_noise_ratio, 1 + value_noise_ratio)
    )

    allocation: Dict[str, Set[str]] = divide(algorithm, instance=instance)

    total_value = sum(
        instance._valuations[agent][gift]
        for agent, gifts in allocation.items()
        for gift in gifts
    )
    min_value = min(
        sum(instance._valuations[agent][gift] for gift in gifts)
        for agent, gifts in allocation.items()
    )

    # Building a palette map of all the fields that will be recorded in the CSV
    result: Dict[str, Any] = {
                "total_value": total_value,
                "min_value": min_value
                              }
    # adding all the players from allocation
    result.update(allocation)

    return result

def run_experiment_santa():
    ex = experiments_csv.Experiment("results/", "santaVSroundrobin_experiment.csv",  backup_folder="results/backup/")
    ex.logger.setLevel(logging.INFO)

    # Define parameter grid matching function signature
    input_ranges = {
        "num_of_players": [8],
        "num_of_gifts": [8],
        "value_noise_ratio": [0.0],
        "algorithm": [crs.santa_claus_main, crs.round_robin], # comparing between these two algorithms
        "random_seed": list(range(5)),
    }

    ex.clear_previous_results()
    # Run experiment with time limit
    ex.run_with_time_limit(allocation_with_random_instance,input_ranges,time_limit=TIME_LIMIT)
    print("Experiment complete. Results saved to results/santaVSroundrobin_experiment.csv")
    print("\n DataFrame: \n", ex.dataFrame)


def plot_algorithm_comparison(csv_path: str):
    single_plot_results(
        csv_path,
        filter={},
        x_field="random_seed",
        y_field="total_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/total_value_comparison.png"
    )
    single_plot_results(
        csv_path,
        filter={},
        x_field="random_seed",
        y_field="min_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/min_value_comparison.png"
    )


if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)

    run_experiment_santa()

    import pandas as pd

    df = pd.read_csv("results/santaVSroundrobin_experiment.csv")
    for alg, group in df.groupby("algorithm"):
        print(f"\n=== {alg} ===")
        print(group[["random_seed", "s1", "s2", "s3", "s4", "s5", "s6"]])

    plot_algorithm_comparison("results/santaVSroundrobin_experiment.csv")

