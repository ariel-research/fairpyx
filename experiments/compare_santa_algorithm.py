"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""

from fairpyx.instances import Instance
from fairpyx import divide
from typing import Dict, Set,Any
import fairpyx.algorithms as crs
import experiments_csv,logging
import numpy as np
from typing import Callable
from experiments_csv import single_plot_results
import matplotlib.pyplot as plt
import random

max_value = 1000
normalized_sum_of_values = 1000
TIME_LIMIT = 60

def random_binary_instance(
    num_of_players: int,
    num_of_gifts: int,
    max_value: int,
) -> Instance:
    """
    Creates an Instance satisfying:
      * Each player values at least one gift (value > 0)
      * Each gift is valued by at least one player
      * valuations are either 0 or the gift’s base value
    """
    agents = [f"P{i+1}" for i in range(num_of_players)]
    items  = [f"c{j+1}" for j in range(num_of_gifts)]
    base_values = {item: random.randint(1, max_value) for item in items}
    valuations = {a: {} for a in agents}

    for a in agents:
        k = random.randint(1, num_of_gifts)
        chosen = random.sample(items, k=k)
        for item in items:
            valuations[a][item] = base_values[item] if item in chosen else 0

    for item in items:
        if not any(valuations[a][item] > 0 for a in agents):
            a = random.choice(agents)
            valuations[a][item] = base_values[item]

    agent_caps = {a: num_of_gifts for a in agents}
    item_caps  = {i: 1 for i in items}

    return Instance(
        valuations=valuations,
        agent_capacities=agent_caps,
        item_capacities=item_caps
    )

# random instance for the Santa Claus problem
def allocation_with_random_instance(
    num_of_players: int,
    num_of_gifts: int,
    value_noise_ratio: float,
    algorithm: Callable,
    random_seed: int,
) -> Dict[str,Any]:
    """
    Creates a random instance for the Santa Claus problem.
    """

    # Initialize the random seed for reproducibility
    np.random.seed(random_seed)

    instance = random_binary_instance(
        num_of_players=num_of_players,
        num_of_gifts=num_of_gifts,
        max_value=max_value
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
        "num_of_players": [4,5,6,7,8],
        "num_of_gifts": [8,9,10,11,12,13,14,15,16],
        "value_noise_ratio": [0.0],
        "algorithm": [crs.old_santa_claus_main, crs.santa_claus_main, crs.round_robin], # comparing between these two algorithms
        "random_seed": list(range(5)),
    }

    ex.clear_previous_results()
    # Run experiment with time limit
    ex.run_with_time_limit(allocation_with_random_instance,input_ranges,time_limit=TIME_LIMIT)
    print("Experiment complete. Results saved to results/santaVSroundrobin_experiment.csv")
    print("\n DataFrame: \n", ex.dataFrame)


def plot_by_num_players(csv_path: str):
    # 1) num_of_players vs mean min_value
    plt.close('all')
    single_plot_results(
        csv_path,
        filter={"algorithm": ["round_robin", "santa_claus_main"]},
        x_field="num_of_players",
        y_field="min_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/santa1_min_value_by_num_players.png"
    )

    # 2) num_of_players vs mean total_value
    plt.close('all')
    single_plot_results(
        csv_path,
        filter={"algorithm": ["round_robin", "santa_claus_main"]},
        x_field="num_of_players",
        y_field="total_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/santa2_total_value_by_num_players.png"
    )

    # 3) num_of_players vs mean runtime
    plt.close('all')
    single_plot_results(
        csv_path,
        filter={},
        x_field="num_of_players",
        y_field="runtime",
        z_field="algorithm",
        mean=True,
        save_to_file="results/santa3_runtime_by_num_players.png"
    )


if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    run_experiment_santa()

    import pandas as pd

    df = pd.read_csv("results/santaVSroundrobin_experiment.csv")
    for alg, group in df.groupby("algorithm"):
        print(f"\n=== {alg} ===")
        print()

    plot_by_num_players("results/santaVSroundrobin_experiment.csv")

    df = pd.read_csv("results/santaVSroundrobin_experiment.csv")
    print(df.groupby("algorithm")["runtime"].mean())
