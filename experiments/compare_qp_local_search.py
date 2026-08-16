"""
Compare the performance of the QP Local Search algorithm vs the Santa Claus algorithm.

Based on:
"Quasi-Polynomial Local Search for Restricted Max-Min Fair Allocation"
By Lukas Polacek and Ola Svensson (2014)
https://arxiv.org/pdf/1205.1373

Programmers: Rotem Melamed
Date: 2025-04-23
"""

from fairpyx.instances import Instance
from fairpyx import divide
from typing import Dict, Set, Any
import fairpyx.algorithms as crs
from fairpyx.algorithms.polacek_svensson import qp_max_min_allocation
import experiments_csv, logging
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
      * valuations are either 0 or the gift's base value
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

def evaluate_algorithm_on_instance(algorithm, instance):
    """
    Runs the given algorithm on the instance and returns evaluation metrics.
    """
    # qp_max_min_allocation takes Instance directly, santa_claus_main uses divide()
    if algorithm is qp_max_min_allocation:
        allocation = algorithm(instance)
    else:
        allocation = divide(algorithm, instance=instance)

    total_value = sum(
        instance._valuations[agent][gift]
        for agent, gifts in allocation.items()
        for gift in gifts
    )
    min_value = min(
        sum(instance._valuations[agent][gift] for gift in gifts)
        for agent, gifts in allocation.items()
    )

    return {
        "total_value": total_value,
        "min_value": min_value,
    }

# random instance for the QP Local Search vs Santa Claus comparison
def allocation_with_random_instance(
    num_of_players: int,
    num_of_gifts: int,
    value_noise_ratio: float,
    algorithm: Callable,
    random_seed: int,
) -> Dict[str, Any]:
    """
    Creates a random instance and evaluates the given algorithm.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    instance = random_binary_instance(
        num_of_players=num_of_players,
        num_of_gifts=num_of_gifts,
        max_value=max_value
    )

    return evaluate_algorithm_on_instance(algorithm, instance)

def run_experiment_qp():
    ex = experiments_csv.Experiment("results/", "qpVSsanta_experiment.csv", backup_folder="results/backup/")
    ex.logger.setLevel(logging.INFO)

    input_ranges = {
        "num_of_players": [3, 4, 5, 6],
        "num_of_gifts": [6, 8, 10, 12, 14],
        "value_noise_ratio": [0.0],
        "algorithm": [qp_max_min_allocation, crs.santa_claus_main],
        "random_seed": list(range(5)),
    }

    ex.clear_previous_results()
    ex.run_with_time_limit(allocation_with_random_instance, input_ranges, time_limit=TIME_LIMIT)
    print("Experiment complete. Results saved to results/qpVSsanta_experiment.csv")
    print("\n DataFrame: \n", ex.dataFrame)


def plot_by_num_players(csv_path: str):
    # 1) num_of_players vs mean min_value
    plt.close('all')
    single_plot_results(
        csv_path,
        filter={"algorithm": ["qp_max_min_allocation", "santa_claus_main"]},
        x_field="num_of_players",
        y_field="min_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/qp1_min_value_by_num_players.png"
    )

    # 2) num_of_players vs mean total_value
    plt.close('all')
    single_plot_results(
        csv_path,
        filter={"algorithm": ["qp_max_min_allocation", "santa_claus_main"]},
        x_field="num_of_players",
        y_field="total_value",
        z_field="algorithm",
        mean=True,
        save_to_file="results/qp2_total_value_by_num_players.png"
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
        save_to_file="results/qp3_runtime_by_num_players.png"
    )


if __name__ == "__main__":
    experiments_csv.logger.setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    run_experiment_qp()

    import pandas as pd

    df = pd.read_csv("results/qpVSsanta_experiment.csv")
    for alg, group in df.groupby("algorithm"):
        print(f"\n=== {alg} ===")
        print()

    plot_by_num_players("results/qpVSsanta_experiment.csv")

    df = pd.read_csv("results/qpVSsanta_experiment.csv")
    print(df.groupby("algorithm")["runtime"].mean())
