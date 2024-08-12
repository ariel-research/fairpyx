import logging
import numpy as np
import matplotlib.pyplot as plt
from fairpyx.algorithms.almost_egalitarian import almost_egalitarian_allocation
from fairpyx.satisfaction import AgentBundleValueMatrix
from fairpyx.adaptors import divide, divide_with_priorities
from fairpyx.algorithms.course_match.main_course_match import course_match_algorithm, check_envy
from fairpyx.instances import Instance
import time

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to logging.INFO to reduce verbosity
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Define a function to compute and collect metrics
def compute_metrics(satisfaction_matrix):
    metrics = {
        "Max Envy": satisfaction_matrix.max_envy(),
        "Mean Envy": satisfaction_matrix.mean_envy(),
        "Count Agents with Top Rank 1": satisfaction_matrix.count_agents_with_top_rank(1),
        "Count Agents with Top Rank 2": satisfaction_matrix.count_agents_with_top_rank(2),
    }
    return metrics

# Function to make a budget
def make_budget(num_of_agents: int = 30, low: int = 2, high: int = 2.1, agent_name_template: str = "s{index}"):
    logger.debug(f"Creating budget for {num_of_agents} agents with budgets ranging from {low} to {high}")
    budget_list = np.random.uniform(low=low, high=high, size=num_of_agents)
    agents = [agent_name_template.format(index=i + 1) for i in range(num_of_agents)]
    budget = {agent: agent_budget for agent, agent_budget in zip(agents, budget_list)}
    return budget

def create_random_sublists(original_array: list, num_divisions: int = 4, random_seed: int = None):
    logger.debug(f"Creating random sublists from array of length {len(original_array)} with {num_divisions} divisions")
    if random_seed is None:
        random_seed = np.random.randint(1, 2**31)
    np.random.seed(random_seed)

    # Generate the number of divisions
    if num_divisions is None:
        num_divisions = np.random.randint(1, len(original_array) + 1)

    # Shuffle the original array
    shuffled_array = np.random.permutation(original_array)
    
    # Determine sizes of each sublist
    sublist_sizes = []
    remaining_elements = len(original_array)
    
    for _ in range(num_divisions - 1):
        # Random size for each sublist
        size = np.random.randint(1, remaining_elements - (num_divisions - len(sublist_sizes) - 1) + 1)
        sublist_sizes.append(size)
        remaining_elements -= size
    
    # Add the remaining elements as the last sublist size
    sublist_sizes.append(remaining_elements)
    
    # Create the sublists
    sublists = []
    current_index = 0
    
    for size in sublist_sizes:
        sublist = shuffled_array[current_index:current_index + size]
        sublists.append(sublist.tolist())
        current_index += size
    
    return sublists

# Plot the metrics
def plot_metrics(metrics_all, metric_name, algorithm_names, title):
    logger.debug(f"Plotting metrics for {metric_name}")
    plt.figure(figsize=(12, 6))
    for metrics, algo_name in zip(metrics_all, algorithm_names):
        values = [m[metric_name] for m in metrics]
        plt.plot(range(len(values)), values, label=algo_name)
    
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'{title}_{metric_name}.png')
    plt.close()


# Plot the execution times
def plot_execution_times(times_all, algorithm_names, title):
    logger.debug("Plotting execution times")
    plt.figure(figsize=(12, 6))
    for times, algo_name in zip(times_all, algorithm_names):
        plt.plot(range(len(times)), times, label=algo_name)
    
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.savefig(f'{title}_execution_time.png')
    plt.close() 

def simulation_with_diffrent_distributions(list_num_items_capacity: list, list_num_agents: list, divide_type='default'):
    logger.info(f"Starting simulation with distributions: {list_num_items_capacity} items capacity, {list_num_agents} agents, divide_type={divide_type}")
    metrics_cm_all = []
    metrics_al_eg_all = []
    times_cm = []
    times_al_eg = []

    for num_agents, item_capacity in zip(list_num_agents, list_num_items_capacity):
        logger.debug(f"Generating random instance for {num_agents} agents and {item_capacity} items capacity")
        random_instance = Instance.random_uniform(
            num_of_agents=num_agents,
            num_of_items=15,
            agent_capacity_bounds=[2, 3],
            item_capacity_bounds=[item_capacity, item_capacity],
            item_base_value_bounds=[1, 100],
            item_subjective_ratio_bounds=[0.5, 1.5],
            normalized_sum_of_values=100,
            random_seed=1
        )

        # Budget
        budget = make_budget(num_of_agents=num_agents)
        
        # Perform allocations using different algorithms with time checks
        start_time = time.time()
        if divide_type == 'default':
            logger.debug("Performing allocation using Course Match Algorithm without priorities")
            alloc_cm = divide(algorithm=course_match_algorithm, instance=random_instance, budget=budget)
        elif divide_type == 'with_priorities':
            logger.debug("Performing allocation using Course Match Algorithm with priorities")
            priorities_list = create_random_sublists(original_array=list(random_instance.agents), random_seed=1)
            alloc_cm = divide(algorithm=course_match_algorithm, instance=random_instance, budget=budget, priorities_student_list=priorities_list)
        times_cm.append(time.time() - start_time)

        start_time = time.time()
        if divide_type == 'default':
            logger.debug("Performing allocation using Almost Egalitarian Allocation without priorities")
            alloc_al_eg = divide(algorithm=almost_egalitarian_allocation, instance=random_instance)
        elif divide_type == 'with_priorities':
            logger.debug("Performing allocation using Almost Egalitarian Allocation with priorities")
            alloc_al_eg = divide_with_priorities(algorithm=almost_egalitarian_allocation, instance=random_instance, agent_priority_classes=priorities_list)
        times_al_eg.append(time.time() - start_time)

        # Instantiate AgentBundleValueMatrix with the random instance and for both allocations
        logger.debug("Instantiating AgentBundleValueMatrix for Course Match allocation")
        satisfaction_matrix_cm = AgentBundleValueMatrix(random_instance, alloc_cm, normalized=False)
        logger.debug("Instantiating AgentBundleValueMatrix for Almost Egalitarian allocation")
        satisfaction_matrix_al_eg = AgentBundleValueMatrix(random_instance, alloc_al_eg, normalized=False)
        
        metrics_cm = compute_metrics(satisfaction_matrix_cm)
        metrics_al_eg = compute_metrics(satisfaction_matrix_al_eg)
        
        metrics_cm_all.append(metrics_cm)
        metrics_al_eg_all.append(metrics_al_eg)

    metric_names = [
        "Max Envy", "Mean Envy", "Count Agents with Top Rank 1", "Count Agents with Top Rank 2"
    ]

    algorithm_names = ["Course Match Algorithm", "Almost Egalitarian Allocation"]

    for metric_name in metric_names:
        plot_metrics([metrics_cm_all, metrics_al_eg_all], metric_name, algorithm_names, f"Comparison of {metric_name}")

    plot_execution_times([times_cm, times_al_eg], algorithm_names, "Execution Time Comparison")


if __name__ == "__main__":
    list_num_agents = [num_agents for num_agents in range(30, 210, 30)]
    list_num_items_capacity = [items_capacity for items_capacity in range(7, 42, 6)]
    

    simulation_with_diffrent_distributions(list_num_items_capacity, list_num_agents, divide_type='default')

    simulation_with_diffrent_distributions(list_num_items_capacity, list_num_agents, divide_type='with_priorities')
