import time
import random
import csv
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.leximin_primal import leximin_primal

def generate_random_instance(n_agents, n_items, max_agent_cap=2, max_item_cap=5):
    agents = list(range(1, n_agents + 1))
    items = [chr(ord('a') + i) for i in range(n_items)]
    valuations = {
        a: {item: 1 for item in random.sample(items, k=random.randint(1, n_items // 2))}
        for a in agents
    }
    agent_capacities = {a: random.randint(1, max_agent_cap) for a in agents}
    item_capacities = {i: random.randint(1, max_item_cap) for i in items}
    return Instance(valuations, agent_capacities, item_capacities)

def evaluate_allocation(distribution):
    expected_utilities = {a: 0 for a in distribution[0][0].keys()}
    for alloc, prob in distribution:
        for a, items in alloc.items():
            expected_utilities[a] += prob * sum(items.values())
    return sum(expected_utilities.values()), min(expected_utilities.values())

def benchmark(n_agents, n_items, output_file):
    instance = generate_random_instance(n_agents, n_items)
    alloc = AllocationBuilder(instance)
    start = time.time()
    leximin_primal(alloc)
    end = time.time()

    total_util, min_util = evaluate_allocation(alloc.distribution)
    runtime = end - start

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n_agents, n_items, total_util, min_util, runtime])

# Run benchmarks
for n in range(10, 101, 10):
    benchmark(n, n, "../experiments-csv/leximin_benchmark.csv")
