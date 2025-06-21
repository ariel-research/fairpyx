import time
import matplotlib.pyplot as plt
import numpy as np
from fairpyx import divide, Instance
from fairpyx.algorithms.markakis_psomas import algorithm1_worst_case_allocation
from fairpyx.algorithms.Markakis_Psomas_Optimization.Markakis_Psomas_Heap import algorithm1_worst_case_allocation_optimized

# Generate random instance for benchmarking
def generate_random_instance(agents, items, valuation_range=(1, 100)):
    """
    Creates a random instance with the specified number of agents and items.
    
    :param agents: Number of agents
    :param items: Number of items
    :param valuation_range: Tuple (min_value, max_value) for random valuations
    :return: Instance object
    """
    valuations = {}
    for i in range(agents):
        agent_name = f"Agent{i+1}"
        valuations[agent_name] = {}
        for j in range(items):
            item_name = f"Item{j+1}"
            value = np.random.randint(valuation_range[0], valuation_range[1] + 1)
            valuations[agent_name][item_name] = value
    return Instance(valuations=valuations)

# Generate performance comparison data
def benchmark():
    sizes = [5, 10, 20, 50]
    original_times = []
    optimized_times = []
    
    for size in sizes:
        # Create random instance
        instance = generate_random_instance(
            agents=size,
            items=size*2,
            valuation_range=(1, 100)
        )
        
        print(f"\n=== Benchmarking size: {size} agents, {size*2} items ===")
        
        # Benchmark original
        start = time.time()
        try:
            divide(algorithm=algorithm1_worst_case_allocation, instance=instance)
            orig_time = time.time() - start
            original_times.append(orig_time)
            print(f"Original completed in {orig_time:.2f}s")
        except Exception as e:
            orig_time = float('inf')
            original_times.append(orig_time)
            print(f"Original failed: {str(e)}")
        
        # Benchmark optimized
        start = time.time()
        try:
            divide(algorithm=algorithm1_worst_case_allocation_optimized, instance=instance)
            opt_time = time.time() - start
            optimized_times.append(opt_time)
            print(f"Optimized completed in {opt_time:.2f}s")
        except Exception as e:
            opt_time = float('inf')
            optimized_times.append(opt_time)
            print(f"Optimized failed: {str(e)}")
        
        if orig_time < float('inf') and opt_time < float('inf') and opt_time > 0:
            print(f"Speedup: {orig_time/opt_time:.1f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, original_times, 'ro-', label='Original')
    plt.plot(sizes, optimized_times, 'go-', label='Optimized')
    plt.xlabel('Number of Agents')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison: Original vs Optimized')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    
    # Save results to CSV
    with open('benchmark_results.csv', 'w') as f:
        f.write("Agents,Items,OriginalTime,OptimizedTime,Speedup\n")
        for i, size in enumerate(sizes):
            items = size * 2
            speedup = original_times[i]/optimized_times[i] if optimized_times[i] > 0 else 0
            f.write(f"{size},{items},{original_times[i]},{optimized_times[i]},{speedup}\n")
    
    print("\nBenchmark completed! Results saved to performance_comparison.png and benchmark_results.csv")

# Run benchmark
if __name__ == "__main__":
    benchmark()