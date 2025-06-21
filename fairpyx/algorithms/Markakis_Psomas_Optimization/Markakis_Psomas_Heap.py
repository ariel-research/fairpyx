import math
import logging
from functools import lru_cache
import heapq
from fairpyx import AllocationBuilder, divide, Instance
"""
Optimized implementation of Markakis-Psomas algorithm using:
- Priority queue (heap) for faster agent selection.
- LRU caching of Vn(alpha) function.
"""

# Optimized Vn computation with caching
@lru_cache(maxsize=1000)
def compute_vn(alpha: float, n: int) -> float:
    """
    Computes the worst-case guarantee value Vn(alpha) with caching and optimized calculations.
    """
    # Edge cases handling
    if n <= 1:
        return 0.0
    if alpha <= 0.0 or math.isclose(alpha, 0.0):
        return 1.0 / n
    if alpha >= 1.0 / (n - 1):
        return 0.0

    # Precompute commonly used values
    n_minus_1 = n - 1
    alpha_n = alpha * n
    inv_alpha_n = 1.0 / alpha_n if alpha_n > 0 else float('inf')
    
    # Calculate k using optimized formula
    k = max(1, math.floor((1 + alpha) * inv_alpha_n))
    
    # Precompute boundaries
    k_plus_1 = k + 1
    k_n_minus_1 = k * n - 1
    k_plus_1_n_minus_1 = k_plus_1 * n - 1
    
    # Calculate interval boundaries with safe division
    I_left = k_plus_1 / (k * k_plus_1_n_minus_1) if k * k_plus_1_n_minus_1 != 0 else float('inf')
    I_right = 1.0 / k_n_minus_1 if k_n_minus_1 > 0 else float('inf')
    
    NI_left = 1.0 / k_plus_1_n_minus_1 if k_plus_1_n_minus_1 != 0 else float('inf')
    NI_right = k_plus_1 / (k * k_plus_1_n_minus_1) if k * k_plus_1_n_minus_1 != 0 else float('inf')
    
    # Check intervals with tolerance
    tol = 1e-10
    if I_left - tol <= alpha <= I_right + tol:
        return max(0.0, 1.0 - k * n_minus_1 * alpha)
    
    if NI_left - tol <= alpha <= NI_right + tol:
        return max(0.0, 1.0 - (k_plus_1 * n_minus_1) / k_plus_1_n_minus_1)
    
    return 0.0

def algorithm1_worst_case_allocation_optimized(alloc: AllocationBuilder) -> None:
    """
    Optimized version of Algorithm 1 using a min-heap for efficient agent selection.
    
    :param alloc: AllocationBuilder — the current allocation state
    :return: None — allocation is done in-place
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    if alloc is None:
        return {}

    n = len(alloc.remaining_agents())
    logger.info(f"\nNew recursive call with {n} agents")

    # Base case: single agent
    if n == 1:
        agent = next(iter(alloc.remaining_agents()))
        items = list(alloc.remaining_items_for_agent(agent))
        logger.info(f"Only one agent '{agent}' remains — giving all items: {items}")
        alloc.give_bundle(agent, items)
        return

    # Initialize data structures
    Vn_alpha_i = {}
    max_values = {}
    sorted_items = {}
    heap = []
    
    # Precompute thresholds and prepare data for each agent
    for agent in alloc.remaining_agents():
        items = alloc.remaining_items_for_agent(agent)
        total_val = sum(alloc.effective_value(agent, item) for item in items)
        
        if total_val <= 0:  # Handle zero total value
            max_val = 0
            alpha = 0
        else:
            # Find max value efficiently
            max_val = max((alloc.effective_value(agent, item) for item in items), default=0)
            alpha = max_val / total_val
        
        Vn_alpha = compute_vn(alpha, n)
        threshold = Vn_alpha * total_val
        Vn_alpha_i[agent] = threshold
        max_values[agent] = max_val
        
        # Sort items in descending order for this agent
        sorted_items[agent] = sorted(
            items,
            key=lambda item: alloc.effective_value(agent, item),
            reverse=True
        )
        
        # Initialize heap with (gap, agent, pointer)
        heapq.heappush(heap, (threshold, agent, 0))

    # Track current state
    bundles = {agent: [] for agent in alloc.remaining_agents()}
    values = {agent: 0.0 for agent in alloc.remaining_agents()}
    pointers = {agent: 0 for agent in alloc.remaining_agents()}
    chosen_agent = None

    # Process items until an agent meets the threshold
    while heap and chosen_agent is None:
        gap, agent, ptr = heapq.heappop(heap)
        
        # Check if agent has items left
        if ptr >= len(sorted_items[agent]):
            continue
            
        # Get next item for this agent
        item = sorted_items[agent][ptr]
        item_value = alloc.effective_value(agent, item)
        
        # Update bundle and value
        bundles[agent].append(item)
        values[agent] += item_value
        pointers[agent] += 1
        
        # Calculate new gap
        new_gap = Vn_alpha_i[agent] - values[agent]
        
        # Check if agent meets the threshold
        if values[agent] >= Vn_alpha_i[agent]:
            chosen_agent = agent
            logger.info(f"Agent '{agent}' reached threshold: {values[agent]:.2f} ≥ {Vn_alpha_i[agent]:.2f}")
        else:
            # Push back to heap with updated state
            heapq.heappush(heap, (new_gap, agent, pointers[agent]))
    
    # Fallback if no agent was chosen (shouldn't normally happen)
    if chosen_agent is None:
        # Select agent with highest current value
        chosen_agent = max(alloc.remaining_agents(), key=lambda a: values[a])
        logger.warning(f"No agent reached threshold, selecting '{chosen_agent}' with highest value {values[chosen_agent]:.2f}")

    # Assign bundle and remove agent
    alloc.give_bundle(chosen_agent, bundles[chosen_agent])
    alloc.remove_agent_from_loop(chosen_agent)
    
    # Recursively allocate remaining items
    algorithm1_worst_case_allocation_optimized(alloc=alloc)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a sample instance
    valuations = {
        "A": {"1": 6, "2": 3, "3": 1},
        "B": {"1": 2, "2": 5, "3": 5}
    }
    instance = Instance(valuations=valuations)
    
    # Run the optimized algorithm
    allocation = divide(algorithm=algorithm1_worst_case_allocation_optimized, instance=instance)
    print("Final allocation:", allocation)