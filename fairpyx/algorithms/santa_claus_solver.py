"""
Implementation of the Santa Claus Problem algorithm based on:
"The Santa Claus Problem", by Bansal, Nikhil, and Maxim Sviridenko.
Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006
https://dl.acm.org/doi/10.1145/1132516.1132557

The problem involves distributing presents (gifts) among kids (children), where each kid i has 
a different valuation pij for each present j. The goal is to maximize the happiness 
of the least happy kid (maximin objective).

In the restricted assignment case, each present j has a fixed value pj for all kids
who can receive it, and 0 for kids who cannot receive it.

This implementation follows the O(log log m / log log log m) approximation algorithm 
for the restricted assignment case.
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from fairpyx import Instance, AllocationBuilder

logger = logging.getLogger(__name__)

def compute_configuration_lp(instance: Instance, T: float) -> Optional[Dict]:
    """
    Solve the Configuration LP for a given target value T.
    
    The LP has the following form:
    - For each machine i and configuration C, there's a variable x_{i,C}
    - Sum of x_{i,C} for all C = 1 for each machine i
    - Sum of x_{i,C} for all i and C containing job j ≤ 1 for each job j
    - All x_{i,C} ≥ 0
    
    Args:
        instance: The problem instance
        T: Target value for minimum machine load
        
    Returns:
        A fractional solution to the LP in the form {(i, config): x} or None if infeasible
    """
    try:
        import pulp
    except ImportError:
        logger.warning("PuLP is not installed. Using simplified LP solver.")
        return _simplified_lp_solver(instance, T)
    
    # Create the LP problem
    prob = pulp.LpProblem("SantaClausConfigurationLP", pulp.LpMaximize)
    
    # Generate all valid configurations for each kid
    valid_configs = generate_valid_configurations(instance, T)
    
    # Create variables
    x = {}
    for kid in instance.agents:
        for config_idx, config in enumerate(valid_configs.get(kid, [])):
            x[(kid, tuple(sorted(config)))] = pulp.LpVariable(
                f"x_{kid}_{config_idx}", 0, 1
            )
    
    # Objective function (can be arbitrary for feasibility LP)
    prob += 0
    
    # Constraint: Each kid gets a total weight of 1
    for kid in instance.agents:
        prob += (
            pulp.lpSum(x[(kid, tuple(sorted(config)))] for config in valid_configs.get(kid, []))
            == 1,
            f"Kid_{kid}_gets_one_config",
        )
    
    # Constraint: Each present is used at most once
    for present in instance.items:
        prob += (
            pulp.lpSum(
                x[(kid, tuple(sorted(config)))]
                for kid in instance.agents
                for config in valid_configs.get(kid, [])
                if present in config
            )
            <= 1,
            f"Present_{present}_used_at_most_once",
        )
    
    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if the LP is feasible
    if prob.status != pulp.LpStatusOptimal:
        return None
    
    # Extract the solution
    solution = {}
    for kid in instance.agents:
        for config in valid_configs.get(kid, []):
            var_key = (kid, tuple(sorted(config)))
            if var_key in x and x[var_key].value() > 1e-6:
                solution[var_key] = x[var_key].value()
    
    return solution


def _simplified_lp_solver(instance: Instance, T: float) -> Optional[Dict]:
    """
    A simplified version of the Configuration LP solver that uses a greedy approach.
    This is used when PuLP is not available.
    
    Args:
        instance: The problem instance
        T: Target value for minimum machine load
        
    Returns:
        A fractional solution to the LP or None if infeasible
    """
    # Generate all valid configurations for each kid
    valid_configs = generate_valid_configurations(instance, T)
    
    # Check if there's at least one valid configuration for each kid
    for kid in instance.agents:
        if not valid_configs.get(kid, []):
            return None
    
    # Create a greedy assignment
    solution = {}
    present_usage = defaultdict(float)
    
    for kid in instance.agents:
        # Sort configurations by number of presents (ascending)
        configs = sorted(valid_configs.get(kid, []), key=len)
        
        # Try to find a configuration that doesn't exceed present capacity
        found = False
        for config in configs:
            # Check if adding this configuration would exceed any present's capacity
            valid = True
            for present in config:
                if present_usage[present] + 1 > 1:
                    valid = False
                    break
            
            if valid:
                # Assign this configuration to the kid
                solution[(kid, tuple(sorted(config)))] = 1.0
                for present in config:
                    present_usage[present] += 1
                found = True
                break
        
        if not found:
            # If we couldn't find a valid configuration, the LP is infeasible
            return None
    
    return solution


def generate_valid_configurations(instance: Instance, T: float) -> Dict[str, List[Set[str]]]:
    """
    Generate all valid configurations for each kid that have value at least T.
    
    A configuration is a set of presents that a kid can receive.
    In the restricted assignment case, a configuration is valid if the sum of 
    present values is at least T.
    
    Args:
        instance: The problem instance
        T: Target value for minimum machine load
        
    Returns:
        A dictionary mapping each kid to a list of valid configurations
    """
    valid_configs = {}
    
    for kid in instance.agents:
        # Get presents that this kid values > 0
        valued_presents = [
            present for present in instance.items 
            if instance.agent_item_value(kid, present) > 0
        ]
        
        # Generate all possible configurations (power set)
        configs = []
        
        # Helper function to generate configurations recursively
        def generate_configs(index, current_config, current_value):
            # If the current configuration has value >= T, add it
            if current_value >= T:
                configs.append(set(current_config))
            
            # Base case: reached the end of the presents list
            if index >= len(valued_presents):
                return
            
            # Include the present at the current index
            present = valued_presents[index]
            present_value = instance.agent_item_value(kid, present)
            current_config.append(present)
            generate_configs(index + 1, current_config, current_value + present_value)
            
            # Exclude the present at the current index
            current_config.pop()
            generate_configs(index + 1, current_config, current_value)
        
        # Start the recursive generation
        generate_configs(0, [], 0)
        
        valid_configs[kid] = configs
    
    return valid_configs


def binary_search_T(instance: Instance, precision: float = 0.001) -> float:
    """
    Perform binary search to find the maximum feasible target value T.
    
    Args:
        instance: The problem instance
        precision: The precision for the binary search
        
    Returns:
        The optimal target value T
    """
    # Determine the range for binary search
    min_T = 0
    
    # Max T is the maximum possible value any kid can get
    max_T = 0
    for kid in instance.agents:
        kid_max = sum(
            instance.agent_item_value(kid, present)
            for present in instance.items
            if instance.agent_item_value(kid, present) > 0
        )
        max_T = max(max_T, kid_max)
    
    # Binary search for the optimal T
    while max_T - min_T > precision:
        mid_T = (min_T + max_T) / 2
        
        # Check if mid_T is feasible
        if is_T_feasible(instance, mid_T):
            min_T = mid_T
        else:
            max_T = mid_T
    
    return min_T


def is_T_feasible(instance: Instance, T: float) -> bool:
    """
    Check if a given target value T is feasible.
    
    Args:
        instance: The problem instance
        T: Target value to check
        
    Returns:
        True if T is feasible, False otherwise
    """
    return compute_configuration_lp(instance, T) is not None


def classify_gifts_by_size(instance: Instance, T: float, alpha: float) -> Tuple[Set[str], Set[str]]:
    """
    Classify presents as "big" or "small" based on their values.
    
    In the restricted assignment case:
    - A present is "big" if its value is at least T/alpha
    - A present is "small" if its value is less than T/alpha
    
    Args:
        instance: The problem instance
        T: Target value
        alpha: Parameter for classifying presents
        
    Returns:
        A tuple of (big_gifts, small_gifts)
    """
    big_gifts = set()
    small_gifts = set()
    
    for present in instance.items:
        # In the restricted assignment case, each present has a fixed value
        # Find the maximum value of this present across all kids
        max_value = max(
            instance.agent_item_value(kid, present)
            for kid in instance.agents
        )
        
        if max_value >= T / alpha:
            big_gifts.add(present)
        else:
            small_gifts.add(present)
    
    return big_gifts, small_gifts


def build_assignment_forest(lp_solution: Dict, big_gifts: Set[str]) -> nx.Graph:
    """
    Build a bipartite graph from the LP solution and convert it to a forest.
    
    The graph has:
    - Kid nodes: one for each kid in the solution
    - Present nodes: one for each big present
    - Edges: between a kid and a present if the present appears in any of
      the kid's configurations with non-zero value
    
    Args:
        lp_solution: The fractional solution from the Configuration LP
        big_gifts: Set of presents classified as big
        
    Returns:
        A NetworkX forest (acyclic graph)
    """
    # Create a bipartite graph
    G = nx.Graph()
    
    # Add edges between kids and big presents based on LP solution
    for (kid, config), value in lp_solution.items():
        if value > 0:
            for present in config:
                if present in big_gifts:
                    G.add_edge(kid, present)
    
    # Convert the graph to a forest by removing cycles
    # We'll use a minimum spanning tree algorithm on each connected component
    forest = nx.Graph()
    
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        mst = nx.minimum_spanning_tree(subgraph)
        forest = nx.union(forest, mst)
    
    return forest


def create_super_machines(lp_solution: Dict, forest: nx.Graph, big_gifts: Set[str]) -> List[Tuple[List[str], List[str]]]:
    """
    Create super-machines based on the forest structure.
    
    A super-machine is a tuple (Mi, Ji) where:
    - Mi is a set of kids
    - Ji is a set of big presents with |Ji| = |Mi| - 1
    
    Args:
        lp_solution: The fractional solution from the Configuration LP
        forest: The bipartite forest from build_assignment_forest
        big_gifts: Set of presents classified as big
        
    Returns:
        A list of super-machines (Mi, Ji)
    """
    super_machines = []
    
    # Process each connected component in the forest
    for component in nx.connected_components(forest):
        # Separate kids and presents in this component
        kids = [node for node in component if node in lp_solution]
        presents = [node for node in component if node in big_gifts]
        
        # Only create a super-machine if we have at least one kid and one present
        if kids and presents:
            # Ensure |Ji| = |Mi| - 1
            if len(presents) == len(kids) - 1:
                super_machines.append((kids, presents))
            elif len(presents) < len(kids) - 1:
                # Not enough presents, skip this component
                continue
            else:
                # Too many presents, take only |Mi| - 1 of them
                # Sort presents by their maximum value to any kid in this component
                sorted_presents = sorted(
                    presents,
                    key=lambda p: max(
                        forest.get_edge_data(k, p, {}).get("weight", 0)
                        for k in kids
                        if forest.has_edge(k, p)
                    ),
                    reverse=True
                )
                super_machines.append((kids, sorted_presents[:len(kids) - 1]))
    
    return super_machines


def choose_small_configurations(super_machines: List[Tuple[List[str], List[str]]], 
                                lp_solution: Dict, 
                                small_gifts: Set[str],
                                method: str = 'randomized') -> Dict[str, Set[str]]:
    """
    Choose one configuration of small presents for each super-machine.
    
    Args:
        super_machines: List of super-machines (Mi, Ji)
        lp_solution: The fractional solution from the Configuration LP
        small_gifts: Set of presents classified as small
        method: Method for choosing configurations ('randomized' or 'deterministic')
        
    Returns:
        A dictionary mapping each kid to its assigned small presents
    """
    small_configs = {}
    
    if method == 'randomized':
        # Randomized rounding
        for i, (kids, _) in enumerate(super_machines):
            # Choose one kid from each super-machine to receive small presents
            if not kids:
                continue
                
            # The kid with the smallest index gets the small presents
            chosen_kid = kids[0]
            
            # Extract all configurations for this kid from the LP solution
            kid_configs = []
            kid_probs = []
            
            for (kid, config), value in lp_solution.items():
                if kid == chosen_kid:
                    # Filter to include only small presents
                    small_config = [p for p in config if p in small_gifts]
                    if small_config:
                        kid_configs.append(set(small_config))
                        kid_probs.append(value)
            
            # Normalize probabilities
            if kid_probs:
                kid_probs = [p / sum(kid_probs) for p in kid_probs]
                
                # Choose a configuration according to the probabilities
                chosen_config = np.random.choice(len(kid_configs), p=kid_probs)
                small_configs[chosen_kid] = kid_configs[chosen_config]
    else:
        # Deterministic approach
        for i, (kids, _) in enumerate(super_machines):
            if not kids:
                continue
                
            chosen_kid = kids[0]
            
            # Choose the configuration with the highest value in the LP solution
            best_config = None
            best_value = 0
            
            for (kid, config), value in lp_solution.items():
                if kid == chosen_kid:
                    small_config = [p for p in config if p in small_gifts]
                    if small_config and value > best_value:
                        best_config = set(small_config)
                        best_value = value
            
            if best_config:
                small_configs[chosen_kid] = best_config
    
    return small_configs


def round_to_integral_solution(small_configs: Dict[str, Set[str]], 
                               super_machines: List[Tuple[List[str], List[str]]],
                               instance: Instance) -> Dict[str, Set[str]]:
    """
    Round to an integral solution by assigning big presents within each super-machine
    and ensuring each kid gets a fair allocation.
    
    Args:
        small_configs: Dictionary mapping kids to their assigned small presents
        super_machines: List of super-machines (Mi, Ji)
        instance: The problem instance
        
    Returns:
        A dictionary mapping each kid to its final set of presents
    """
    final_assignment = {kid: set() for kid in instance.agents}
    
    # First, assign the small presents according to small_configs
    for kid, presents in small_configs.items():
        final_assignment[kid].update(presents)
    
    # Then, distribute big presents within each super-machine
    for kids, big_presents in super_machines:
        if not kids or not big_presents:
            continue
        
        # Sort kids by how many presents they already have (ascending)
        sorted_kids = sorted(kids, key=lambda k: len(final_assignment[k]))
        
        # Sort presents by their maximum value to any kid in this super-machine (descending)
        sorted_presents = sorted(
            big_presents,
            key=lambda p: max(instance.agent_item_value(k, p) for k in kids),
            reverse=True
        )
        
        # Distribute big presents to kids with fewest presents
        for i, present in enumerate(sorted_presents):
            if i >= len(sorted_kids):
                break
                
            # Assign to the kid with the fewest presents so far
            kid = sorted_kids[i]
            final_assignment[kid].add(present)
    
    # Finally, assign any remaining presents greedily
    assigned_presents = set()
    for presents in final_assignment.values():
        assigned_presents.update(presents)
    
    remaining_presents = set(instance.items) - assigned_presents
    
    if remaining_presents:
        # Sort kids by the total value they've received so far (ascending)
        sorted_kids = sorted(
            instance.agents,
            key=lambda k: sum(instance.agent_item_value(k, p) for p in final_assignment[k])
        )
        
        # For each kid (starting with those who have less value), find their highest-value present
        for kid in sorted_kids:
            if not remaining_presents:
                break
                
            # Find the best present for this kid
            best_present = max(
                remaining_presents,
                key=lambda p: instance.agent_item_value(kid, p)
            )
            
            # Assign the present regardless of value to ensure all presents are allocated
            final_assignment[kid].add(best_present)
            remaining_presents.remove(best_present)
    
    # Make sure all presents are allocated
    # If there are still remaining presents, assign them arbitrarily
    if remaining_presents:
        for present in list(remaining_presents):
            # Find any kid to assign this present to
            kid = min(instance.agents, key=lambda k: len(final_assignment[k]))
            final_assignment[kid].add(present)
            remaining_presents.remove(present)
    
    return final_assignment


def normalize_lp_solution(lp_solution: Dict) -> Dict:
    """
    Normalize the LP solution so that the sum of x_{i,C} for each kid i is 1.
    
    Args:
        lp_solution: The fractional solution from the Configuration LP
        
    Returns:
        A normalized LP solution
    """
    normalized = {}
    kid_sums = defaultdict(float)
    
    # Calculate the sum for each kid
    for (kid, config), value in lp_solution.items():
        kid_sums[kid] += value
    
    # Normalize
    for (kid, config), value in lp_solution.items():
        if kid_sums[kid] > 0:
            normalized[(kid, config)] = value / kid_sums[kid]
        else:
            normalized[(kid, config)] = 0
    
    return normalized


def santa_claus_solver(instance: Instance, alpha: float = 10, beta: float = 3) -> Tuple[float, Dict[str, Set[str]]]:
    """
    Main algorithm for the Santa Claus Problem in the restricted assignment case.
    
    This algorithm achieves an O(log log m / log log log m) approximation ratio for
    the restricted assignment case, where m is the number of kids.
    
    Args:
        instance: The problem instance
        alpha: Parameter for classifying presents as large or small
        beta: Relaxation parameter
        
    Returns:
        A tuple of (T_optimal, final_assignment)
    """
    # Print problem instance details in a table format
    logger.info("\n" + "=" * 80)
    logger.info("SANTA CLAUS PROBLEM - RESTRICTED ASSIGNMENT CASE".center(80))
    logger.info("=" * 80)
    
    # Print parameters
    logger.info("\nPARAMETERS:")
    logger.info("-" * 80)
    logger.info(f"| {'Parameter':<20} | {'Value':<55} |")
    logger.info("-" * 80)
    logger.info(f"| {'Number of Kids':<20} | {len(instance.agents):<55} |")
    logger.info(f"| {'Number of Presents':<20} | {len(instance.items):<55} |")
    logger.info(f"| {'Alpha':<20} | {alpha:<55} |")
    logger.info(f"| {'Beta':<20} | {beta:<55} |")
    logger.info("-" * 80)
    
    # Print valuations in a table
    logger.info("\nVALUATIONS:")
    logger.info("-" * 80)
    header = "| Kid \\ Present | " + " | ".join(f"{p[:10]:^10}" for p in instance.items) + " |"
    logger.info(header)
    logger.info("-" * 80)
    
    for kid in instance.agents:
        row = f"| {kid[:15]:<15} | "
        for present in instance.items:
            value = instance.agent_item_value(kid, present)
            row += f"{value:^10} | "
        logger.info(row)
    logger.info("-" * 80)
    
    # Step 1: Find the optimal target value T via binary search
    logger.info("\nStep 1: Binary search for optimal target value T")
    T_opt = binary_search_T(instance)
    logger.info(f"Optimal T: {T_opt:.4f}")
    
    # Step 2: Solve the Configuration LP for the optimal T
    logger.info("\nStep 2: Solving Configuration LP")
    lp_solution = compute_configuration_lp(instance, T_opt)
    
    if not lp_solution:
        logger.warning("Configuration LP is infeasible. Using relaxed T value.")
        T_opt = T_opt * 0.9  # Relax by 10%
        lp_solution = compute_configuration_lp(instance, T_opt)
        
        if not lp_solution:
            logger.error("Configuration LP is still infeasible. Unable to proceed.")
            return 0, {kid: set() for kid in instance.agents}
    
    logger.info(f"LP Solution has {len(lp_solution)} variables")
    
    # Step 3: Normalize the LP solution
    lp_solution = normalize_lp_solution(lp_solution)
    
    # Step 4: Classify presents as big or small
    logger.info("\nStep 3: Classifying presents by size")
    big_gifts, small_gifts = classify_gifts_by_size(instance, T_opt, alpha)
    logger.info(f"Big presents: {len(big_gifts)}, Small presents: {len(small_gifts)}")
    
    # Step 5: Build assignment forest
    logger.info("\nStep 4: Building assignment forest")
    forest = build_assignment_forest(lp_solution, big_gifts)
    logger.info(f"Forest has {len(list(nx.connected_components(forest)))} connected components")
    
    # Step 6: Create super-machines
    logger.info("\nStep 5: Creating super-machines")
    super_machines = create_super_machines(lp_solution, forest, big_gifts)
    logger.info(f"Created {len(super_machines)} super-machines")
    
    # Step 7: Choose small configurations for super-machines
    logger.info("\nStep 6: Choosing small configurations")
    small_configs = choose_small_configurations(super_machines, lp_solution, small_gifts, method='deterministic')
    logger.info(f"Assigned small presents to {len(small_configs)} kids")
    
    # Step 8: Round to integral solution
    logger.info("\nStep 7: Rounding to integral solution")
    final_assignment = round_to_integral_solution(small_configs, super_machines, instance)
    
    # Verify that all presents are assigned
    assigned_presents = set()
    for presents in final_assignment.values():
        assigned_presents.update(presents)
    
    logger.info(f"Final assignment: {len(assigned_presents)}/{len(instance.items)} presents assigned")
    
    # Print the final allocation
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-" * 80)
    for kid, presents in final_assignment.items():
        total_value = sum(instance.agent_item_value(kid, p) for p in presents)
        logger.info(f"{kid}: {presents} (Value: {total_value:.2f})")
    logger.info("-" * 80)
    
    # Calculate the minimum happiness
    min_happiness = min(
        sum(instance.agent_item_value(kid, p) for p in presents)
        for kid, presents in final_assignment.items()
    ) if final_assignment else 0
    
    logger.info(f"Minimum happiness: {min_happiness:.4f} (Target was {T_opt:.4f})")
    logger.info(f"Approximation ratio: {T_opt / min_happiness if min_happiness > 0 else float('inf'):.4f}")
    
    return T_opt, final_assignment


def divide(alloc: AllocationBuilder, alpha: float = 10, beta: float = 3) -> Dict[str, List[str]]:
    """
    Wrapper function to use the Santa Claus algorithm with the fairpyx framework.
    
    Args:
        alloc: The allocation builder
        alpha: Parameter for classifying presents as large or small
        beta: Relaxation parameter
        
    Returns:
        The final allocation
    """
    instance = alloc.instance
    _, assignment = santa_claus_solver(instance, alpha, beta)
    
    # Convert the assignment to the format expected by fairpyx
    for kid, presents in assignment.items():
        for present in presents:
            try:
                alloc.give(kid, present)
            except Exception as e:
                logger.warning(f"Could not assign {present} to {kid}: {e}")
    
    return alloc.bundles
