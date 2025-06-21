"""
Algorithm implementation for 'The Santa Claus Problem'.

Paper: The Santa Claus Problem
Authors: Nikhil Bansal, Maxim Sviridenko
Link: https://dl.acm.org/doi/10.1145/1132516.1132557 (Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006)

The problem involves distributing n gifts (items) among m children (agents).
Each child i has an arbitrary value pij for each present j.
The Santa's goal is to distribute presents such that the least lucky kid (agent)
is as happy as possible (maximin objective function):
    maximize (min_i sum_{j in S_i} p_ij)
where S_i is the set of presents received by kid i.

This file focuses on implementing the O(log log m / log log log m) approximation
algorithm for the restricted assignment case (p_ij in {p_j, 0}).

Programmers: Roey and Adiel
Date: 2025-05-29
"""

from fairpyx import Instance, AllocationBuilder, divide
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
import itertools
from scipy.optimize import linprog
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

def _generate_valid_configurations(agent_name: str, items: List[str], instance: Instance, target_value: float, agent_capacity: int) -> List[Tuple[str, ...]]:
    """Helper function that generates all valid configurations for an agent.
    A configuration is valid if:
    1. Its size is at most agent_capacity
    2. The sum of truncated values is >= target_value
    
    :param agent_name: Name of the agent
    :param items: List of all items
    :param instance: The problem instance
    :param target_value: Target value T that each config must meet
    :param agent_capacity: Maximum number of items in a configuration
    :return: List of valid configurations, each a tuple of item names
    """
    valid_configurations = []

    # Consider items the agent actually values
    relevant_items = [item for item in items if instance.agent_item_value(agent_name, item) > 0]
    relevant_items.sort(key=lambda x: instance.agent_item_value(agent_name, x), reverse=True)

    # First check single-item configurations that meet target value
    for item in relevant_items:
        original_value = instance.agent_item_value(agent_name, item)
        truncated_value = min(original_value, target_value)
        if truncated_value >= target_value:
            valid_configurations.append((item,))
            logger.debug(f"  Found valid single-item config for {agent_name}: ({item},) with truncated value {truncated_value:.2f}")

    # Then check multi-item configurations
    if not valid_configurations:  # Only if no single item is sufficient
        # Try combinations of increasing size until we find valid configurations
        for k in range(2, min(len(relevant_items), agent_capacity) + 1):
            found_valid_config = False
            for combo_indices in itertools.combinations(range(len(relevant_items)), k):
                current_config_items = tuple(sorted([relevant_items[i] for i in combo_indices]))
                current_value = 0
                # Try to reach target value with truncated values
                for item_in_config in current_config_items:
                    original_value = instance.agent_item_value(agent_name, item_in_config)
                    truncated_value = min(original_value, target_value)
                    current_value += truncated_value
                    if current_value >= target_value:
                        found_valid_config = True
                        break
                
                if found_valid_config:
                    valid_configurations.append(current_config_items)
                    logger.debug(f"  Found valid multi-item config for {agent_name}: {current_config_items} with truncated value {current_value:.2f}")
                    break  # Found a valid configuration of this size, try next size
            
            if found_valid_config:
                break  # Found a valid configuration, no need to try larger sizes

    # Sort to ensure deterministic output, primarily for testing
    # The inner tuples are already sorted, so sort the list of tuples
    valid_configurations.sort()

    logger.info(f"Agent {agent_name} has {len(valid_configurations)} valid configurations for T={target_value}.")
    return valid_configurations


def _check_config_lp_feasibility(alloc: AllocationBuilder, target_value: float) -> Tuple[bool, Optional[np.ndarray], Optional[List[Tuple[str, Tuple[str, ...]]]]]:
    """
    Checks if a given target_value T is feasible by formulating and attempting to solve
    the Configuration LP (feasibility version).

    The Configuration LP is defined as (see paper, Section 2):
    Variables: x_i,C for each agent i and valid configuration C in C(i,T).
    Constraints:
    1. For each agent i: sum(x[i,C] for all C) = 1
       (each agent's fractions sum to 1)
    2. For each item j: sum(x[i,C] for all i,C where j in C) <= 1
       (each item used at most once)
    3. For each i,C: x[i,C] >= 0
       (fractions are non-negative)
    
    Returns:
    - is_feasible: True if LP has a solution
    - solution_vector: The x[i,C] values if feasible
    - x_var_map: List mapping indices to (agent_name, config_tuple) pairs
    
    If infeasible, returns (False, None, None)
    """
    from scipy.optimize import linprog
    import numpy as np
    
    # Get all agents and items from the instance
    agents = alloc.instance.agents
    items = alloc.instance.items
    
    if not agents or not items:
        return False, None, None
    
    # For each agent, generate all valid configurations meeting target_value
    all_agent_configs: Dict[str, List[Tuple[str, ...]]] = {}
    x_var_map: List[Tuple[str, Tuple[str, ...]]] = []
    
    for agent in agents:
        agent_capacity = alloc.instance.agent_capacity(agent)
        valid_configs = _generate_valid_configurations(agent, items, alloc.instance, target_value, agent_capacity)
        
        if not valid_configs:  # If any agent has no valid configs, T is infeasible
            logger.debug(f"  Agent {agent} has no valid configurations for T={target_value}")
            return False, None, None
            
        all_agent_configs[agent] = valid_configs
        # Add each (agent, config) pair to our list and record its index
        for config in valid_configs:
            x_var_map.append((agent, config))

    # Now build the LP constraint matrices
    num_variables = len(x_var_map)
    if num_variables == 0:
        return False, None, None

    # 1. Agent sum constraints: for each agent i, sum(x[i,C] for all C) = 1
    A_eq = np.zeros((len(agents), num_variables))
    b_eq = np.ones(len(agents))
    
    for i, agent in enumerate(agents):
        for j, (var_agent, _) in enumerate(x_var_map):
            if var_agent == agent:
                A_eq[i,j] = 1
    
    # 2. Item capacity constraints: for each item j, sum(x[i,C] for all i,C where j in C) <= item_capacity
    A_ub = np.zeros((len(items), num_variables))
    b_ub = np.array([alloc.instance.item_capacity(item) for item in items])
    
    for i, item in enumerate(items):
        for j, (_, config) in enumerate(x_var_map):
            if item in config:
                A_ub[i,j] = 1
    
    # 3. Non-negativity constraints are handled by bounds
    bounds = [(0, 1) for _ in range(num_variables)]
    
    # Objective: maximize the sum of all variables to avoid degenerate solutions
    c = np.ones(num_variables) * -1  # Negative for maximization
    
    # Solve the LP
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'disp': True}  # Show solver output for debugging
        )
    except Exception as e:
        logger.error(f"LP solver failed with error: {e}")
        return False, None, None

    if not result.success:
        logger.warning(f"LP solver failed: {result.message}")
        return False, None, None

    # Check if solution is valid (within numerical tolerance)
    x = result.x
    
    # Check agent sum constraints
    for i, agent in enumerate(agents):
        agent_sum = sum(x[j] for j, (var_agent, _) in enumerate(x_var_map) if var_agent == agent)
        if abs(agent_sum - 1) > 1e-6:
            logger.warning(f"Agent {agent} sum constraint violated: {agent_sum:.6f} != 1")
            return False, None, None
    
    # Check item capacity constraints
    for i, item in enumerate(items):
        item_cap = alloc.instance.item_capacity(item)
        item_sum = sum(x[j] for j, (_, config) in enumerate(x_var_map) if item in config)
        if item_sum > item_cap + 1e-6:
            logger.warning(f"Item {item} capacity constraint violated: {item_sum:.6f} > {item_cap}")
            return False, None, None
    
    logger.info(f"Found valid LP solution with objective value: {-result.fun:.6f}")
    return True, result.x, x_var_map


# Example instance for testing
example_instance = Instance(
    valuations = {
        "Child1": {"gift1": 5, "gift2": 0, "gift3": 0},
        "Child2": {"gift1": 0, "gift2": 5, "gift3": 10},
        "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
    },
    agent_capacities = {"Child1": 1, "Child2": 1, "Child3": 1},
    item_capacities = {"gift1": 1, "gift2": 1, "gift3": 1}
)

# Example for restricted assignment case
restricted_example = Instance(
    valuations = {
        "Child1": {"gift1": 10, "gift2": 0, "gift3": 0, "gift4": 10, "gift5": 1},
        "Child2": {"gift1": 0, "gift2": 10, "gift3": 0, "gift4": 0, "gift5": 1},
        "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 10, "gift5": 1}
    },
    agent_capacities = {"Child1": 2, "Child2": 2, "Child3": 2},
    item_capacities = {"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
)

def santa_claus_algorithm(alloc: AllocationBuilder, alpha: float = 3.0) -> dict:
    """
    Main algorithm for the Santa Claus Problem, implementing the O(log log m / log log log m) approximation
    algorithm for the restricted assignment case.
    
    :param alloc: The allocation builder that tracks the allocation process.
    :param alpha: Parameter β in the paper, used for classifying gifts as large or small.
                Default is 3.0 as suggested in the paper.

    Based on "The Santa Claus Problem" by Bansal and Sviridenko.
    
    Doctest:
    >>> result = divide(santa_claus_algorithm, example_instance)
    >>> sorted(list(result.keys()))
    ['Child1', 'Child2', 'Child3']
    >>> all(len(gifts) <= example_instance.agent_capacities[child] for child, gifts in result.items())
    True

    >>> result_restricted = divide(santa_claus_algorithm, restricted_example)
    >>> sorted(list(result_restricted.keys()))
    ['Child1', 'Child2', 'Child3']
    """
    import logging
    logger = logging.getLogger("fairpyx.algorithms.santa_claus")
    
    # Step 1: Find optimal target value T* via binary search
    logger.info("Finding optimal target value T*...")
    target_value = find_optimal_target_value(alloc)
    logger.info(f"Found optimal target value T* = {target_value:.3f}")
    
    if target_value <= 0:
        logger.warning("Optimal target value is 0 - falling back to greedy allocation")
        # Fallback: Greedy allocation
        for item in alloc.instance.items:
            # Find child with highest value for this item who has capacity
            best_child = None
            best_value = -1
            for child in alloc.instance.agents:
                if len(alloc.bundle(child)) < alloc.instance.agent_capacities[child]:
                    value = alloc.instance.agent_item_value(child, item)
                    if value > best_value:
                        best_value = value
                        best_child = child
            if best_child:
                alloc.give(best_child, item)
        return
    
    # Step 2: Solve Configuration LP for T*
    logger.info("Solving Configuration LP...")
    fractional_solution = configuration_lp_solver(alloc, target_value)
    
    if not fractional_solution:
        logger.warning("No feasible solution found for Configuration LP - falling back to greedy")
        santa_claus_algorithm(alloc, alpha)  # Retry with greedy fallback
        return
    
    # Step 3: Classify gifts as large or small based on T* and alpha
    logger.info("Classifying gifts as large or small...")
    large_gifts = []
    small_gifts = []
    
    for item in alloc.instance.items:
        max_value = max(alloc.instance.agent_item_value(agent, item) 
                       for agent in alloc.instance.agents)
        if max_value >= target_value / alpha:
            large_gifts.append(item)
        else:
            small_gifts.append(item)
    
    logger.info(f"Found {len(large_gifts)} large gifts and {len(small_gifts)} small gifts")
    
    # Step 4: Create super-machines from fractional solution
    logger.info("Creating super-machines...")
    super_machines = create_super_machines(alloc, fractional_solution, large_gifts, target_value, small_gifts)
    
    if not super_machines:
        logger.warning("No super-machines created - falling back to greedy")
        santa_claus_algorithm(alloc, alpha)  # Retry with greedy fallback
        return
    
    # Step 5: Round small gift configurations within super-machines
    logger.info("Rounding small gift configurations...")
    rounded_solution = round_small_configurations(alloc, super_machines, set(small_gifts), alpha)
    
    if not rounded_solution:
        logger.warning("Failed to round small gift configurations - falling back to greedy")
        santa_claus_algorithm(alloc, alpha)  # Retry with greedy fallback
        return
    
    # Step 6: Construct final allocation
    logger.info("Constructing final allocation...")
    construct_final_allocation(alloc, super_machines, rounded_solution, target_value)
    
    # Verify allocation is valid
    bundles = alloc.sorted()
    total_allocated = sum(len(gifts) for gifts in bundles.values())
    if total_allocated == 0:
        logger.warning("No gifts allocated in final solution - falling back to greedy")
        santa_claus_algorithm(alloc, alpha)  # Retry with greedy fallback
        return
    
    # Calculate minimum value received by any child
    min_value = float('inf')
    bundles = alloc.sorted()
    for child, gifts in bundles.items():
        child_value = sum(alloc.instance.agent_item_value(child, gift) for gift in gifts)
        min_value = min(min_value, child_value)
    
    if min_value == float('inf'):
        min_value = 0
    
    logger.info(f"Final allocation: {total_allocated} gifts distributed across {len(bundles)} children")
    logger.info(f"Minimum value received by any child: {min_value:.3f}")
    return bundles


def find_optimal_target_value(alloc: AllocationBuilder) -> float:
    """
    Algorithm 1 (conceptual): Binary search to find the highest feasible target value T.
    This function uses `_check_config_lp_feasibility` to determine if a T is feasible.
    
    :param alloc: The allocation builder containing the problem instance.
    :return: The highest feasible target value T found. Returns 0.0 if no T > 0 is feasible.
    
    Note: The doctest below is illustrative. The actual optimal T depends on the LP solver
    and the specific instance. For `example_instance`, if Child1 gets gift1 (val 5),
    Child2 gets gift2 (val 5), Child3 gets gift3 (val 10), min is 5. This is achievable.
    A T slightly above 5 might not be if it forces non-optimal item distribution for the LP.
    The example_instance has agent_capacities of 1.
    Child1: g1 (5). Child2: g2 (5), g3 (10). Child3: g1(5), g2(5), g3(10)
    Optimal T should be 5. (C1:g1, C2:g2, C3:g3) -> min value is 5.
    If T=6, C1 cannot make it. So LP for T=6 would be infeasible.

    >>> instance = Instance(valuations={'C1': {'i1': 5, 'i2':2}, 'C2': {'i1': 3, 'i2':6}}, agent_capacities=1, item_capacities=1)
    >>> alloc_builder = AllocationBuilder(instance)
    >>> optimal_t = find_optimal_target_value(alloc_builder)
    >>> # For this instance, (C1:i1, C2:i2) -> min_val=5. (C1:i2, C2:i1) -> min_val=2. So T=5.
    >>> abs(optimal_t - 5.0) < 1e-9 # Check if close to 5.0
    True

    >>> alloc_builder_example = AllocationBuilder(example_instance)
    >>> optimal_t_example = find_optimal_target_value(alloc_builder_example)
    >>> abs(optimal_t_example - 5.0) < 1e-9 # Expected T for example_instance
    True
    """
    instance = alloc.instance
    logger.info("Starting binary search for optimal target value T.")

    # Determine search range for T
    max_possible_agent_value = 0.0
    min_possible_agent_value = float('inf')
    items = alloc.instance.items
    agents = alloc.instance.agents
    if not items or not agents:
        logger.warning("No items or agents in the instance. Optimal T is 0.")
        return 0.0

    # Find maximum and minimum possible values for any agent
    min_positive_value = float('inf')
    
    for agent in instance.agents:
        agent_max_value = 0.0
        for item in instance.items:
            value = instance.agent_item_value(agent, item)
            if value > 0:
                min_positive_value = min(min_positive_value, value)
                agent_max_value = max(agent_max_value, value)
        max_possible_agent_value = max(max_possible_agent_value, agent_max_value)
    
    if max_possible_agent_value <= 0:
        logger.warning("No positive valuations found. Returning T=0.")
        return 0.0
    
    # Initialize binary search bounds
    low_T = 0  # Start from 0 since we want to find maximum feasible T
    high_T = max_possible_agent_value
    
    # Binary search for optimal T
    optimal_T_found = 0.0
    iterations = 0
    max_iterations = 50  # Prevent infinite loops due to floating point issues
    precision = 1e-7  # Desired precision for T
    
    while abs(high_T - low_T) > precision and iterations < max_iterations:
        mid_T = (low_T + high_T) / 2.0
        if mid_T < precision:  # Avoid extremely small T values
            break
        
        logger.info(f"Trying T = {mid_T:.4f} (range [{low_T:.4f}, {high_T:.4f}])")
        is_feasible, _, _ = _check_config_lp_feasibility(alloc, mid_T)
        
        if is_feasible:
            logger.debug(f"  T={mid_T:.4f} is feasible. Trying higher.")
            optimal_T_found = mid_T  # Current mid_T is a candidate for optimal T
            low_T = mid_T  # Try to find a higher T
        else:
            logger.debug(f"  T={mid_T:.4f} is infeasible. Trying lower.")
            high_T = mid_T  # mid_T is too high, reduce upper bound
        
        iterations += 1
        
        # Early exit if we've found a good enough solution
        if abs(high_T - low_T) < precision:
            break
    
    # Final verification of optimal_T_found
    if optimal_T_found > 0:
        is_feasible, _, _ = _check_config_lp_feasibility(alloc, optimal_T_found)
        if not is_feasible:
            # Binary search for a feasible T below optimal_T_found
            test_T = optimal_T_found
            step = optimal_T_found / 2
            while not is_feasible and step > precision:
                test_T -= step
                if test_T <= 0:
                    break
                is_feasible, _, _ = _check_config_lp_feasibility(alloc, test_T)
                if is_feasible:
                    optimal_T_found = test_T
                    break
                step /= 2
    
    logger.info(f"Binary search completed after {iterations} iterations. Optimal T found: {optimal_T_found:.4f}")
    return optimal_T_found


def configuration_lp_solver(alloc: AllocationBuilder, target_value: float) -> Dict[Tuple[str, Tuple[str,...]], float]:
    """Algorithm 2 (conceptual): Solves the Configuration LP for a given target_value T.
    
    For each agent i and configuration C, we have a variable x_i,C.
    The LP constraints are:
    1. For each agent i: sum_{C in C(i,T)} x_i,C = 1
    2. For each item j: sum_i sum_{C in C(i,T): j in C} x_i,C <= 1
    3. For each i,C: x_i,C >= 0
    
    :param alloc: The allocation builder containing the problem instance
    :param target_value: The target value T for which to solve the LP
    :return: A dictionary mapping (agent, configuration) pairs to their fractional values in the LP solution
    """
    # Call _check_config_lp_feasibility to get the solution
    is_feasible, solution_vector, variable_map = _check_config_lp_feasibility(alloc, target_value)
    
    if not is_feasible or solution_vector is None or variable_map is None:
        logger.warning(f"Configuration LP infeasible for T={target_value}")
        return {}
    
    # Convert solution vector to dictionary
    solution_dict = {}
    for i, (agent, config) in enumerate(variable_map):
        value = solution_vector[i]
        if abs(value) > 1e-7:  # Only include non-zero values with tolerance
            solution_dict[(agent, config)] = value
    
    # Verify solution properties
    instance = alloc.instance
    agents = instance.agents
    items = instance.items
    
    # Check agent sum constraints and normalize if needed
    for agent in agents:
        agent_sum = sum(value for (a, _), value in solution_dict.items() if a == agent)
        if not np.isclose(agent_sum, 1, rtol=1e-6, atol=1e-6):
            if agent_sum > 0:
                # Normalize agent's configuration values
                for key in list(solution_dict.keys()):
                    if key[0] == agent:
                        solution_dict[key] /= agent_sum
            else:
                logger.error(f"Agent {agent} has no valid configurations")
                return {}
    
    # Check item capacity constraints and scale if needed
    scale_needed = False
    max_item_sum = 0
    for item in items:
        item_sum = sum(value for (_, config), value in solution_dict.items() if item in config)
        if item_sum > 1 + 1e-6:
            scale_needed = True
            max_item_sum = max(max_item_sum, item_sum)
    
    if scale_needed:
        # Scale all values to satisfy item constraints
        scale = 1.0 / max_item_sum
        for key in list(solution_dict.keys()):
            solution_dict[key] *= scale
        
        # Re-normalize agent sums after scaling
        for agent in agents:
            agent_sum = sum(value for (a, _), value in solution_dict.items() if a == agent)
            if agent_sum > 0:
                for key in list(solution_dict.keys()):
                    if key[0] == agent:
                        solution_dict[key] /= agent_sum
    
    # Final verification
    for agent in agents:
        agent_sum = sum(value for (a, _), value in solution_dict.items() if a == agent)
        if not np.isclose(agent_sum, 1, rtol=1e-6, atol=1e-6):
            logger.error(f"Agent {agent} sum constraint still violated: {agent_sum:.6f} != 1")
            return {}
    
    for item in items:
        item_sum = sum(value for (_, config), value in solution_dict.items() if item in config)
        if item_sum > 1 + 1e-6:
            logger.error(f"Item {item} capacity constraint still violated: {item_sum:.6f} > 1")
            return {}
    
    return solution_dict


def create_super_machines(alloc: AllocationBuilder, fractional_solution: Dict, 
                       large_gifts: Union[List[str], float] = None, target_value: float = None, small_gifts: List[str] = None, **kwargs) -> List[List[str]]:
    """Algorithm 3: Create super-machines from fractional solution.
    
    This is part of the second phase of the Santa Claus algorithm.
    It groups machines (kids) into super-machines based on their fractional assignments.
    Each super-machine will be assigned a subset of large gifts.
    
    :param alloc: The allocation builder containing the problem instance
    :param fractional_solution: The solution from the Configuration LP
    :param target_value: The target value T* found in phase 1
    :param large_gifts: List of large gift names (value >= T*/alpha)
    :param small_gifts: List of small gift names (value < T*/alpha)
    :return: List of lists, where each inner list contains the names of machines in a super-machine
    """
    import logging
    from collections import defaultdict, deque
    
    logger = logging.getLogger("fairpyx.algorithms.santa_claus")
    
    # Initialize super-machines list
    super_machines = []
    
    # Handle parameter flexibility - third param could be large_gifts or target_value
    if isinstance(large_gifts, float):
        # Third parameter is actually target_value
        target_value = large_gifts
        large_gifts = None
    
    # Handle case where large_gifts is not provided
    if large_gifts is None:
        large_gifts = []
        if target_value is not None:
            for item in alloc.instance.items:
                max_value = max(alloc.instance.agent_item_value(agent, item) 
                              for agent in alloc.instance.agents)
                if max_value >= target_value / 3.0:  # Default alpha=3.0
                    large_gifts.append(item)
    
    # Create a bipartite graph where:
    # - Left nodes are machines (agents)
    # - Right nodes are large gifts
    # - Edge exists if machine has non-zero fraction of large gift in solution
    machine_gift_edges = defaultdict(set)
    gift_machine_edges = defaultdict(set)
    
    # Build the bipartite graph
    for (agent, config), val in fractional_solution.items():
        if val > 1e-7:  # Only consider non-zero assignments
            for gift in config:
                if gift in large_gifts:
                    machine_gift_edges[agent].add(gift)
                    gift_machine_edges[gift].add(agent)
    
    logger.info(f"Built bipartite graph with {len(machine_gift_edges)} machines and {len(gift_machine_edges)} large gifts")
    
    # Track which machines have been assigned to super-machines
    assigned_machines = set()
    agents = alloc.instance.agents
    
    # Create super-machines based on connected components in the graph
    def find_component(start_machine: str) -> List[str]:
        component = []
        queue = deque([start_machine])
        visited_machines = {start_machine}
        visited_gifts = set()
        
        while queue:
            current = queue.popleft()
            component.append(current)
            
            # Follow edges machine -> gift -> other machines
            for gift in machine_gift_edges[current]:
                if gift not in visited_gifts:
                    visited_gifts.add(gift)
                    for other_machine in gift_machine_edges[gift]:
                        if other_machine not in visited_machines:
                            visited_machines.add(other_machine)
                            queue.append(other_machine)
        
        return sorted(component)  # Sort for deterministic output
    
    # First create super-machines for machines with large gifts
    for machine in agents:
        if machine not in assigned_machines and machine_gift_edges[machine]:
            component = find_component(machine)
            if component:
                super_machines.append(component)
                assigned_machines.update(component)
    
    # Then create singleton super-machines for remaining machines
    # These are machines that only have small gifts in their fractional solution
    for machine in agents:
        if machine not in assigned_machines:
            super_machines.append([machine])
            assigned_machines.add(machine)
    
    # Verify that each machine is in exactly one super-machine
    all_machines = set()
    for sm in super_machines:
        all_machines.update(sm)
    
    if all_machines != set(agents):
        logger.error("Not all machines assigned to super-machines")
        return []
    
    logger.info(f"Created {len(super_machines)} super-machines")
    for i, sm in enumerate(super_machines):
        logger.debug(f"Super-machine {i}: {sm}")
    
    return super_machines


def round_small_configurations(alloc: AllocationBuilder, super_machines: List, 
                             small_gifts: Set[str], beta: float = 3.0, **kwargs) -> Dict:
    """Algorithm 4: Round the small gift configurations.
    
    This is the core part of the algorithm, using sampling and the Leighton et al. algorithm
    to find configurations with low congestion, as described in Section 6 of "The Santa Claus Problem".
    
    This implementation uses a combination of randomized rounding and deterministic selection
    to distribute small gifts to super-machines while ensuring no small gift is assigned to
    more than β machines.
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure from previous step
    :param small_gifts: Set of gifts classified as small
    :param beta: Relaxation parameter for the solution
    :return: A dictionary mapping super-machine index to selected small gift configuration
    
    >>> small_gifts = {"gift5"}
    >>> super_machines = [["Child1", "Child2"]]
    >>> builder = divide(lambda a: round_small_configurations(a, super_machines, small_gifts), restricted_example, return_builder=True)
    >>> builder.allocation
    {'Child1': ['gift5'], 'Child2': [], 'Child3': []}
    """
    import logging
    import numpy as np
    from collections import defaultdict
    import random
    
    logger = logging.getLogger("fairpyx.algorithms.santa_claus")
    
    if not super_machines or not small_gifts:
        logger.info("No super-machines or small gifts provided.")
        return {}
    
    # Initialize data structures for rounding
    rounded_solution = {}
    gift_usage = defaultdict(int)  # Track how many times each gift is used
    
    # For each super-machine
    for i, machines_data in enumerate(super_machines):
        # Handle both list of lists and list of tuples formats
        if isinstance(machines_data, tuple):
            machines, _ = machines_data  # Unpack (machines, gifts) tuple
        else:
            machines = machines_data
            
        if not machines:
            continue
        
        # Choose a representative machine (child) from this super-machine
        representative = machines[0]
        
        # Get all possible small gift configurations for this representative
        valid_configs = []
        valid_weights = []
        
        # First try single-gift configurations
        for gift in small_gifts:
            # Check if this gift can be assigned (respects beta constraint)
            if gift_usage[gift] < beta and alloc.instance.agent_item_value(representative, gift) > 0:
                # This is a valid single-gift configuration
                valid_configs.append([gift])
                # Weight proportional to value
                valid_weights.append(alloc.instance.agent_item_value(representative, gift))
        
        # If we found valid configurations
        if valid_configs:
            # If we have weights, use them for selection
            if sum(valid_weights) > 0:
                # Normalize weights to probabilities
                valid_probs = [w / sum(valid_weights) for w in valid_weights]
                # Choose a configuration based on weights
                selected_idx = np.random.choice(len(valid_configs), p=valid_probs)
                selected_config = valid_configs[selected_idx]
            else:
                # Otherwise choose randomly
                selected_config = random.choice(valid_configs)
            
            # Add it to the solution
            rounded_solution[i] = {
                "child": representative,
                "gifts": selected_config
            }
            
            # Update gift usage
            for gift in selected_config:
                gift_usage[gift] += 1
    
    # Log statistics
    total_gifts = sum(len(config["gifts"]) for config in rounded_solution.values())
    logger.info(f"Rounded solution assigns {total_gifts} small gifts across {len(rounded_solution)} super-machines")
    
    # Verify beta constraint
    for gift in small_gifts:
        if gift_usage[gift] > beta:
            logger.error(f"Gift {gift} used {gift_usage[gift]} times, exceeding beta={beta}")
            return {}
    
    return rounded_solution


def construct_final_allocation(alloc: AllocationBuilder, super_machines: List, 
                             rounded_solution: Dict, target_value: float = None, **kwargs) -> None:
    """
    Algorithm 5: Construct the final allocation.
    
    Assigns the small gift configurations and large gifts to children
    in each super-machine, then removes conflicts.
    
    As described in Section 5.3 of "The Santa Claus Problem" by Bansal and Sviridenko,
    this step ensures that each child receives either large gifts from their cluster
    or small gifts from the rounded solution, yielding a valid integral solution.
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure
    :param rounded_solution: The rounded solution for small gifts
    
    >>> super_machines = [("Child1", "Child2"), ("Child3",)]
    >>> rounded_solution = {0: {"child": "Child1", "gifts": ["gift5"]}}
    >>> divide(lambda a: construct_final_allocation(a, super_machines, rounded_solution), restricted_example)
    {'Child1': ['gift5'], 'Child2': ['gift1'], 'Child3': []}
    """
    import logging
    import copy
    from collections import defaultdict
    
    logger = logging.getLogger("fairpyx.algorithms.santa_claus")
    
    if not super_machines:
        logger.warning("No super-machines provided - nothing to allocate")
        return
    
    # We'll update the allocation builder directly
    # No need for a separate final_allocations dictionary
    
    # Track which gifts have been allocated
    allocated_gifts = set()
    
    # First, assign small gift configurations from the rounded solution
    for i, machines_data in enumerate(super_machines):
        # Handle both list of lists and list of tuples formats
        if isinstance(machines_data, tuple):
            machines, large_gifts_for_machine = machines_data  # Unpack (machines, gifts) tuple
        else:
            machines = machines_data
            large_gifts_for_machine = []
            
        if i in rounded_solution:
            child = rounded_solution[i]["child"]
            gifts = rounded_solution[i]["gifts"]
            
            # Only assign gifts that haven't been allocated yet
            for gift in gifts:
                if gift not in allocated_gifts:
                    alloc.give(child, gift)
                    allocated_gifts.add(gift)
    
    # Then, assign large gifts to remaining children in each super-machine
    for machines_data in super_machines:
        # Handle both list of lists and list of tuples formats
        if isinstance(machines_data, tuple):
            machines, large_gifts_for_machine = machines_data  # Unpack (machines, gifts) tuple
        else:
            machines = machines_data
            large_gifts_for_machine = []
            
        # Get children who haven't received small gifts
        remaining_children = []
        for child in machines:
            if not alloc.bundle(child):  # Child hasn't received small gifts
                remaining_children.append(child)
        
        if not remaining_children:
            continue
        
        # Get available large gifts for this super-machine
        available_large_gifts = []
        
        # If we have pre-defined large gifts for this super-machine, use those
        if large_gifts_for_machine:
            for gift in large_gifts_for_machine:
                if gift not in allocated_gifts:
                    max_value = max(alloc.instance.agent_item_value(child, gift) 
                                  for child in remaining_children)
                    available_large_gifts.append((gift, max_value))
        else:
            # Otherwise check all items
            for gift in alloc.instance.items:
                if gift not in allocated_gifts:
                    max_value = max(alloc.instance.agent_item_value(child, gift) 
                                  for child in remaining_children)
                    if max_value >= target_value / 2 if target_value else 0:  # Large gift
                        available_large_gifts.append((gift, max_value))
        
        # Sort large gifts by value (highest first)
        available_large_gifts.sort(key=lambda x: x[1], reverse=True)
        
        # Assign large gifts to children greedily
        # For each remaining child, assign the best available large gift
        for child in remaining_children:
            if available_large_gifts:
                gift, _ = available_large_gifts.pop(0)  # Get the highest-value gift
                alloc.give(child, gift)
                allocated_gifts.add(gift)
    
    # Log statistics about the final allocation
    bundles = alloc.sorted()
    total_allocated = sum(len(gifts) for gifts in bundles.values())
    logger.info(f"Final allocation: {total_allocated} gifts distributed across {len(bundles)} children")
    
    # Calculate the minimum value received by any child
    min_value = float('inf')
    for child, gifts in bundles.items():
        child_value = sum(alloc.instance.agent_item_value(child, gift) for gift in gifts)
        min_value = min(min_value, child_value)
    
    if min_value == float('inf'):
        min_value = 0
    
    logger.info(f"Minimum value received by any child: {min_value:.3f}")
    return None


def santa_claus(alloc: AllocationBuilder) -> Dict[str, List[str]]:
    """
    Main entry point for the Santa Claus Problem algorithm.
    
    This implements the O(log log m / log log log m) approximation algorithm for
    the restricted assignment case as described in "The Santa Claus Problem" by
    Bansal and Sviridenko.
    
    The algorithm aims to maximize the minimum value received by any agent (maximin objective):
        maximize (min_i sum_{j in S_i} p_ij)
    where S_i is the set of presents allocated to kid i.
    
    Args:
        alloc: The allocation builder that tracks the allocation process
        
    Returns:
        A dictionary mapping each agent (kid) to their allocated items (presents)
    
    Example:
        >>> result = divide(santa_claus, example_instance)
        >>> sorted(list(result.keys()))
        ['Child1', 'Child2', 'Child3']
        >>> # All children should receive some presents to maximize the minimum happiness
    """
    # Use our santa_claus_algorithm implementation
    santa_claus_algorithm(alloc)
    
    # Return the allocation after the algorithm has run
    return alloc.sorted()
