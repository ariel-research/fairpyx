"""
This is a dummy algorithm, that serves as an example how to implement an algorithm.

Programmer: Erel Segal-Halevi
Since: 2024-03
"""

# The end-users of the algorithm feed the input into an "Instance" variable, which tracks the original input (agents, items and their capacities).
# But the algorithm implementation uses an "AllocationBuilder" variable, which tracks both the ongoing allocation and the remaining input (the remaining capacities of agents and items).
# The function `divide` is an adaptor - it converts an Instance to an AllocationBuilder with an empty allocation.
from fairpyx import Instance, AllocationBuilder, divide



# The `logging` facility is used both for debugging and for illustrating the steps of the algorithm.
# It can be used to automatically generate running examples or explanations.
import logging
logger = logging.getLogger(__name__)

# This example instance is used in doctests throughout this file:
example_instance = Instance(
    valuations = {"Alice": {"c1": 8, "c2": 6, "c3": 10}, "Bob": {"c1": 8, "c2": 10, "c3": 6}, "Chana": {"c1": 6, "c2": 8, "c3": 10}, "Dana": {"c1": 6, "c2": 8, "c3": 10}},
    agent_capacities = {"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3},      
    item_capacities = {"c1": 2, "c2": 3, "c3": 4},
)

def algorithm1(alloc: AllocationBuilder):
    """
    This dummy algorithm gives one item to the first agent, and all items to the second agent.
    This algorithm ignores agent and item conflicts.

    >>> divide(algorithm1, example_instance)
    {'Alice': ['c1'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': [], 'Dana': []}
    """
    logger.info("\nAlgorithm 1 starts. items %s , agents %s", alloc.remaining_item_capacities, alloc.remaining_agent_capacities)
    remaining_agents = list(alloc.remaining_agents())          # `remaining_agents` returns the list of agents with remaining capacities.
    remaining_items = list(alloc.remaining_items())
    alloc.give(remaining_agents[0], remaining_items[0])        # `give` gives the specified agent the specified item, and updates the capacities.
    alloc.give_bundle(remaining_agents[1], remaining_items)    # `give_bundle` gives the specified agent the specified set of items, and updates the capacities.
    # No need to return a value. The `divide` function returns the output.

def algorithm2(alloc: AllocationBuilder):
    """
    This is a serial dictatorship algorithm: it lets each agent in turn 
    pick all remaining items that have no agent or item conflicts.
    
    >>> divide(algorithm2, example_instance)
    {'Alice': ['c1', 'c2'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c3']}
    """
    logger.info("\nAlgorithm 2 starts. items %s , agents %s", alloc.remaining_item_capacities, alloc.remaining_agent_capacities)
    picking_order = list(alloc.remaining_agents())
    for agent in picking_order:
        bundle = list(alloc.remaining_items_for_agent(agent))    # remaining_items_for_agent returns all items that the agent can pick (i.e., all items with remaining capacities that the agent does not already have, and have no agent or item conflits).
        agent_capacity = alloc.remaining_agent_capacities[agent]
        if agent_capacity >= len(bundle):
            alloc.give_bundle(agent, bundle)
        else:
            for i in range(agent_capacity):
                alloc.give(agent, bundle[i])
        logger.info("%s picks %s", agent, bundle)

def algorithm3(alloc: AllocationBuilder):
    """
    This dummy algorithm demonstrates how to call algorithms sequentially.

    It first allocates one high-value item to each agent, and then run algorithm2.
    
    >>> divide(algorithm3, example_instance)
    {'Alice': ['c1', 'c3'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c2', 'c3']}
    """
    logger.info("\nAlgorithm 3 starts. items %s , agents %s", alloc.remaining_item_capacities, alloc.remaining_agent_capacities)

    logger.info("\nFirst phase: allocate one high-value item per agent.")
    for agent in list(alloc.remaining_agents()):
        for item in list(alloc.remaining_items_for_agent(agent)):
            value = alloc.effective_value(agent, item)    # `effective_value` returns the agent's item to the value if there is no conflict; otherwise, it returns minus infinity.
            if value >= 10:
                logger.info("%s gets %s, with value %s", agent, item, value)   
                alloc.give(agent, item)
                break

    logger.info("\nSecond phase: serial-dictatorship.")
    algorithm2(alloc)    

### MAIN PROGRAM

if __name__ == "__main__":
    # 1. Run the doctests:    
    import doctest, sys
    print("\n",doctest.testmod(), "\n")

    # 2. Run the algorithm on random instances, with logging:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    from fairpyx.adaptors import divide_random_instance
    divide_random_instance(algorithm=algorithm3, 
                           num_of_agents=30, num_of_items=10, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
                           item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
                           random_seed=1)
