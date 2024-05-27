"""
An implementation of the algorithms in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2010), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import random
from itertools import cycle
from networkx import DiGraph
import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
handler=logging.FileHandler('heterogeneous_matroid_constraints_algorithms.log',mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def envy(source: str, target: str, bundles: dict[str, set or list], val_func: callable, item_categories: dict,
         agent_category_capacities: dict):
    """
        Determine if the source agent envies the target agent's bundle.

        Parameters:
        source (str): The agent who might feel envy.
        target (str): The agent whose bundle is being evaluated.
        bundles (dict[str, set or list]): A dictionary where keys are agents and values are sets or lists of items allocated to each agent.
        val_func (callable): A function that takes an agent and an item, returning the value of that item for the agent.
        item_categories (dict): A dictionary mapping items to their categories.
        agent_category_capacities (dict): A dictionary where keys are agents and values are dictionaries mapping categories to capacities.

        Returns:
        bool: True if the source agent envies the target agent's bundle, False otherwise.

        Examples:
        >>> bundles = {'agent1': {'m1', 'm2'}, 'agent2': {'m3', 'm4'}}
        >>> item_categories = {'c1':['m1','m2','m3','m4']}
        >>> agent_category_capacities = {'agent1': {'c1': 2}, 'agent2': {'c1': 2}}
        >>> val_func = lambda agent, item: {'agent1': {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4},'agent2': {'m1': 4, 'm2': 3, 'm3': 2, 'm4': 1}}[agent][item]
        >>> envy('agent1', 'agent2', bundles, val_func, item_categories, agent_category_capacities)
        True
        """
    val = val_func
    source_bundle_val = sum(list(val(source, current_item) for current_item in bundles[source]))
    #target_bundle_val = sum(list(val(source, current_item) for current_item in bundles[target])) #old non feasible method
    copy = bundles[target].copy()
    target_bundle_val = 0
    sorted(copy, key=lambda x: val(source, x), reverse=True)  # sort target items  in the perspective of the envier
    for category in item_categories.keys():
        candidates = [item for item in copy if
                      item in item_categories[category]]  # assumes sorted in reverse (maximum comes first)
        target_bundle_val += sum(val(source, x) for x in candidates[:agent_category_capacities[source][category]])

    #print(source_bundle_val, target_bundle_val)
    return target_bundle_val > source_bundle_val


# def categorization_friendly_picking_sequence(alloc: AllocationBuilder, agent_order: list, item_categories: dict,
#                                              agent_category_capacities: dict, target_category: str = 'c1'):
#     #print(agent_category_capacities)#TODO remove
#     if agent_order is None: agent_order = list(alloc.remaining_agents())
#     remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
#                                            agent_category_capacities if agent_category_capacities[agent][
#                                                target_category] != 0}  # should look {'Agenti':[k:int]}
#     remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[target_category]]
#     # print(f'remaining agent capacities for {target_category}: {remaining_category_agent_capacities}')
#     # print(f'remaining {target_category} items: {remaining_category_items}')
#     #logger.info("\nPicking-sequence with items %s , agents %s, and agent-order %s", alloc.remaining_item_capacities,
#     #           alloc.remaining_agent_capacities, agent_order)
#     for agent in cycle(agent_order):
#         if len(remaining_category_agent_capacities) == 0 or len(
#                 remaining_category_items) == 0:  #replaces-> alloc.isdone():  isdone impl : return len(self.remaining_item_capacities) == 0 or len(self.remaining_agent_capacities) == 0
#             break
#         if not agent in remaining_category_agent_capacities:
#             continue  # skip to next agent
#         potential_items_for_agent = set(remaining_category_items).difference(
#             alloc.bundles[agent])  # we only deal with relevant items which are in target category
#         if len(potential_items_for_agent) == 0:  # means you already have 1 of the same item and there is conflict
#             #logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", agent,
#              #           alloc.remaining_item_capacities, alloc.bundles[agent])
#             continue
#         best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.effective_value(agent, item))
#         alloc.give(agent, best_item_for_agent,
#                    )
#         remaining_category_agent_capacities[agent] -= 1
#         if remaining_category_agent_capacities[agent] <= 0:
#             del remaining_category_agent_capacities[
#                 agent]
#
#         if best_item_for_agent not in alloc.remaining_item_capacities:  # since alloc deals with this we synchronize with it
#             remaining_category_items.remove(best_item_for_agent)  # equivelant for removing the item in allocationbuiler
#

def categorization_friendly_picking_sequence(alloc, agent_order, item_categories, agent_category_capacities,
                                             target_category='c1'):
    if agent_order is None:
        agent_order = [agent for agent in alloc.remaining_agents() if agent_category_capacities[agent][target_category] > 0]

    remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
                                           agent_category_capacities.keys() if
                                           agent_category_capacities[agent][target_category] != 0} # all the agents with non zero capacities in our category
    logger.info(f"agent_category_capacities-> {agent_category_capacities}")
    remaining_category_items = update_category_items(alloc, item_categories, target_category)
    active_agents=agent_order.copy()
    for agent in cycle(agent_order):
        logger.info(f"agent order is -> {agent_order}")
        logger.info('looping agent {}'.format(agent))
        logger.info(f"remaining_category_agent_capacities -> {remaining_category_agent_capacities}")
        if len(remaining_category_agent_capacities) == 0 or len(remaining_category_items) == 0:
            logger.info("1) IF")
            break
        if agent not in remaining_category_agent_capacities:
            logger.info("2) IF")
            continue# pass to the other agent
        potential_items_for_agent = set(remaining_category_items).difference(alloc.bundles[agent]) # in case difference is empty means already has item / there is no items left
        if len(potential_items_for_agent) == 0:
            logger.info("3) IF")
            logger.info(f'no potential items for agent {agent}')
            # either no items left / or agent already has items (conflicted)
            if remaining_category_agent_capacities[agent]>0:# need to remove agent from our loop
                del remaining_category_agent_capacities[agent] # TODO REMOVE IF NOT USEFUL
                logger.info(f"current agent order is -> {agent_order}")
                continue
        # safe to assume agent has capacity & has the best item to pick
        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.instance.agent_item_value(agent, item))
        logger.info(f'picked best item for {agent} -> item -> {best_item_for_agent}')
        alloc.give(agent, best_item_for_agent)# this handles capacity of item and capacity of agent !
        remaining_category_agent_capacities[agent] -= 1
        if remaining_category_agent_capacities[agent] <= 0:
            logger.info("4) IF")
            del remaining_category_agent_capacities[agent]
        remaining_category_items = update_category_items(alloc, item_categories, target_category) # in case an item is out of capacity the list is updated !
        logger.info(f'remaining_category_items -> {remaining_category_items} & remaining agents {remaining_category_agent_capacities}')


def update_category_items(alloc, item_categories, target_category):
    remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[target_category]]
    return remaining_category_items


def update_envy_graph(curr_bundles: dict, valuation_func: callable, envy_graph: DiGraph, item_categories: dict,
                      agent_category_capacities: dict):
    """
    simply a helper function to update the envy-graph based on given params
    :param curr_bundles: the current allocation
    :param valuation_func: simply a callable(agent,item)->value
    :param envy_graph: the envy-graph: of course the graph we're going to modify
    :param item_categories: a dictionary mapping items to categorize for the ease of checking:
    :param agent_category_capacities: a dictionary where keys are agents and values are dictionaries mapping categories to capacities.

    Example:
    >>> graph=DiGraph()
    >>> graph.add_edge('Agent1','Agent2')
    >>> valuation_func= lambda  agent,item: {'Agent1':{'m1':100,'m2':0},'Agent2':{'m1':100,'m2':0}}[agent][item]
    >>> bundle={'Agent1':['m1'],'Agent2':['m2']}
    >>> item_categories={'c1':['m1','m2']}
    >>> agent_category_capacities= {'Agent1':{'c1':1},'Agent2':{'c1':1}}
    >>> update_envy_graph(envy_graph=graph, valuation_func=valuation_func, item_categories=item_categories, agent_category_capacities=agent_category_capacities, curr_bundles=bundle )
    """
    envy_graph.clear_edges()
    for agent1, bundle1 in curr_bundles.items():
        for agent2, bundle_agent2 in curr_bundles.items():
            if agent1 is not agent2:  # make sure w're not comparing same agent to himself
                # make sure to value with respect to the constraints of feasibility
                # since in algo 1 its always feasible because everyone has equal capacity we dont pay much attention to it
                if envy(source=agent1, target=agent2, bundles=curr_bundles, val_func=valuation_func,
                        item_categories=item_categories, agent_category_capacities=agent_category_capacities):
                    #print(f"{agent1} envies {agent2}")  # works great .
                    # we need to add edge from the envier to the envyee
                    envy_graph.add_edge(agent1, agent2)


def visualize_graph(envy_graph):
    plt.figure(figsize=(8, 6))
    nx.draw(envy_graph, with_labels=True)
    plt.title('Basic Envy Graph')
    plt.show()


def remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities):
    attempt = 0
    max_attempts = 10000  # Set a maximum number of attempts to prevent infinite loops
    while not nx.is_directed_acyclic_graph(envy_graph) and attempt < max_attempts:
        attempt += 1
        logger.info(f"Cycle removal attempt {attempt}")
        try:
            cycle = nx.find_cycle(envy_graph, orientation='original')
            agents_in_cycle = [edge[0] for edge in cycle]
            logger.info(f"Detected cycle: {agents_in_cycle}")

            # Copy the bundle of the first agent in the cycle
            temp_val = alloc.bundles[agents_in_cycle[0]].copy()
            logger.info(f"Initial temp_val (copy of first agent's bundle): {temp_val}")

            # Perform the swapping
            for i in range(len(agents_in_cycle)):
                current_agent = agents_in_cycle[i]
                next_agent = agents_in_cycle[(i + 1) % len(agents_in_cycle)]
                logger.info(
                    f"Before swapping: {current_agent} -> {alloc.bundles[current_agent]}, {next_agent} -> {alloc.bundles[next_agent]}")

                # Swap the bundles
                alloc.bundles[next_agent], temp_val = temp_val, alloc.bundles[next_agent].copy()

                logger.info(
                    f"After swapping: {current_agent} -> {alloc.bundles[current_agent]}, {next_agent} -> {alloc.bundles[next_agent]}")
                logger.info(f"Updated temp_val: {temp_val}")

            # Update the envy graph after swapping
            old_envy_graph = envy_graph.copy()
            update_envy_graph(alloc.bundles, valuation_func, envy_graph, item_categories, agent_category_capacities)
            logger.info(f"Updated envy graph. Graph is acyclic: {nx.is_directed_acyclic_graph(envy_graph)}")

            # Check if the envy graph changed
            if nx.is_isomorphic(old_envy_graph, envy_graph):
                logger.error("Envy graph did not change after cycle removal attempt. Potential logical error.")
                break

        except nx.NetworkXNoCycle:
            logger.info("No more cycles detected")
            break
        except Exception as e:
            logger.error(f"Error during cycle removal: {e}")
            break

    if not nx.is_directed_acyclic_graph(envy_graph):
        logger.warning("Cycle removal failed to achieve an acyclic graph within the maximum attempts.")
        raise RuntimeError("Cycle removal failed to achieve an acyclic graph.")

    logger.info("Cycle removal process ended successfully")


# Assuming update_envy_graph and other required functions are defined elsewhere

# Assuming update_envy_graph and other required functions are defined elsewhere

# def remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities):
#     """
#     Removes cycles from the envy graph by updating the bundles.
#
#     :param envy_graph: The envy graph (a directed graph).
#     :param alloc: An AllocationBuilder instance for managing allocations.
#     :param valuation_func: Function to determine the value of an item for an agent.
#     :param item_categories: A dictionary of categories, with each category paired with a list of items.
#     :param agent_category_capacities: A dictionary of dictionaries mapping agents to their capacities for each category.
#
#     >>> import networkx as nx
#     >>> from fairpyx import Instance, AllocationBuilder
#     >>> valuations = {'Agent1': {'Item1': 3, 'Item2': 5,'Item3': 1}, 'Agent2': {'Item1': 2, 'Item2': 6 ,'Item3':10}, 'Agent3': {'Item1': 4, 'Item2': 1,'Item3':2.8}}
#     >>> items = ['Item1', 'Item2','Item3']
#     >>> instance = Instance(valuations=valuations, items=items)
#     >>> alloc = AllocationBuilder(instance)
#     >>> alloc.give('Agent1', 'Item1')
#     >>> alloc.give('Agent2', 'Item2')
#     >>> alloc.give('Agent3', 'Item3')
#     >>> item_categories = {'c1': ['Item1', 'Item2','Item3']}
#     >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}, 'Agent3': {'c1': 1}}
#     >>> envy_graph = nx.DiGraph()
#     >>> valuation_callable=alloc.instance.agent_item_value
#     >>> def valuation_func(agent, item): return valuation_callable(agent,item)
#     >>> update_envy_graph(alloc.bundles, valuation_func, envy_graph, item_categories, agent_category_capacities)
#     >>> list(envy_graph.edges)
#     [('Agent1', 'Agent2'), ('Agent2', 'Agent3'), ('Agent3', 'Agent1')]
#     >>> not nx.is_directed_acyclic_graph(envy_graph)
#     True
#     >>> remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
#     >>> list(nx.simple_cycles(envy_graph))
#     []
#     >>> list(envy_graph.edges)
#     []
#     >>> alloc.bundles
#     {'Agent1': {'Item2'}, 'Agent2': {'Item3'}, 'Agent3': {'Item1'}}
#     """
#     while not nx.is_directed_acyclic_graph(envy_graph):
#         try:
#             cycle = nx.find_cycle(envy_graph, orientation='original')
#             #print("Detected cycle:", cycle)  # Debugging: Print detected cycle
#
#             # Extract the agents involved in the cycle
#             agents_in_cycle = [edge[0] for edge in cycle]
#             #print(f'agents in cycle{agents_in_cycle}')
#             # Perform bundle switching along the cycle
#             temp_val = alloc.bundles[agents_in_cycle[0]]  # we begin with temp as the bundle of the first agent
#             for i in range(len(agents_in_cycle)):
#                 #print(f"bundle in iteration {i} is {alloc.bundles}")
#                 current_agent = agents_in_cycle[i]
#                 next_agent = agents_in_cycle[(i + 1) % len(agents_in_cycle)]
#                 alloc.bundles[next_agent], temp_val = temp_val, alloc.bundles[
#                     next_agent]  # instead of the boring swap which requires extra temp val
#                 #print(f"bundle in iteration {i} after modification {alloc.bundles}")
#             # Update the envy graph
#             update_envy_graph(alloc.bundles, valuation_func, envy_graph, item_categories, agent_category_capacities)
#
#         except nx.NetworkXNoCycle:
#             break

def per_category_round_robin(alloc, item_categories, agent_category_capacities, initial_agent_order):

    logger.info(f"allocationbuilder : {alloc} \n item_categories -> {item_categories} \n agent_category_capacities -> {agent_category_capacities} \n -> initial_agent_order are -> {initial_agent_order}\n ")
    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value

    for category in item_categories.keys():
        logger.info(f"item_categories.keys() -> {item_categories.keys()}")
        categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities, category)
        logger.info(f"done with RR in {category}")
        update_envy_graph(alloc.bundles, valuation_func, envy_graph, item_categories, agent_category_capacities)
        logger.info(f"done with envy_graph in {category}")
        if not nx.is_directed_acyclic_graph(envy_graph):
            logger.info(
                f"CYCLE REMOVAL STARTED ")

            remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
            logger.info(
                f"CYCLE REMOVAL ENDED ")
        current_order = list(nx.topological_sort(envy_graph))
        logger.info("TOPOLOGICAL SORT DONE ")


# def per_category_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
#                              initial_agent_order: list):
#     """
#     this is the Algorithm 1 from the paper
#     per category round-robin is an allocation algorithm which guarantees EF1 (envy-freeness up to 1 good) allocation
#     under settings in which agent-capacities are equal across all agents,
#     no capacity-inequalities are allowed since this algorithm doesnt provie a cycle-prevention mechanism
#     TLDR: same partition constriants , same capacities , may have different valuations across agents  -> EF1 allocation
#
#     :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
#     :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
#     :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
#     paired with a dictionary of category-capacity.
#     :param initial_agent_order: a list representing the order we start with in the algorithm
#
#     >>> # Example 1
#     >>> from fairpyx import  divide
#     >>> order=['Agent1','Agent2']
#     >>> items=['m1','m2','m3']
#     >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
#     >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
#     >>> valuations = {'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
#     >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
#     >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
#     {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}
#
#     >>> # Example 2
#     >>> from fairpyx import  divide
#     >>> order=['Agent1','Agent3','Agent2']
#     >>> items=['m1','m2','m3']
#     >>> item_categories = {'c1': ['m1','m3'], 'c2': ['m2']}
#     >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':3}, 'Agent2': {'c1':3,'c2':3},'Agent3': {'c1':3,'c2':3}}
#     >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':4},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':4,'m2':6,'m3':5}}
#     >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
#     >>> result=divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
#     >>> assert result in [{'Agent1': ['m2'], 'Agent2': ['m1'], 'Agent3': ['m3']},{'Agent1': ['m1'], 'Agent2': ['m3'], 'Agent3': ['m2']}]
#
#     >>> # example 3 but trying to get the expected output exactly (modified valuations different than on papers)  (4 agents ,4 items)
#     >>> from fairpyx import  divide
#     >>> order=['Agent1','Agent2','Agent3','Agent4']
#     >>> items=['m1','m2','m3','m4']
#     >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
#     >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2},'Agent4': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
#     >>> valuations = {'Agent1':{'m1':2,'m2':1,'m3':1,'m4':10},'Agent2':{'m1':1,'m2':2,'m3':1,'m4':10},'Agent3':{'m1':1,'m2':1,'m3':2,'m4':10},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':10}}
#     >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
#     >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
#     {'Agent1': ['m1'], 'Agent2': ['m2'], 'Agent3': ['m3'], 'Agent4': ['m4']}
#     """
#     logger.info(f"allocationbuilder : {alloc} \n item_categories -> {item_categories} \n agent_category_capacities -> {agent_category_capacities} \n -> initial_agent_order are -> {initial_agent_order}\n ")
#     envy_graph = nx.DiGraph()
#     current_order = initial_agent_order
#     valuation_func = alloc.instance.agent_item_value
#
#     for category in item_categories.keys():
#         categorization_friendly_picking_sequence(
#             alloc, current_order, item_categories, agent_category_capacities, category
#         )
#         update_envy_graph(
#             curr_bundles=alloc.bundles,
#             valuation_func=valuation_func,
#             envy_graph=envy_graph,
#             item_categories=item_categories,
#             agent_category_capacities=agent_category_capacities
#         )
#         if not nx.is_directed_acyclic_graph(envy_graph):
#             remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
#         current_order = list(nx.topological_sort(envy_graph))


def capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                       initial_agent_order: list, target_category: str = 'c1'):
    """
    this is Algorithm 2 CRR (capped round-robin) algorithm TLDR: single category , may have differnt capacities
    capped in CRR stands for capped capacity for each agent unlke RR , maye have different valuations -> F-EF1 (
    feasible envy-freeness up to 1 good) allocation

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param initial_agent_order: a list representing the order we start with in the algorithm
        :param target_category: a string representing the category we are going to be processing

    >>> # Example 1 (2 agents 1 of them with capacity of 0)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1']
    >>> item_categories = {'c1': ['m1']}
    >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1}}
    >>> valuations = {'Agent1':{'m1':0},'Agent2':{'m1':420}}
    >>> target_category='c1'
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order,target_category=target_category)
    {'Agent1': [], 'Agent2': ['m1']}

    >>> # Example 2 (3 agents , 4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3','m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':2}, 'Agent2': {'c1':2},'Agent3': {'c1':2}}
    >>> valuations = {'Agent1':{'m1':10,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':10,'m3':1},'Agent3':{'m1':1,'m2':1,'m3':10}}
    >>> target_category='c1'
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order,target_category=target_category)
    {'Agent1': ['m1', 'm4'], 'Agent2': ['m2'], 'Agent3': ['m3']}


    >>> # Example 3  (to show that F-EF (feasible envy-free) is sometimes achievable in good scenarios)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1','m2']
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':1},'Agent3': {'c1':1}}
    >>> valuations = {'Agent1':{'m1':10,'m2':5},'Agent2':{'m1':5,'m2':10}}
    >>> target_category='c1'
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order,target_category=target_category)
    {'Agent1': ['m1'], 'Agent2': ['m2']}

    >>> # Example 4  (4 Agents , different capacities ,extra unallocated item at the termination of the algorithm)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4','m5','m6','m7']
    >>> item_categories = {'c1': ['m1', 'm2','m3','m4','m5','m6','m7']}
    >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1},'Agent3': {'c1':2},'Agent4': {'c1':3}}
    >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6,'m7':0},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1,'m7':0},'Agent3':{'m1':1,'m2':2,'m3':5,'m4':6,'m5':3,'m6':4,'m7':0},'Agent4':{'m1':5,'m2':4,'m3':1,'m4':2,'m5':3,'m6':6,'m7':0}}
    >>> target_category='c1'
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order,target_category=target_category)
    {'Agent1': [], 'Agent2': ['m1'], 'Agent3': ['m3', 'm4'], 'Agent4': ['m2', 'm5', 'm6']}
        """

    # no need for envy graphs whatsoever
    current_order = initial_agent_order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             target_category=target_category)  # this is RR without wrapper


def two_categories_capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      initial_agent_order: list, target_category_pair: tuple[str] = ('c1', 'c2')):
    """
        this is Algorithm 3 back and forth capped round-robin algorithm (2 categories,may have different capacities,may have different valuations)
        in which we simply
        1)call capped_round_robin(arg1 ,.... argk,item_categories=<first category>)
        2) reverse(order)
        3)call capped_round_robin(arg1 ,.... argk,item_categories=<second category>)
        -> F-EF1 (feasible envy-freeness up to 1 good) allocation

            :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
            :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
            :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
            paired with a dictionary of category-capacity.
            :param initial_agent_order: a list representing the order we start with in the algorithm
            :param target_category_pair: a pair of 2 categories since our algorithm deals with 2

            >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2']
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1','m2'],'c2':['m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':10,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1}}
            >>> target_category_pair=('c1','c2')
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order,target_category_pair=target_category_pair)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3']}

            >>> # Example 2 (case of single category so we deal with it as the normal CRR)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2','Agent3']
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':[]}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':0}, 'Agent2': {'c1':2,'c2':0},'Agent3': {'c1':0,'c2':0}}
            >>> valuations = {'Agent1':{'m1':10,'m2':3,'m3':3},'Agent2':{'m1':3,'m2':1,'m3':1},'Agent3':{'m1':10,'m2':10,'m3':10}}
            >>> target_category_pair=('c1','c2')
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order,target_category_pair=target_category_pair)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3'], 'Agent3': []}


             >>> # Example 3  (4 agents 6 items same valuations same capacities)-> EF in best case scenario
            >>> from fairpyx import  divide
            >>> order=['Agent2','Agent4','Agent1','Agent3']
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2'],'c2': ['m3', 'm4','m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1}, 'Agent2': {'c1':1,'c2':1},'Agent3': {'c1':1,'c2':1},'Agent4': {'c1':1,'c2':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':10,'m5':1,'m6':1},'Agent2':{'m1':10,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':1,'m2':1,'m3':10,'m4':1,'m5':1,'m6':1},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':10,'m6':1}}
            >>> target_category_pair=('c1','c2')
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order,target_category_pair=target_category_pair)
            {'Agent1': ['m4'], 'Agent2': ['m1', 'm6'], 'Agent3': ['m3'], 'Agent4': ['m2', 'm5']}


            >>> # Example 4  (3 agents 6 items different valuations different capacities, remainder item at the end)-> F-EF1
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2','Agent3']
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2','m3', 'm4'],'c2': ['m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':1}, 'Agent2': {'c1':0,'c2':2},'Agent3': {'c1':0,'c2':5}}
            >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1},'Agent3':{'m1':5,'m2':3,'m3':1,'m4':2,'m5':4,'m6':6}}
            >>> target_category_pair=('c1','c2')
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order,target_category_pair=target_category_pair)
            {'Agent1': ['m2', 'm3', 'm4'], 'Agent2': ['m5'], 'Agent3': ['m6']}
            >>> # m1 remains unallocated unfortunately :-(
            """
    current_order = initial_agent_order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             target_category=target_category_pair[0])  #calling CRR on first category
    current_order.reverse()  #reversing order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             target_category=target_category_pair[1])  # calling CRR on first category


def per_category_capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                    initial_agent_order: list):
    """
    this is Algorithm 4 deals with (Different Capacities, Identical Valuations), suitable for any number of categories
    CRR (per-category capped round-robin) algorithm
    TLDR: multiple categories , may have different capacities , but have identical valuations -> F-EF1 (feasible envy-freeness up to 1 good) allocation

    Lemma 1. For any setting with identical valuations (possibly with different capacities), the
    feasible envy graph of any feasible allocation is acyclic.-> no need to check for cycles and eleminate them

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionary in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param order: a list representing the order we start with in the algorithm

         >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2']
            >>> items=['m1','m2','m3','m4']
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':10,'m2':5,'m3':1,'m4':4},'Agent2':{'m1':10,'m2':5,'m3':1,'m4':4}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order)
            {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

            >>> # Example 2 (3 agents 3 categories , different capacities )
            >>> from fairpyx import  divide
            >>> order=['Agent2','Agent3','Agent1']#TODO change in papers
            >>> items=['m1','m2','m3','m4','m5','m6','m7','m8','m9']
            >>> item_categories = {'c1': ['m1','m2','m3','m4'],'c2':['m5','m6','m7'],'c3':['m8','m9']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':4,'c3':4}, 'Agent2': {'c1':4,'c2':0,'c3':4},'Agent3': {'c1':4,'c2':4,'c3':0}}
            >>> valuations = {'Agent1':{'m1':11,'m2':10,'m3':9,'m4':1,'m5':10,'m6':9.5,'m7':9,'m8':2,'m9':3},'Agent2':{'m1':11,'m2':10,'m3':9,'m4':1,'m5':10,'m6':9.5,'m7':9,'m8':2,'m9':3},'Agent3':{'m1':11,'m2':10,'m3':9,'m4':1,'m5':10,'m6':9.5,'m7':9,'m8':2,'m9':3}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order)
            {'Agent1': ['m3', 'm5', 'm7', 'm8'], 'Agent2': ['m1', 'm4', 'm9'], 'Agent3': ['m2', 'm6']}

             >>> # Example 3 (3 agents 3 categories , 1 item per category)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2','Agent3']
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2'],'c3':['m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':1,'c2':1,'c3':1}}
            >>> valuations = {'Agent1':{'m1':7,'m2':8,'m3':9},'Agent2':{'m1':7,'m2':8,'m3':9},'Agent3':{'m1':7,'m2':8,'m3':9}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order = order)
            {'Agent1': ['m1'], 'Agent2': ['m2'], 'Agent3': ['m3']}
    """
    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    for category in item_categories.keys():
        categorization_friendly_picking_sequence(alloc=alloc, agent_order=current_order,
                                                 item_categories=item_categories,
                                                 agent_category_capacities=agent_category_capacities,
                                                 target_category=category)
        update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                          item_categories=item_categories, agent_category_capacities=agent_category_capacities)
        current_order = list(nx.topological_sort(envy_graph))


def iterated_priority_matching(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict):# TODO recheck algorithm around 10% of tests fall due to not satisfying f-ef1
    """
    this is Algorithm 5  deals with (partition Matroids with Binary Valuations, may have different capacities)
    loops as much as maximum capacity in per each category , each iteration we build :
    1) agent-item graph (bidirectional graph)
    2) envy graph
    3) topological sort the order based on the envy graph (always a-cyclic under such settings,proven in papers)
    4) compute priority matching based on it we allocate the items among the agents
    we do this each loop , and in case there remains item in that category we arbitrarily give it to random agent

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionary in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param order: a list representing the order we start with in the algorithm

            >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1','m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':0,'m3':0},'Agent2':{'m1':0,'m2':1,'m3':0}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3']}


            >>> # Example 2 ( 3 agents  with common interests in certain items)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2},'Agent3': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':0},'Agent3':{'m1':0,'m2':0,'m3':0}} # TODO change valuation in paper
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1': ['m1', 'm3'], 'Agent2': ['m2'], 'Agent3': []}

             >>> # Example 3 ( 3 agents , 3 categories , with common interests, and remainder unallocated items at the end )
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3','m4','m5','m6']#TODO change in papers since in case there is no envy we cant choose whatever order we want. maybe on papers yes but in here no
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':0,'c2':0,'c3':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':0,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':0,'m2':0,'m3':0,'m4':0,'m5':0,'m6':1}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)# m3 remains unallocated ....
            {'Agent1': ['m1', 'm5', 'm6'], 'Agent2': ['m2', 'm4'], 'Agent3': []}
   """
    envy_graph = nx.DiGraph()
    envy_graph.add_nodes_from(alloc.remaining_agents())  # adding agent nodes (no edges involved yet)
    current_order = [agent for agent in alloc.remaining_agents()]  # in this algorithm no need for initial_agent_order
    valuation_func = alloc.instance.agent_item_value

    for category in item_categories.keys():
        maximum_capacity = max(
            [agent_category_capacities[agent][category] for agent in
             agent_category_capacities.keys()])  # for the sake of inner iteration
        remaining_category_agent_capacities = {
            agent: agent_category_capacities[agent][category] for agent in agent_category_capacities if
            agent_category_capacities[agent][category] != 0
        }  # dictionary of the agents paired with capacities with respect to the current category we're dealing with

        current_item_list = update_item_list(alloc, category,
                                             item_categories)  # items we're dealing with with respect to the category
        current_agent_list = update_ordered_agent_list(current_order,
                                                       remaining_category_agent_capacities)  #  items we're dealing with with respect to the constraints
        print(f'current item list: {current_item_list}\n current agent list: {current_agent_list} \n and allocationbuilder bundles{alloc.bundles}\n and allocationbuilder remaining capcities {alloc.remaining_agent_capacities} \nand remaining agent-capacity list: {remaining_category_agent_capacities} \n agent_category_capacities {agent_category_capacities} ')
        print('*******************************************************')
        for i in range(
                maximum_capacity):  # as in papers we run for the length of the maximum capacity out of all agents for the current category
            # Creation of agent-item graph
            agent_item_bipartite_graph = create_agent_item_bipartite_graph(
                agents=remaining_category_agent_capacities.keys(),  # remaining agents
                items=[item for item in alloc.remaining_items() if item in item_categories[category]],
                # remaining items
                valuation_func=valuation_func,
                current_agent_list=current_agent_list  # remaining agents with respect to the order
            )  # building the Bi-Partite graph

            # Creation of envy graph
            update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                              item_categories=item_categories,
                              agent_category_capacities=agent_category_capacities)  # updating envy graph with respect to matchings (first iteration we get no envy, cause there is no matching)
            #topological sort (papers prove graph is always a-cyclic)
            try:
                sort = list(nx.topological_sort(envy_graph))
            except nx.NetworkXUnfeasible:
                pass
                #print("graph has cycle !")
                #print(alloc.bundles)
                #print({agent:{item:alloc.instance.agent_item_value(agent,item)} for agent in alloc.remaining_agents() for item in alloc.remaining_items()}) # TODO there is mismatch between this and the VALUATIONS (valuations binary this is 2112313)
            current_order = current_order if not sort else sort
            # Perform priority matching
            priority_matching(agent_item_bipartite_graph, current_order, alloc,
                              remaining_category_agent_capacities)  # deals with eliminating finished agents from agent_category_capacities

            current_item_list = update_item_list(alloc, category,
                                                 item_categories)  # important to update the item list after priority matching.
            current_agent_list = update_ordered_agent_list(current_order,
                                                           remaining_category_agent_capacities)  # important to update the item list after priority matching.


        while len(remaining_category_agent_capacities) > 0 and len(current_item_list) > 0:# Note current_agent_list = f(remaining_category_agent_capacities,order)
            arbitrary_agent = random.choice(list(remaining_category_agent_capacities.keys()))
            arbitrary_item = random.choice(current_item_list)
            if arbitrary_item not in alloc.instance.agent_conflicts(arbitrary_agent):
                alloc.give(arbitrary_agent, arbitrary_item)  # give item
                if arbitrary_item not in alloc.remaining_item_capacities.keys():
                    current_item_list.remove(
                        arbitrary_item)  #remove item from list
                remaining_category_agent_capacities[
                    arbitrary_agent] -= 1  # decrease capacity in out variable (the alloc capacity is handled in alloc.give(...))
                if remaining_category_agent_capacities[arbitrary_agent] <= 0:  # eliminate if capacity reached 0
                    del remaining_category_agent_capacities[arbitrary_agent]
                    current_agent_list.remove(arbitrary_agent)
            else: continue # agent has conflict so we go on to the next randomization option


def update_ordered_agent_list(current_order: list, remaining_category_agent_capacities: dict) -> list:
    """
       Update the ordered list of agents based on remaining category agent capacities.

       Args:
           current_order (list): Current order of agents.
           remaining_category_agent_capacities (dict): A dictionary with category capacities for each agent.

       Returns:
           list: The updated list of agents ordered by remaining capacities.

       Example:
           >>> current_order = ['Alice', 'Bob']
           >>> remaining_category_agent_capacities = {'Alice': 1, 'Bob': 2}
           >>> update_ordered_agent_list(current_order, remaining_category_agent_capacities)
           ['Alice', 'Bob']
    """
    current_agent_list = [agent for agent in current_order if agent in remaining_category_agent_capacities.keys()]
    return current_agent_list


def update_item_list(alloc: AllocationBuilder, category: str, item_categories: dict) -> list:
    """
        Update the item list for a given category and allocation.

        Args:
            alloc: Current allocationBuilder.
            category: The category to update.
            item_categories : A dictionary mapping categories to items .

        Returns:
            list: The updated allocation list with items for the specified category.

        Example:
            >>> instance =Instance = Instance(valuations={'Agent1': {'m1': 1, 'm2': 0}, 'Agent2': {'m1': 0, 'm2': 1}}, items=['m1', 'm2'])
            >>> alloc= AllocationBuilder(instance=instance)
            >>> alloc.give('Agent1','m1')
            >>> category = 'c1'
            >>> item_categories = {'c1':'m1', 'c2':'m2'}
            >>> update_item_list(alloc, category, item_categories)
            []
        """
    current_item_list = [item for item in alloc.remaining_items() if item in item_categories[category]]
    return current_item_list


def priority_matching(agent_item_bipartite_graph, current_order, alloc, remaining_category_agent_capacities):
    """
    Performs priority matching based on the agent-item bipartite graph and the current order of agents.

    :param agent_item_bipartite_graph: A bipartite graph with agents and items as nodes, and edges with weights representing preferences.
    :param current_order: The current order of agents for matching.
    :param alloc: An AllocationBuilder instance to manage the allocation process.
    :param remaining_category_agent_capacities: A dictionary mapping agents to their remaining capacities for the category.

    >>> import networkx as nx
    >>> from fairpyx import Instance, AllocationBuilder
    >>> agent_item_bipartite_graph = nx.Graph()
    >>> agent_item_bipartite_graph.add_nodes_from(['Agent1', 'Agent2'], bipartite=0)
    >>> agent_item_bipartite_graph.add_nodes_from(['Item1', 'Item2'], bipartite=1)
    >>> agent_item_bipartite_graph.add_edge('Agent1', 'Item1', weight=1)
    >>> agent_item_bipartite_graph.add_edge('Agent2', 'Item2', weight=1)
    >>> current_order = ['Agent1', 'Agent2']
    >>> instance = Instance(valuations={'Agent1': {'Item1': 1, 'Item2': 0}, 'Agent2': {'Item1': 0, 'Item2': 1}}, items=['Item1', 'Item2'])
    >>> alloc = AllocationBuilder(instance)
    >>> remaining_category_agent_capacities = {'Agent1': 1, 'Agent2': 1}
    >>> priority_matching(agent_item_bipartite_graph, current_order, alloc, remaining_category_agent_capacities)
    >>> alloc.bundles
    {'Agent1': {'Item1'}, 'Agent2': {'Item2'}}
    """
    matching=nx.max_weight_matching(agent_item_bipartite_graph)
    # print(f'matching is {matching}')
    # print(f'capacity is {alloc.remaining_item_capacities}')
    # print(f'alloc bundle is {alloc.bundles}')
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    for match in matching:
        # TODO check if agent doesnt heve conflict with the item (already has 1 )
        if match[0].startswith('A'):
            if ((match[0], match[1]) not in alloc.remaining_conflicts) and match[0] in remaining_category_agent_capacities.keys():
                alloc.give(match[0], match[1])
                remaining_category_agent_capacities[match[0]] -= 1
                if remaining_category_agent_capacities[match[0]] <= 0:
                    del remaining_category_agent_capacities[match[0]]
            #else do nothing ....
        else:
            # TODO oppa ! apparently problem is with the instance generator in pytest valuations are not in  (0,1)
            if  ((match[1], match[0]) not in alloc.remaining_conflicts) and match[1] in remaining_category_agent_capacities.keys():
                alloc.give(match[1], match[0])
                remaining_category_agent_capacities[match[1]] -= 1
                if remaining_category_agent_capacities[match[1]] <= 0:
                    del remaining_category_agent_capacities[match[1]]


def create_agent_item_bipartite_graph(agents, items, valuation_func,
                                      current_agent_list):  # TODO remove unnecessary argument (agents or current_agent_list)
    """
    Creates an agent-item bipartite graph with edges weighted based on agent preferences.

    :param agents: List of agents.
    :param items: List of items.
    :param valuation_func: Function to determine the value of an item for an agent.
    :param current_agent_list: List of agents currently being considered for matching.(ordered)
    :return: A bipartite graph with agents and items as nodes, and edges with weights representing preferences.

    >>> import networkx as nx
    >>> agents = ['Agent1', 'Agent2']
    >>> items = ['Item1', 'Item2']
    >>> def valuation_func(agent, item):
    ...     values = {('Agent1', 'Item1'): 3, ('Agent1', 'Item2'): 1, ('Agent2', 'Item1'): 2, ('Agent2', 'Item2'): 4}
    ...     return values.get((agent, item), 0)
    >>> bipartite_graph = create_agent_item_bipartite_graph(agents, items, valuation_func, agents)
    >>> sorted(bipartite_graph.edges(data=True))
    [('Agent1', 'Item1', {'weight': 2}), ('Agent1', 'Item2', {'weight': 2}), ('Agent2', 'Item1', {'weight': 1}), ('Agent2', 'Item2', {'weight': 1})]
    """
    agent_item_bipartite_graph = nx.Graph()
    agent_item_bipartite_graph.add_nodes_from(agents, bipartite=0)
    agent_item_bipartite_graph.add_nodes_from(items, bipartite=1)

    weight = len(current_agent_list)
    for agent in current_agent_list:
        for item in items:
            if valuation_func(agent, item) != 0:
                agent_item_bipartite_graph.add_edge(agent, item, weight=weight)
        weight -= 1

    return agent_item_bipartite_graph


if __name__ == "__main__":
    import doctest

    doctest.testmod()
