"""
Implement an "almost-egalitarian" course allocation,
by rounding a linear program.

Programmer: Erel Segal-Halevi.
Since: 2023-07
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
from fairpyx.algorithms.iterated_maximum_matching import iterated_maximum_matching
from fairpyx.algorithms.fractional_egalitarian import fractional_egalitarian_utilitarian_allocation

import cvxpy, numpy as np, networkz as nx
from fairpyx.utils.solve import solve
from collections import defaultdict
# import matplotlib.pyplot as plt # for plotting the consumption graph (for debugging)


import logging
logger = logging.getLogger(__name__)

class ConsumptionGraph:
    """
    A graph that represents a fractional allocation.
    It is a bipartite graph with agents in one side, items on the other side,
    and an edge between each agent and the items of which he has positive quantity.

    Overrides networkx graph to fix a bug when agent and item have the same name.
    """

    def __init__(self, allocation:dict, min_fraction=0.01, agent_item_value=None):
        """
        :param allocation - the fractional allocation (maps each agent to a map from item to fraction).
        :param min_fraction - smallest fraction for which an edge will be created.
        :param agent_item_value - a function that maps agent,item to the agent's value for the item.
        """
        self.graph = nx.Graph()

        self.map_agent_to_items = defaultdict(dict)
        self.map_item_to_agents  = defaultdict(dict)
        self.num_of_edges = 0

        for agent,bundle in allocation.items():
            for item,fraction in bundle.items():
                if fraction>=min_fraction:
                    # value = None if agent_item_value is None else agent_item_value(agent,item)
                    self.add_edge(agent,item, weight=np.round(fraction,2))

    def has_edge(self, agent, item):
        return item in self.map_item_to_agents and agent in self.map_item_to_agents[item]

    def add_edge(self, agent, item, weight=0):
        if not self.has_edge(agent,item):
            self.map_item_to_agents[item][agent] = weight
            self.map_agent_to_items[agent][item] = weight
            self.num_of_edges += 1

    def remove_edge(self, agent, item):
        if self.has_edge(agent,item):
            del self.map_item_to_agents[item][agent]
            del self.map_agent_to_items[agent][item]
            self.num_of_edges -= 1

    def agent_degree(self, agent)->int:
        return len(self.map_agent_to_items[agent])

    def item_degree(self, item)->int:
        return len(self.map_item_to_agents[item])
    
    def number_of_edges(self, u=None, v=None) -> int:
        return self.num_of_edges
        
    def contains_item(self,item):
        return item in self.map_item_to_agents
        
    def contains_agent(self,agent):
        return agent in self.map_agent_to_items

    def agent_neighbors(self, agent):
        return self.map_agent_to_items[agent].keys()
    
    def agent_first_neighbor(self, agent):
        return next(iter(self.agent_neighbors(agent)))

    def item_neighbors(self, item):
        return self.map_item_to_agents[item].keys()
    
    def item_first_neighbor(self, item):
        return next(iter(self.item_neighbors(item)))

    def weight(self, agent, item):
        if self.has_edge(agent,item):
            return self.map_item_to_agents[item][agent]
            return self.map_agent_to_items[agent][item]  # Should be the same

    def set_weight(self, agent, item, weight):
        self.map_item_to_agents[item][agent] = weight
        self.map_agent_to_items[agent][item] = weight

    def edges(self): 
        """
        Returns a sequence of all edges in the graph. Each edge is a tuple (agent,item).
        """
        for agent,items in self.map_agent_to_items.items():
            for item in items.keys():
                yield (agent,item)

    def __repr__(self):
        return list(self.edges())
        # return self.map_agent_to_items
    
    def __str__(self):
        return str(self.__repr__())



MIN_EDGE_FRACTION=0.01
def almost_egalitarian_allocation(alloc: AllocationBuilder, surplus_donation:bool=False, explanation_logger:ExplanationLogger=ExplanationLogger(), **solver_options):
    """
    Finds an almost-egalitarian allocation.
    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents. of the fair course allocation problem. 
    :param surplus_donation: if True, agents who gain utility from previously-removed edges will donate some of their edges to others.

    >>> from fairpyx.adaptors import divide

    >>> from fairpyx.utils.test_utils import stringify

    >>> instance = Instance(valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":2, "y":3, "z":4, "w":5}}, agent_capacities=1, item_capacities=1)
    >>> stringify(divide(almost_egalitarian_allocation, instance=instance))
    "{avi:['x'], beni:['w']}"

    >>> instance = Instance(valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":2, "y":3, "z":4, "w":5}}, agent_capacities=2, item_capacities=1)
    >>> stringify(divide(almost_egalitarian_allocation, instance=instance))
    "{avi:['x', 'y'], beni:['w', 'z']}"

    >>> instance = Instance(valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":2, "y":3, "z":4, "w":5}}, agent_capacities=3, item_capacities=2)
    >>> stringify(divide(almost_egalitarian_allocation, instance=instance))
    "{avi:['x', 'y', 'z'], beni:['w', 'y', 'z']}"

    >>> instance = Instance(valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":2, "y":3, "z":4, "w":5}}, agent_capacities=4, item_capacities=2)
    >>> stringify(divide(almost_egalitarian_allocation, instance=instance))
    "{avi:['w', 'x', 'y', 'z'], beni:['w', 'x', 'y', 'z']}"

    ### Matrix of values:
    >>> instance = Instance(valuations=[[5,4,3,2],[2,3,4,5]], agent_capacities=2, item_capacities=1)
    >>> stringify(divide(almost_egalitarian_allocation, instance=instance))
    '{0:[0, 1], 1:[2, 3]}'
    """
    # fractional_allocation = fractional_leximin_optimal_allocation(alloc.remaining_instance(), **solver_options) # Too slow

    explanation_logger.info("\nAlgorithm Almost-Egalitarian starts.\n")

    fractional_allocation = fractional_egalitarian_utilitarian_allocation(alloc.remaining_instance(), **solver_options)
    explanation_logger.explain_fractional_allocation(fractional_allocation, alloc.instance)

    fractional_allocation_graph = ConsumptionGraph(
        fractional_allocation, 
        min_fraction=MIN_EDGE_FRACTION, 
        agent_item_value=alloc.effective_value
        )
    explanation_logger.debug("\nfractional_allocation_graph: %s", fractional_allocation_graph)

    explanation_logger.info("\nStarting to round the fractional allocation.\n")

    agent_surplus = {agent: 0 for agent in alloc.remaining_agents()}

    def add_surplus (agent, value_to_add):
        agent_surplus[agent] += value_to_add
        items_to_remove = []
        for neighbor_item in fractional_allocation_graph.agent_neighbors(agent):
            current_neighbor_weight = fractional_allocation_graph.weight(agent,neighbor_item)
            current_neighbor_value  = current_neighbor_weight * alloc.effective_value(agent,neighbor_item)
            if current_neighbor_value <= agent_surplus[agent]:
                explanation_logger.info("  You have a surplus of %g, so you donate your share of %g%% in course %s (value %g)", agent_surplus[agent], np.round(100*current_neighbor_weight), neighbor_item, current_neighbor_value, agents=agent)
                items_to_remove.append(neighbor_item)
                agent_surplus[agent] -= current_neighbor_value
        for neighbor_item in items_to_remove:
            remove_edge_from_graph(agent, neighbor_item)


    def remove_edge_from_graph(agent,item):
        """
        Remove the edge (agent,item) from the graph, and redistribute its weight among the neighboring agents of item.
        """
        weight_for_redistribution = fractional_allocation_graph.weight(agent,item) # this weight should be redistributed to other neighbors of the item
        # weight_for_redistribution = fractional_allocation[agent][item] # this weight should be redistributed to other neighbors of the item
        explanation_logger.debug(f"  Your fraction {weight_for_redistribution} of item {item} is given to other agents", agents=agent)
        fractional_allocation[agent][item] = 0
        fractional_allocation_graph.remove_edge(agent,item)
        surplus_to_add = {}
        for neighbor_agent in fractional_allocation_graph.item_neighbors(item):
            current_neighbor_weight = fractional_allocation_graph.weight(neighbor_agent,item)

            weight_to_add = min(weight_for_redistribution, 1-current_neighbor_weight)
            new_neighbor_weight = current_neighbor_weight + weight_to_add
            fractional_allocation[neighbor_agent][item] = new_neighbor_weight
            fractional_allocation_graph.set_weight(neighbor_agent, item, new_neighbor_weight)
            weight_for_redistribution -= weight_to_add

            value_to_add = weight_to_add*alloc.effective_value(agent,item)
            explanation_logger.info("  Edge (%s,%s) is removed, so you receive additional %g%% of course %s (value %g).", agent,item,np.round(100*weight_to_add),item, value_to_add, agents=neighbor_agent)
            surplus_to_add[neighbor_agent] = value_to_add
            if weight_for_redistribution<=0:
                break
        if surplus_donation:
            for neighbor_agent,value_to_add in surplus_to_add.items():
                add_surplus(neighbor_agent, value_to_add)


    def remove_agent_from_graph(agent):
        """
        Remove the agent from the graph, and redistribute its belongings among the neighboring agents of these items.
        """
        neighbors = list(fractional_allocation_graph.agent_neighbors(agent))
        for item in neighbors:
            remove_edge_from_graph(agent,item)

    # draw_bipartite_weighted_graph(fractional_allocation_graph, alloc.remaining_agents())
    while fractional_allocation_graph.number_of_edges()>0:
        # Look for an item leaf:
        # edges_with_fraction_near_1 = [(u,v) for u,v in fractional_allocation_graph.edges if fractional_allocation_graph[u][v]['weight'] >= 1-2*MIN_EDGE_FRACTION]
        # max_value_edge = max(edges_with_fraction_near_1, key=lambda u,v: fractional_allocation_graph[u][v]['value'])

        found_item_leaf = False
        for item_min_weight in list(alloc.remaining_items()):
            if not fractional_allocation_graph.contains_item(item_min_weight):
                continue
            item_neighbors = list(fractional_allocation_graph.item_neighbors(item_min_weight))
            for agent_min_weight in item_neighbors:
                if agent_min_weight not in fractional_allocation:
                    raise ValueError(f"agent {agent_min_weight} not in fractional allocation {fractional_allocation}")
                fractional_bundle = fractional_allocation[agent_min_weight]
                if item_min_weight not in fractional_bundle:
                    raise ValueError(f"item {item_min_weight} not in fractional bundle of agent {agent_min_weight} = {fractional_bundle}")
                if fractional_allocation[agent_min_weight][item_min_weight] >= 1-2*MIN_EDGE_FRACTION:
                    # Give an entire unit of the item to the neighbor agent
                    alloc.give(agent_min_weight, item_min_weight)
                    explanation_logger.info("Course %s is a leaf node, and you are its only neighbor, so you get all of it to yourself.", item_min_weight, agents=agent_min_weight)
                    fractional_allocation[agent_min_weight][item_min_weight] = 0
                    fractional_allocation_graph.remove_edge(agent_min_weight,item_min_weight)
                    if not agent_min_weight in alloc.remaining_agent_capacities:
                        explanation_logger.info("You have received %s and you have no remaining capacity.", alloc.bundles[agent_min_weight], agents=agent_min_weight)
                        remove_agent_from_graph(agent_min_weight)
                    explanation_logger.debug("\nfractional_allocation_graph: %s", fractional_allocation_graph)
                    found_item_leaf = True
        if found_item_leaf:
            # draw_bipartite_weighted_graph(fractional_allocation_graph, alloc.remaining_agents())
            continue

        # No item is a leaf - look for an agent leaf:
        found_agent_leaf = False
        for agent_min_weight in list(alloc.remaining_agents()):
            if not fractional_allocation_graph.contains_agent(agent_min_weight):
                continue
            agent_degree = fractional_allocation_graph.agent_degree(agent_min_weight)
            explanation_logger.debug(f"  Your degree in the consumption graph is {agent_degree}", agents=agent_min_weight)
            if agent_degree==1:
                # A leaf agent: disconnect him from his only neighbor (since it is a good)
                item_min_weight = fractional_allocation_graph.agent_first_neighbor(agent_min_weight)
                if fractional_allocation_graph.item_degree(item_min_weight)>1:
                    explanation_logger.info("\nYou are a leaf node, so you lose your only neighbor %s", item_min_weight, agents=agent_min_weight)
                    remove_agent_from_graph(agent_min_weight)
                else:
                    fractional_allocation[agent_min_weight][item_min_weight] = 0
                    fractional_allocation_graph.remove_edge(agent_min_weight,item_min_weight)
                    if agent_min_weight not in alloc.remaining_agent_capacities:
                        logger.warn("Agent %s is the only one who could get item %s, but the agent has no remaining capacity!", agent_min_weight, item_min_weight)
                    elif item_min_weight not in alloc.remaining_item_capacities:
                        logger.warn("Agent %s is the only one who could get item %s, but the item has no remaining capacity!", agent_min_weight, item_min_weight)
                    else:
                        alloc.give(agent_min_weight, item_min_weight)
                        explanation_logger.info("Both you and course %s are leaf nodes, so you get all of it to yourself.", item_min_weight, agents=agent_min_weight)

                explanation_logger.debug("\nfractional_allocation_graph: %s", fractional_allocation_graph)
                found_agent_leaf = True
                break  # after removing one agent, proceed to remove leaf items
        if found_agent_leaf:
            # draw_bipartite_weighted_graph(fractional_allocation_graph, alloc.remaining_agents())
            continue

        # No leaf at all - remove an edge with a small weight:
        edge_with_min_weight = min(fractional_allocation_graph.edges(), key=lambda edge:fractional_allocation_graph.weight(edge[0],edge[1]))
        agent_min_weight,item_min_weight = edge_with_min_weight
        min_weight = fractional_allocation_graph.weight(agent_min_weight,item_min_weight)
        # logger.warning("No leafs - removing edge %s with minimum weight %g", edge_with_min_weight, min_weight)
        explanation_logger.warning("There are no leaf nodes, but the edge %s has minimum weight %g, so it is removed.", edge_with_min_weight, min_weight, agents=agent_min_weight)
        remove_edge_from_graph(agent_min_weight,item_min_weight)

    iterated_maximum_matching(alloc)  # Avoid waste
    return alloc.sorted()


def almost_egalitarian_without_donation(alloc:AllocationBuilder, **kwargs):
    return almost_egalitarian_allocation(alloc, surplus_donation=False, **kwargs)

def almost_egalitarian_with_donation(alloc:AllocationBuilder, **kwargs):
    return almost_egalitarian_allocation(alloc, surplus_donation=True, **kwargs)


almost_egalitarian_allocation.logger = logger



def draw_bipartite_weighted_graph(G: ConsumptionGraph, top_nodes:list):
    draw_options = {
        "font_size": 10,
        "node_size": 700,
        "node_color": "yellow",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
        "with_labels": True
    }
    pos = nx.bipartite_layout(G, top_nodes)
    nx.draw(G, **draw_options, pos=pos)
    nx.draw_nx.edge_labels(G, pos, nx.get_edge_attributes(G, "weight"))
    # plt.show()



if __name__ == "__main__":
    import doctest, sys
    print("\n",doctest.testmod(), "\n")

    # sys.exit(0)

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    from fairpyx.adaptors import divide_random_instance, divide
    from fairpyx.explanations import ConsoleExplanationLogger, FilesExplanationLogger, StringsExplanationLogger

    instance = Instance(valuations=[[5, 4, 3, 2], [2, 3, 4, 5]])
    print(divide(almost_egalitarian_allocation, instance=instance, explanation_logger=ConsoleExplanationLogger()))

    # sys.exit(1)

    num_of_agents = 30
    num_of_items = 10

    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))

    console_explanation_logger = ConsoleExplanationLogger()
    files_explanation_logger = FilesExplanationLogger({
        f"s{i+1}": f"{dir_path}/logs/s{i+1}.log"
        for i in range(num_of_agents)
    }, mode='w', level=logging.INFO)
    string_explanation_logger = StringsExplanationLogger(f"s{i+1}" for i in range(num_of_agents))

    divide_random_instance(algorithm=almost_egalitarian_allocation, 
                           explanation_logger=files_explanation_logger,
                           num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
                           item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
                           random_seed=1)
