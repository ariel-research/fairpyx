from itertools import cycle

from networkx import DiGraph

import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)  #TODO understand what the flip is this


def envy(source: str, target: str, bundles: dict[str,set or list ], val_func: callable, item_categories: dict,
         agent_category_capacities: dict):
    val = val_func
    source_bundle_val = sum(list(val(source, current_item) for current_item in bundles[source]))
    #target_bundle_val = sum(list(val(source, current_item) for current_item in bundles[target])) #old non feasible method
    copy= bundles[target].copy()
    target_bundle_val=0
    sorted(copy, key=lambda x: val(source, x),reverse=True) # sort target items  in the perspective of the envier
    for category in item_categories.keys():
        candidates=[item for item in copy if item in item_categories[category]] # assumes sorted in reverse (maximum comes first)
        target_bundle_val+=sum(val(source,x) for x in candidates[:agent_category_capacities[source][category]])

    #print(source_bundle_val, target_bundle_val)
    return target_bundle_val > source_bundle_val


def categorization_friendly_picking_sequence(alloc: AllocationBuilder, agent_order: list, item_categories: dict,
                                             agent_category_capacities: dict, target_category: str):
    if agent_order is None: agent_order = list(alloc.remaining_agents())
    remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
                                           agent_category_capacities if agent_category_capacities[agent][
                                               target_category] != 0}  # should look {'Agenti':[k:int]}
    remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[target_category]]
    # print(f'remaining agent capacities for {target_category}: {remaining_category_agent_capacities}')
    # print(f'remaining {target_category} items: {remaining_category_items}')
    logger.info("\nPicking-sequence with items %s , agents %s, and agent-order %s", alloc.remaining_item_capacities,
                alloc.remaining_agent_capacities, agent_order)
    for agent in cycle(agent_order):
        if len(remaining_category_agent_capacities) == 0 or len(
                remaining_category_items) == 0:  #replaces-> alloc.isdone():  isdone impl : return len(self.remaining_item_capacities) == 0 or len(self.remaining_agent_capacities) == 0
            break
        if not agent in remaining_category_agent_capacities:
            continue
        potential_items_for_agent = set(remaining_category_items).difference(
            alloc.bundles[agent])  # we only deal with relevant items which are in target category
        if len(potential_items_for_agent) == 0:  # means you already have 1 of the same item and there is onflic or you simply no reamining items
            logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", agent,
                        alloc.remaining_item_capacities, alloc.bundles[agent])
            alloc.remove_agent_from_loop(agent)
            continue
        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.effective_value(agent, item))
        alloc.give(agent, best_item_for_agent, logger)
        remaining_category_agent_capacities[agent] -= 1
        if remaining_category_agent_capacities[agent] <= 0:
            del remaining_category_agent_capacities[
                agent]  # this doesnt apply to the allocation builder since we havent finished yet

        if best_item_for_agent not in alloc.remaining_item_capacities:
            remaining_category_items.remove(best_item_for_agent)  # equivelant for removing the item in allocationbuiler


def update_envy_graph(curr_bundles: dict, valuation_func: callable, envy_graph: DiGraph, item_categories: dict,
                      agent_category_capacities: dict):
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


def per_category_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                             initial_agent_order: list):
    """
    this is the Algorithm 1 from the paper
    per category round-robin is an allocation algorithm which guarantees EF1 (envy-freeness up to 1 good) allocation
    under settings in which agent-capacities are equal across all agents,
    no capacity-inequalities are allowed since this algorithm doesnt provie a cycle-prevention mechanism
    TLDR: same partition constriants , same capacities , may have different valuations across agents  -> EF1 allocation

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
    :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
    paired with a dictionary of category-capacity.
    :param initial_agent_order: a list representing the order we start with in the algorithm

    >>> # Example 1
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    >>> valuations = {'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}

    >>> # Example 2
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent3','Agent2']
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1','m3'], 'c2': ['m2']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':3}, 'Agent2': {'c1':3,'c2':3},'Agent3': {'c1':3,'c2':3}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':4},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':4,'m2':6,'m3':5}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> result=divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    >>> assert result in [{'Agent1': ['m2'], 'Agent2': ['m1'], 'Agent3': ['m3']},{'Agent1': ['m1'], 'Agent2': ['m3'], 'Agent3': ['m2']}]

    >>> # example 3 but trying to get the expected output exactly (modified valuations different than on papers)  (4 agents ,4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2},'Agent4': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
    >>> valuations = {'Agent1':{'m1':2,'m2':1,'m3':1,'m4':10},'Agent2':{'m1':1,'m2':2,'m3':1,'m4':10},'Agent3':{'m1':1,'m2':1,'m3':2,'m4':10},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':10}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    {'Agent1': ['m1'], 'Agent2': ['m2'], 'Agent3': ['m3'], 'Agent4': ['m4']}
    """
    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    for category in item_categories.keys():
        categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                                 category)  # this is RR without wrapper
        update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                          item_categories=item_categories, agent_category_capacities=agent_category_capacities)
        #print(f"{category} bundle is {alloc.bundles}")
        #visualize_graph(envy_graph)
        if not nx.algorithms.dag.is_directed_acyclic_graph(envy_graph):
            for cycle in nx.simple_cycles(envy_graph):
                #do bundle switching along the cycle
                temp_val = alloc.bundles[cycle[0]]
                for i in range(len(cycle)):
                    original = alloc.bundles[cycle[(i + 1) % len(cycle)]]
                    alloc.bundles[cycle[(i + 1) % len(cycle)]] = temp_val
                    temp_val = original
                #after eleminating a cycle we update the graph there migh be no need to touch the other cycle because it might disappear after dealing with 1 !
                update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                                  item_categories=item_categories, agent_category_capacities=agent_category_capacities)
                if nx.algorithms.dag.is_directed_acyclic_graph(envy_graph):
                    break
            #update the graph after cycle removal
            update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                              item_categories=item_categories, agent_category_capacities=agent_category_capacities)
        current_order = list(nx.topological_sort(envy_graph))


def capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                       initial_order: list):
    """
    this is Algorithm 2 CRR (capped round-robin) algorithm TLDR: single category , may have differnt capacities
    capped in CRR stands for capped capacity for each agent unlke RR , maye have different valuations -> F-EF1 (
    feasible envy-freeness up to 1 good) allocation

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param initial_order: a list representing the order we start with in the algorithm

    >>> # Example 1 (2 agents 1 of them with capacity of 0)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1']
    >>> item_categories = {'c1': ['m1']}
    >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1}}
    >>> valuations = {'Agent1':{'m1':0},'Agent2':{'m1':420}}
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order = order)
    {'Agent1': [], 'Agent2': ['m1']}

    >>> # Example 2 (3 agents , 4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3','m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':2}, 'Agent2': {'c1':2},'Agent3': {'c1':2}}
    >>> valuations = {'Agent1':{'m1':10,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':10,'m3':1},'Agent3':{'m1':1,'m2':1,'m3':10}}
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
    {'Agent1': ['m1', 'm4'], 'Agent2': ['m2'], 'Agent3': ['m3']}


    >>> # Example 3  (to show that F-EF (feasible envy-free) is sometimes achievable in good scenarios)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1','m2']
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':1},'Agent3': {'c1':1}}
    >>> valuations = {'Agent1':{'m1':10,'m2':5},'Agent2':{'m1':5,'m2':10}}
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order = order)
    {'Agent1': ['m1'], 'Agent2': ['m2']}

    >>> # Example 4  (4 Agents , different capacities ,extra unallocated item at the termination of the algorithm)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4','m5','m6','m7']
    >>> item_categories = {'c1': ['m1', 'm2','m3','m4','m5','m6','m7']}
    >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1},'Agent3': {'c1':2},'Agent4': {'c1':3}}
    >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6,'m7':0},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1,'m7':0},'Agent3':{'m1':1,'m2':2,'m3':5,'m4':6,'m5':3,'m6':4,'m7':0},'Agent4':{'m1':5,'m2':4,'m3':1,'m4':2,'m5':3,'m6':6,'m7':0}}
    >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
    {'Agent1': [], 'Agent2': ['m1'], 'Agent3': ['m3', 'm4'], 'Agent4': ['m2', 'm5', 'm6']}
        """
    #TODO i strongly believe we could use RR from the first algorithm simply run it one time since there is no more than 1 category
    # no need for envy graphs whatsoever
    current_order = initial_order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             'c1')  # this is RR without wrapper


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
#

def two_categories_capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      initial_order: list):
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
            :param initial_order: a list representing the order we start with in the algorithm

            >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2']
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1','m2'],'c2':['m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':10,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3']}

            >>> # Example 2 (case of single category so we deal with it as the normal CRR)
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2','Agent3']
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':[]}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':0}, 'Agent2': {'c1':2,'c2':0},'Agent3': {'c1':0,'c2':0}}
            >>> valuations = {'Agent1':{'m1':10,'m2':3,'m3':3},'Agent2':{'m1':3,'m2':1,'m3':1},'Agent3':{'m1':10,'m2':10,'m3':10}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3'], 'Agent3': []}


             >>> # Example 3  (4 agents 6 items same valuations same capacities)-> EF in best case scenario
            >>> from fairpyx import  divide
            >>> order=['Agent2','Agent4','Agent1','Agent3']
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2'],'c2': ['m3', 'm4','m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1}, 'Agent2': {'c1':1,'c2':1},'Agent3': {'c1':1,'c2':1},'Agent4': {'c1':1,'c2':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':10,'m5':1,'m6':1},'Agent2':{'m1':10,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':1,'m2':1,'m3':10,'m4':1,'m5':1,'m6':1},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':10,'m6':1}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
            {'Agent1': ['m4'], 'Agent2': ['m1', 'm6'], 'Agent3': ['m3'], 'Agent4': ['m2', 'm5']}


            >>> # Example 4  (3 agents 6 items different valuations different capacities, remainder item at the end)-> F-EF1
            >>> from fairpyx import  divide
            >>> order=['Agent1','Agent2','Agent3']
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2','m3', 'm4'],'c2': ['m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':1}, 'Agent2': {'c1':0,'c2':2},'Agent3': {'c1':0,'c2':5}}
            >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1},'Agent3':{'m1':5,'m2':3,'m3':1,'m4':2,'m5':4,'m6':6}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_order=order)
            {'Agent1': ['m2', 'm3', 'm4'], 'Agent2': ['m5'], 'Agent3': ['m6']}
            >>> # m1 remains unallocated unfortunately :-(
            """
    current_order = initial_order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             'c1')  #calling CRR on first category
    current_order.reverse()  #reversing order
    categorization_friendly_picking_sequence(alloc, current_order, item_categories, agent_category_capacities,
                                             'c2')  # calling CRR on first category


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()


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
        #capped_round_robin(alloc=alloc,item_categories=item_categories,agent_category_capacities=agent_category_capacities,initial_order=initial_order)# TODO its more appropriate to use this but i cant target specific category since it only deals with 'c1'
        categorization_friendly_picking_sequence(alloc=alloc, agent_order=current_order,
                                                 item_categories=item_categories,
                                                 agent_category_capacities=agent_category_capacities,
                                                 target_category=category)
        update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                          item_categories=item_categories, agent_category_capacities=agent_category_capacities)
        current_order = list(nx.topological_sort(envy_graph))


# if __name__ == "__main__":
#     import doctest
#
#     doctest.testmod()
    # import networkx as nx
    # import matplotlib.pyplot as plt
    #
    # # Create an empty bipartite graph
    # G = nx.Graph()
    #
    # # Add nodes to the graph. Nodes in set A are labeled 0 to 4, and nodes in set B are labeled 'a' to 'e'.
    # G.add_nodes_from([0, 1, 2, 3, 4], bipartite=0)  # Set A nodes
    # G.add_nodes_from(['a', 'b', 'c', 'd', 'e'], bipartite=1)  # Set B nodes
    #
    # # Add edges between nodes in set A and set B
    # edges = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
    # G.add_edges_from(edges)
    #
    # # Draw the bipartite graph
    # pos = nx.bipartite_layout(G, [0, 1, 2, 3, 4])  # Specify the nodes in set A for layout
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_weight='bold')
    #
    # # Display the graph
    # plt.show()


def iterated_priority_matching(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict):
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
            {'Agent1':['m1'],'Agent2':['m2','m3']}


            >>> # Example 2 ( 3 agents  with common interests in certain items)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2},'Agent3': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':0},'Agent3':{'m1':0,'m2':0,'m3':1}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1':['m1','m3'],'Agent2':['m2'],'Agent3':[]}

             >>> # Example 3 ( 3 agents , 3 categories , with common interests, and remainder unallocated items at the end )
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':0,'c2':0,'c3':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':0,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':0,'m2':0,'m3':0,'m4':0,'m5':0,'m6':1}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)# m3 remains unallocated ....
            {'Agent1':['m1','m4'],'Agent2':['m2','m5'],'Agent3':['m6']}
   """
    envy_graph = nx.DiGraph()
    current_order = [agent for agent in alloc.remaining_agents()]
    valuation_func = alloc.instance.agent_item_value
    for category in item_categories.keys():
        maximum_capacity=max([agent_category_capacities[agent][category] for agent in agent_category_capacities.keys()])
        remaining_category_agent_capacities = {agent: agent_category_capacities[agent][category] for agent in
                                               agent_category_capacities if agent_category_capacities[agent][
                                                   category] != 0}
        current_item_list = [item for item in alloc.remaining_items() if item in item_categories[category]]
        for i in range(maximum_capacity):
            # creation of agent-item graph and adding edges with weight based on current order
            agent_item_bipartite_graph=nx.Graph()
            agent_item_bipartite_graph.add_nodes_from([agent for agent in  remaining_category_agent_capacities.keys()], bipartite=0) # cant use alloc.remaining_agents() since it doesnt support multiple categorization
            agent_item_bipartite_graph.add_nodes_from([item for item in alloc.remaining_items() if item in item_categories[category]], bipartite=1)
            current_agent_list=[agent for agent in current_order if agent in remaining_category_agent_capacities.keys()]
            #current_item_list=[item for item in alloc.remaining_items() if item in item_categories[category]]
            #print(f'category={category}  index={i} \ncurrent_agent_list is={current_agent_list} , current_item_list={current_item_list}\n current_order={current_order}')
            for agent in current_agent_list:
                counter= len(current_agent_list)
                for item in current_item_list:
                    if valuation_func(agent,item)!=0:
                        agent_item_bipartite_graph.add_edge(agent,item,weight=counter)
                counter-=1
            update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                              item_categories=item_categories, agent_category_capacities=agent_category_capacities)
            #visualize_graph(envy_graph)
            sort=list(nx.topological_sort(envy_graph))
            current_order = current_order if not sort else sort

            print(current_order)
            #TODO priority matching based on topological sort
            for match in nx.max_weight_matching(agent_item_bipartite_graph):
                if match[0].startswith('A'):
                    alloc.give(match[0], match[1], logger)
                    remaining_category_agent_capacities[match[0]] -= 1
                    if remaining_category_agent_capacities[match[0]] <= 0:
                        del remaining_category_agent_capacities[match[0]]
                else:
                    alloc.give(match[1], match[0], logger)
                    remaining_category_agent_capacities[match[1]] -= 1
                    if remaining_category_agent_capacities[match[1]] <= 0:
                        del remaining_category_agent_capacities[match[1]]
        for item in current_item_list:
            for agent,capacity in remaining_category_agent_capacities.items():
                if capacity>0:
                    alloc.give(agent,item, logger)
        #randomly allocate remaining items to agents with remaining capacity







if __name__ == "__main__":
    import doctest

    doctest.testmod()