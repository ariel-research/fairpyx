from itertools import cycle

from networkx import DiGraph

import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__) #TODO understand what the flip is this

def envy(source: str, target: str, bundles: dict, val_func: callable):
    val = val_func
    source_bundle_val = sum(list(val(source, current_item) for current_item in bundles[source]))
    target_bundle_val = sum(list(val(source, current_item) for current_item in bundles[target]))
    return target_bundle_val > source_bundle_val


def initialize_graph(g: nx.DiGraph, alloc: AllocationBuilder):
    g.clear()
    g.clear_edges()
    g.add_nodes_from(alloc.instance.agents)


def update_alloc_builder(allocbuilder: AllocationBuilder, instance: Instance, bundles: dict[any, set]):
    allocbuilder.bundles = bundles
    allocbuilder.instance = instance
    #TODO modify literally everything so it makes sense , not only the bundle because you care about the output


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
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = initial_agent_order)
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}

    >>> # Example 2
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent3','Agent2']
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1','m3'], 'c2': ['m2']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':3}, 'Agent2': {'c1':3,'c2':3},'Agent3': {'c1':3,'c2':3}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':4},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':4,'m2':6,'m3':5}}
    >>> result=divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=initial_agent_order)
    >>> assert result in [{'Agent1': ['m2'], 'Agent2': ['m1'], 'Agent3': ['m3']},{'Agent1': ['m1'], 'Agent2': ['m3'], 'Agent3': ['m2']}]

    >>> # example 3 but trying to get the expected output exactly (modified valuations different than on papers)  (4 agents ,4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2},'Agent4': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
    >>> valuations = {'Agent1':{'m1':2,'m2':1,'m3':1,'m4':10},'Agent2':{'m1':1,'m2':2,'m3':1,'m4':10},'Agent3':{'m1':1,'m2':1,'m3':2,'m4':10},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':10}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=initial_agent_order)
    {'Agent1': ['m1'], 'Agent2': ['m2'], 'Agent3': ['m3'], 'Agent4': ['m4']}
    """
    #TODO Erel suggested that if we stick to working with only 1 AllocationBuilder its recommended
    # thinking of a way to make this
    # regarding remaining agents , item_capacities , conflicts , they are all the same not a big change
    # change is in remaining agent capacities in which im still confused since we'll need to have k different capacities for each agent
    # so im thinking of keeping the remianing agent capacities values as sum(capacity for capacity in agent_category_capacities[agent] for agent in agent_category_capacities.keys())
    # FROM DIVIDE IMPL ->
    # instance = Instance(valuations=valuations, agent_capacities=agent_capacities, item_capacities=item_capacities)
    # alloc = AllocationBuilder(instance)
    # agent_capacities are playing a good role ... im sticking to the sum of all capacities idea and relying on the agent_category_capacities kwarg passed to determine
    # a modification for RR is happening there is no other way.









    # TODO this is old impl
    # per_category_instance_list = per_category_sub_instance_extractor(agent_category_capacities, alloc, item_categories)
    # per_category_allocation_builder_list = [AllocationBuilder(instance) for instance in per_category_instance_list]
    # valuation_func = alloc.effective_value
    # envy_graph = nx.DiGraph()
    # envy_graph.add_nodes_from(alloc.instance.agents)
    # index = 1
    # current_bundle = dict()
    # for curr_alloc in per_category_allocation_builder_list:
    #     #print(f"order before RR category{index} is {order}")
    #     round_robin(alloc=curr_alloc, agent_order=initial_agent_order)
    #     #print(f"alloc after RR category{index} is {curr_alloc.bundles}")
    #     index += 1
    #     for agent, allocations in curr_alloc.bundles.items():
    #         current_bundle.setdefault(agent, set()).update(allocations)
    #     update_envy_graph(curr_bundles=current_bundle, valuation_func=valuation_func, envy_graph=envy_graph)
    #     if not nx.algorithms.dag.is_directed_acyclic_graph(envy_graph):
    #         # print("found cycles")
    #         # visualize_graph(envy_graph)
    #         #TODO find only 1 cycle ,  needs seperate function
    #         # TODO make sure to work with only 1 allocationbuilder , add argument list of intersection with remaining items
    #
    #         envy_cycles = list(nx.simple_cycles(envy_graph))
    #         for cycle in envy_cycles:
    #             #do bundle switching along the cycle
    #             temp_val = current_bundle[cycle[0]]
    #             for i in range(len(cycle)):
    #                 original = current_bundle[cycle[(i + 1) % len(cycle)]]
    #                 current_bundle[cycle[(i + 1) % len(cycle)]] = temp_val
    #                 temp_val = original
    #         #after eleminating a cycle we update the graph there migh be no need to touch the other cycle because it might disappear after dealing with 1 !
    #             update_envy_graph(curr_bundles=current_bundle, valuation_func=valuation_func, envy_graph=envy_graph)
    #             if  nx.algorithms.dag.is_directed_acyclic_graph(envy_graph):
    #                 # print("no need to elimiate the other cycle !")
    #                 # visualize_graph(envy_graph)
    #                 break
    #         #update the graph after cycle removal
    #         update_envy_graph(curr_bundles=current_bundle, valuation_func=valuation_func, envy_graph=envy_graph)
    #         # visualize_graph(envy_graph)
    #         # print(f"current bundle after cycle-check {current_bundle}")
    #     # topological sort
    #     initial_agent_order = list(nx.topological_sort(envy_graph))
    # update_alloc(alloc_from=per_category_allocation_builder_list[len(item_categories.keys()) - 1], alloc_to=alloc,
    #              bundles=current_bundle)


def update_envy_graph(curr_bundles:dict, valuation_func: callable, envy_graph:DiGraph):
    envy_graph.clear_edges()
    for agent1, bundle1 in curr_bundles.items():
        for agent2, bundle_agent2 in curr_bundles.items():
            if agent1 is not agent2:  # make sure w're not comparing same agent to himself
                # make sure to value with respect to the constraints of feasibility
                # since in algo 1 its always feasible because everyone has equal capacity we dont pay much attention to it
                if envy(source=agent1, target=agent2, bundles=curr_bundles, val_func=valuation_func):
                    #print(f"{agent1} envies {agent2}")  # works great .
                    # we need to add edge from the envier to the envyee
                    envy_graph.add_edge(agent1, agent2)


def visualize_graph(envy_graph):
    plt.figure(figsize=(8, 6))
    nx.draw(envy_graph, with_labels=True)
    plt.title('Basic Envy Graph')
    plt.show()


def per_category_sub_instance_extractor(agent_category_capacities: dict, alloc: AllocationBuilder,
                                        item_categories: dict):
    per_category_instance_list = []
    for category in item_categories.keys():
        sub_instance = Instance(items=[item for item in alloc.instance.items if item in item_categories[category]]
                                , valuations={
                agent: {item: alloc.instance.agent_item_value(agent, item) for item in alloc.instance.items if
                        item in item_categories[category]} for agent in alloc.instance.agents}
                                , agent_capacities={agent: capacity for agent, dic in agent_category_capacities.items()
                                                    for current_category, capacity in dic.items() if
                                                    current_category is category}  # 100% right
                                , item_capacities={item: alloc.instance.item_capacity(item) for item in
                                                   alloc.instance.items if item in item_categories[category]}
                                # 100% right
                                )
        per_category_instance_list.append(sub_instance)
        # print(sub_instance)
        # print("*************************************")
    return per_category_instance_list


if __name__ == '__main__':
    #     # order = ['Agent1', 'Agent2', 'Agent3', 'Agent4']
    #     # items = ['m1', 'm2', 'm3', 'm4']
    #     # item_categories = {'c1': ['m1', 'm2', 'm3'], 'c2': ['m4']}
    #     # agent_category_capacities = {'Agent1': {'c1': 3, 'c2': 2}, 'Agent2': {'c1': 3, 'c2': 2},
    #     #                              'Agent3': {'c1': 3, 'c2': 2},
    #     #                              'Agent4': {'c1': 3, 'c2': 2}}  # in the papers its written capacity=size(catergory)
    #     # valuations = {'Agent1': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}, 'Agent2': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10},
    #     #               'Agent3': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}, 'Agent4': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}}
    #     # instance = Instance(valuations=valuations, items=items)
    #     # divide(algorithm=per_category_round_robin, instance=Instance(valuations=valuations, items=items),
    #     #        item_categories=item_categories, agent_category_capacities=agent_category_capacities, order=order)
    #     # Example 1
    #
    order = ['Agent1', 'Agent2']
    items = ['m1', 'm2', 'm3']
    item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    valuations = {'Agent1': {'m1': 2, 'm2': 8, 'm3': 7}, 'Agent2': {'m1': 2, 'm2': 8, 'm3': 1}}
    sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    print(sum_agent_category_capacities)
    divide(algorithm=per_category_round_robin
           , instance=Instance(valuations=valuations, items=items,agent_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()})
           ,item_categories=item_categories, agent_category_capacities=agent_category_capacities, initial_agent_order=order)
    #print(type(AllocationBuilder(instance=Instance(valuations=valuations, items=items)).bundles['Agent1'])) -> set()
    # expected output ------ > {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}


def categorization_friendly_picking_sequence(alloc: AllocationBuilder, agent_order: list,item_categories:dict,agent_category_capacities:dict,target_category:str):
    # TODO we want to make it rely on a given dict of capacities instead the reamining agent capacities which is
    #  built in and used in isdone() now i still cant understand the wisdom behind my choice of giving the classic
    #  remaining_agent_capacities the sum of all their capacities (maybe the only use case is statistics after the
    #  termination but still cant tell if the remaining capacity belongs to category i or category j s.t i!=j )

    # we will stick to the concept of "for each category run RR(category[i]:Any,alloc:AllocationBuilder, <argument>(indicator to force RR only touch items which belong to category[i]))
    # if we're smart enough we could pass only agent_category_capacities , as they have all the info we need .
    """
    Allocate the given items to the given agents using the given picking sequence.
    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param agent_order: a list of indices of agents, representing the picking sequence. The agents will pick items in this order.

    >>> from fairpyx.adaptors import divide
    >>> agent_capacities = {"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3}      # 10 seats required
    >>> course_capacities = {"c1": 2, "c2": 3, "c3": 4}                       # 9 seats available
    >>> valuations = {"Alice": {"c1": 10, "c2": 8, "c3": 6}, "Bob": {"c1": 10, "c2": 8, "c3": 6}, "Chana": {"c1": 6, "c2": 8, "c3": 10}, "Dana": {"c1": 6, "c2": 8, "c3": 10}}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(picking_sequence, instance=instance, agent_order=["Alice","Bob", "Chana", "Dana","Dana","Chana","Bob", "Alice"])
    {'Alice': ['c1', 'c3'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c2', 'c3']}
    """
    remaining_category_agent_capacities={agent:agent_category_capacities[agent][target_category] for agent in agent_category_capacities} # should look {'Agenti':[k:int]}
    remaining_category_items=[x for x in alloc.remaining_items() if x in item_categories[target_category]]
    logger.info("\nPicking-sequence with items %s , agents %s, and agent-order %s", alloc.remaining_item_capacities,
                alloc.remaining_agent_capacities, agent_order)
    for agent in cycle(agent_order):
        if  len(remaining_category_agent_capacities)==0 or len(remaining_category_items)==0:                   #replaces-> alloc.isdone():  isdone impl : return len(self.remaining_item_capacities) == 0 or len(self.remaining_agent_capacities) == 0
            break
        if not agent in remaining_category_agent_capacities:
            continue
        potential_items_for_agent = set(remaining_category_items).difference(alloc.bundles[agent])# we only deal with relevant items which are in target category
        if len(potential_items_for_agent) == 0: # means you already have 1 of the same item and there is onflic or you simply no reamining items
            logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", agent,
                        alloc.remaining_item_capacities, alloc.bundles[agent])
            alloc.remove_agent_from_loop(agent)
            continue
        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.effective_value(agent, item))
        alloc.give(agent, best_item_for_agent, logger)
        remaining_category_agent_capacities[agent]-=1
        if remaining_category_agent_capacities[agent]<=0: del remaining_category_agent_capacities[agent]
        alloc.remaining_item_capacities[best_item_for_agent] -= 1
        if alloc.remaining_item_capacities[best_item_for_agent] <= 0: # by defualt this is triggered after first give() since by default each item capacity is =1
            alloc.remove_item_from_loop(best_item_for_agent)
            remaining_category_items.remove(best_item_for_agent) # equivelant for removing the item in allocationbuiler






def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()  # Make a copy of the first dictionary

    for key, values in dict2.items():
        if key in merged_dict:
            merged_dict[key].update(values)  # Update the set if the key exists
        else:
            merged_dict[key] = values.copy()  # Add a new key-value pair if it doesn't exist

    return merged_dict


def update_alloc(alloc_from: AllocationBuilder, alloc_to: AllocationBuilder, bundles: dict = None):
    #shallow copies everything but bundles is constantly extended without overriding previous info
    alloc_to.instance = alloc_from.instance
    alloc_to.remaining_agent_capacities = alloc_from.remaining_agent_capacities
    alloc_to.remaining_item_capacities = alloc_from.remaining_item_capacities
    alloc_to.remaining_agents = alloc_from.remaining_agents
    alloc_to.remaining_items = alloc_from.remaining_items
    alloc_to.bundles = merge_dicts(alloc_from.bundles, alloc_to.bundles) if bundles is None else merge_dicts(bundles,
                                                                                                             alloc_to.bundles)


# if __name__ == "__main__":
#     import doctest
#
#     doctest.testmod()


def capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict, order: list):
    """
    this is Algorithm 2 CRR (capped round-robin) algorithm TLDR: single category , may have differnt capacities
    capped in CRR stands for capped capacity for each agent unlke RR , maye have different valuations -> F-EF1 (
    feasible envy-freeness up to 1 good) allocation

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param order: a list representing the order we start with in the algorithm

        >>> # Example 1 (2 agents 1 of them with capacity of 0)
        >>> from fairpyx import  divide
        >>> order=[1,2]
        >>> items=['m1']
        >>> item_categories = {'c1': ['m1']}
        >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1}}
        >>> valuations = {'Agent1':{'m1':0},'Agent2':{'m1':420}}
        >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
        >>>{'Agent1':None,'Agent2':['m1']}

        >>> # Example 2 (3 agents , 4 items)
        >>> from fairpyx import  divide
        >>> order=[1,2,3]
        >>> items=['m1','m2','m3','m4']
        >>> item_categories = {'c1': ['m1', 'm2','m3','m4']}
        >>> agent_category_capacities = {'Agent1': {'c1':2}, 'Agent2': {'c1':2},'Agent3': {'c1':2}}
        >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1},'Agent3':{'m1':1,'m2':1,'m3':1}}
        >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
        >>> {'Agent1':['m1','m4'],'Agent2':['m2'],'Agent3':['m3']}


         >>> # Example 3  (to show that F-EF (feasible envy-free) is sometimes achievable in good scenarios)
        >>> from fairpyx import  divide
        >>> order=[1,2]
        >>> items=['m1','m2']
        >>> item_categories = {'c1': ['m1', 'm2']}
        >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':1},'Agent3': {'c1':1}}
        >>> valuations = {'Agent1':{'m1':10,'m2':5},'Agent2':{'m1':5,'m2':10}}
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
        >>> {'Agent1':['m1'],'Agent2':['m2']}

        >>> # Example 4  (4 Agents , different capacities ,extra unallocated item at the termination of the algorithm)
        >>> from fairpyx import  divide
        >>> order=[1,2,3,4]
        >>> items=['m1','m2','m3','m4','m5','m6','m7']
        >>> item_categories = {'c1': ['m1', 'm2']}
        >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1},'Agent3': {'c1':2},'Agent4': {'c1':3}}
        >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6,'m7':0},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1,'m7':0},'Agent3':{'m1':1,'m2':2,'m3':5,'m4':6,'m5':3,'m6':4,'m7':0},'Agent4':{'m1':5,'m2':4,'m3':1,'m4':2,'m5':3,'m6':6,'m7':0}}
        >>> divide(algorithm=capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
        >>> {'Agent1':[],'Agent2':['m1'],'Agent3':['m3','m4'],'Agent4':['m2','m5','m6']}

        """
    pass


def two_categories_capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      order: list):
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
            :param order: a list representing the order we start with in the algorithm

            >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> order=[1,2]
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1','m2'],'c2':['m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>>{'Agent1':['m1'],'Agent2':['m2','m3']}

            >>> # Example 2 (case of single category so we deal with it as the normal CRR)
            >>> from fairpyx import  divide
            >>> order=[1,2,3]
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':[]}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':0}, 'Agent2': {'c1':2,'c2':0},'Agent3': {'c1':0,'c2':0}}
            >>> valuations = {'Agent1':{'m1':2,'m2':3,'m3':3},'Agent2':{'m1':3,'m2':1,'m3':1},'Agent3':{'m1':10,'m2':10,'m3':10}} # in the papers agent 3 values at infinite in here we did 10  which is more than the others
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
            >>> {'Agent1':['m1'],'Agent2':['m2','m3'],'Agent3':[]} # TODO check if better do [] or NONE for an empty alloc


             >>> # Example 3  (4 agents 6 items same valuations same capacities)-> EF in best case scenario
            >>> from fairpyx import  divide
            >>> order=[2,4,1,3]
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2'],'c2': ['m3', 'm4','m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1}, 'Agent2': {'c1':1,'c2':1},'Agent3': {'c1':1,'c2':1},'Agent4': {'c1':1,'c2':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>> {'Agent1':['m4'],'Agent2':['m1','m6'],'Agent3':['m3'],'Agent4':['m2','m5']}


                >>> # Example 4  (3 agents 6 items different valuations different capacities, remainder item at the end)-> F-EF1
            >>> from fairpyx import  divide
            >>> order=[1,2,3]
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1', 'm2','m3', 'm4'],'c2': [,'m5','m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':1}, 'Agent2': {'c1':0,'c2':2},'Agent3': {'c1':0,'c2':5}}
            >>> valuations = {'Agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6},'Agent2':{'m1':6,'m2':5,'m3':4,'m4':3,'m5':2,'m6':1},'Agent3':{'m1':5,'m2':3,'m3':1,'m4':2,'m5':4,'m6':6},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1}}
            >>> divide(algorithm=two_categories_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>> {'Agent1':['m2','m3','m4'],'Agent2':['m5'],'Agent3':['m6']}# m1 remains unallocated unfortunately :-(
            """
    pass


def per_category_capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                    order: list):
    """
    this is Algorithm 4 deals with (Different Capacities, Identical Valuations), suitable for any number of categories
    CRR (per-category capped round-robin) algorithm
    TLDR: single category , may have different capacities , but have identical valuations -> F-EF1 (feasible envy-freeness up to 1 good) allocation

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionary in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param order: a list representing the order we start with in the algorithm

         >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> order=[1,2]
            >>> items=['m1','m2','m3','m4']
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':1},'Agent2':{'m1':1,'m2':1,'m3':1,'m4':1}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>>{'Agent1':['m1','m3'],'Agent2':['m2','m4']}

            >>> # Example 2 (3 agents 3 categories , different capacities )
            >>> from fairpyx import  divide
            >>> order=[1,2,3]
            >>> items=['m1','m2','m3','m4','m5','m6','m7','m8','m9']
            >>> item_categories = {'c1': ['m1','m2','m3','m4'],'c2':['m5','m6','m7'],'c3':['m8','m9']}
            >>> agent_category_capacities = {'Agent1': {'c1':0,'c2':4,'c3':4}, 'Agent2': {'c1':4,'c2':0,'c3':4},'Agent3': {'c1':4,'c2':4,'c3':0}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1,'m7':1,'m8':1,'m9':1},'Agent2':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1,'m7':1,'m8':1,'m9':1},'Agent3':{'m1':1,'m2':1,'m3':1,'m4':1,'m5':1,'m6':1,'m7':1,'m8':1,'m9':1}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>>{'Agent1':['m5','m7','m9'],'Agent2':['m1','m3','m8'],'Agent3':['m2','m4','m6']}

             >>> # Example 3 (3 agents 3 categories , 1 item per category)
            >>> from fairpyx import  divide
            >>> order=[1,2,3]
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2'],'c3':['m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':1,'c2':1,'c3':1}}
            >>> valuations = {'Agent1':{'m1':7,'m2':8,'m3':9},'Agent2':{'m1':7,'m2':8,'m3':9},'Agent3':{'m1':7,'m2':8,'m3':9}}
            >>> divide(algorithm=per_category_capped_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
            >>>{'Agent1':['m1'],'Agent2':['m2'],'Agent3':['m3']}
    """
    pass


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
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities
            >>>{'Agent1':['m1'],'Agent2':['m2','m3']}


            >>> # Example 2 ( 3 agents  with common interests in certain items)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2},'Agent3': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':0},'Agent3':{'m1':0,'m2':0,'m3':1}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            >>>{'Agent1':['m1','m3'],'Agent2':['m2'],'Agent3':[]}

             >>> # Example 3 ( 3 agents , 3 categories , with common interests, and remainder unallocated items at the end )
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':0,'c2':0,'c3':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':0,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':,'m2':0,'m3':0,'m4':0,'m5':0,'m6':1}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            >>>{'Agent1':['m1','m4'],'Agent2':['m2','m5'],'Agent3':['m6']} # m3 remains unallocated ....
   """
