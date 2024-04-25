import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import matplotlib.pyplot as plt


def envy(source: str, target: str, alloc: AllocationBuilder, val_func: callable):
    val = val_func
    source_bundle_val = sum(list(val(source, current_item) for current_item in alloc.bundles[source]))
    target_bundle_val = sum(list(val(source, current_item) for current_item in alloc.bundles[target]))
    return target_bundle_val > source_bundle_val


def initialize_graph(g: nx.DiGraph, alloc: AllocationBuilder):
    g.clear()
    g.clear_edges()
    g.add_nodes_from(alloc.instance.agents)


def per_category_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                             order: list):
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
    :param order: a list representing the order we start with in the algorithm

    >>> # Example 1
    >>> from fairpyx import  divide
    >>> order=[1,2]
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    >>> valuations = {'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
    >>>{'Agent1':['m1','m3'],'Agent2':['m2']}

    >>> # Example 2
    >>> from fairpyx import  divide
    >>> order=[1,3,2]
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2','m3']}
    >>> agent_category_capacities = {'Agent1': {'c1':3}, 'Agent2': {'c1':3},'Agent3': {'c1':3}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':5},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':5,'m2':6,'m3':5}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
    >>> {'Agent1':['m2'],'Agent2':['m1'],'Agent3':['m3']}


     >>> # Example 3  (4 agents ,4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2},'Agent4': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
    >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1,'m4':10},'Agent2':{'m1':1,'m2':1,'m3':1,'m4':10},'Agent3':{'m1':1,'m2':1,'m3':1,'m4':10},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':10}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
    >>> {'Agent1':['m1'],'Agent2':['m2'],'Agent3':['m3'],'Agent4':['m4']}
    """
    per_category_instance_list = per_category_sub_instance_extractor(agent_category_capacities, alloc, item_categories)
    valuation_func = alloc.effective_value
    envy_graph = nx.DiGraph()
    envy_graph.add_nodes_from(alloc.instance.agents)
    index = 1
    current_bundle = dict()
    for curr in per_category_instance_list:
        curr_alloc = AllocationBuilder(curr)
        round_robin(alloc=curr_alloc, agent_order=order)
        #visualize_graph(envy_graph)
        #print(f"bundles after RR#{index}{curr_alloc.bundles}")
        index += 1
        for agent, allocations in curr_alloc.bundles.items():
            current_bundle.setdefault(agent, set()).update(allocations)
        update_envy_graph(curr_alloc, valuation_func, envy_graph)
        if not nx.algorithms.dag.is_directed_acyclic_graph(envy_graph):
            envy_cycles = list(nx.simple_cycles(envy_graph))
            for cycle in envy_cycles:
                #do bundle switching along the cycle
                temp_val = current_bundle[cycle[0]]
                for i in range(len(cycle)):
                    original = current_bundle[cycle[(i + 1) % len(cycle)]]
                    current_bundle[cycle[(i + 1) % len(cycle)]] = temp_val
                    temp_val = original
            #update the graph after cycle removal
            initialize_graph(envy_graph, curr_alloc)
            non_cyclic_alloc = AllocationBuilder(curr_alloc.instance)
            non_cyclic_alloc.bundles = current_bundle
            update_envy_graph(non_cyclic_alloc, valuation_func, envy_graph)
            #visualize_graph(envy_graph)
        # topological sort
        order = list(nx.topological_sort(envy_graph))
    #     print("*****************************")
    # print(f"FINAL ALLOCATION AFTER TERMINATION OF THE ALGORITHM IS {current_bundle}")
    return current_bundle


def update_envy_graph(curr_alloc, valuation_func: callable, envy_graph):
    for agent1, bundle1 in curr_alloc.bundles.items():
        for agent2, bundle_agent2 in curr_alloc.bundles.items():
            if agent1 is not agent2:  # make sure w're not comparing same agent to himself
                # make sure to value with respect to the constraints of feasibility
                # since in algo 1 its always feasible because everyone has equal capacity we dont pay much attention to it
                if envy(source=agent1, target=agent2, alloc=curr_alloc, val_func=valuation_func):
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
        sub_instance = Instance(items=[item for item in items if item in item_categories[category]]
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
    # order = ['Agent1', 'Agent2', 'Agent3', 'Agent4']
    # items = ['m1', 'm2', 'm3', 'm4']
    # item_categories = {'c1': ['m1', 'm2', 'm3'], 'c2': ['m4']}
    # agent_category_capacities = {'Agent1': {'c1': 3, 'c2': 2}, 'Agent2': {'c1': 3, 'c2': 2},
    #                              'Agent3': {'c1': 3, 'c2': 2},
    #                              'Agent4': {'c1': 3, 'c2': 2}}  # in the papers its written capacity=size(catergory)
    # valuations = {'Agent1': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}, 'Agent2': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10},
    #               'Agent3': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}, 'Agent4': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 10}}
    # instance = Instance(valuations=valuations, items=items)
    # divide(algorithm=per_category_round_robin, instance=Instance(valuations=valuations, items=items),
    #        item_categories=item_categories, agent_category_capacities=agent_category_capacities, order=order)
    # Example 1

    order = ['Agent1', 'Agent2']
    items = ['m1', 'm2', 'm3']
    item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    valuations = {'Agent1': {'m1': 2, 'm2': 8, 'm3': 7}, 'Agent2': {'m1': 2, 'm2': 8, 'm3': 1}}
    divide(algorithm=per_category_round_robin, instance=Instance(valuations=valuations, items=items),
           item_categories=item_categories, agent_category_capacities=agent_category_capacities, order=order)
    # expected output ------ > {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}


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
