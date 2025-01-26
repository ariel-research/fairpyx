"""
An implementation of the algorithms in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2020), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import math
import random
from itertools import cycle

from networkx import DiGraph
import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import logging
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def per_category_round_robin(alloc: AllocationBuilder, item_categories: dict[str,list], agent_category_capacities: dict[str,dict[str,int]],
                             initial_agent_order: list,callback:callable=None):
    """
    this is the Algorithm 1 from the paper
    per category round-robin is an allocation algorithm which guarantees EF1 (envy-freeness up to 1 good) allocation
    under settings in which agent-capacities are equal across all agents,
    no capacity-inequalities are allowed since this algorithm doesnt provie a cycle-prevention mechanism
    TLDR: same partition, same capacities, may have different valuations across agents  -> EF1 allocation

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
    #input validation
    helper_validate_item_categories(item_categories)
    helper_validate_duplicate(initial_agent_order)
    helper_validate_duplicate(
        [item for category in item_categories.keys() for item in item_categories[category]])
    helper_validate_capacities(is_identical=True, agent_category_capacities=agent_category_capacities)
    helper_validate_valuations(agent_item_valuations=alloc.instance._valuations)
    # end validation
    logger.info(
        f"Running per_category_round_robin. Initial allocation -> {alloc.bundles} \n item_categories -> {item_categories} \n agent_category_capacities -> {agent_category_capacities} \n -> initial_agent_order -> {initial_agent_order}\n ")
    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    iter=0
    for category in item_categories.keys():
        iter+=1
        logger.info(f'\nCurrent category -> {category}')
        logger.info(f'Envy graph before RR -> {envy_graph.edges}')
        helper_categorization_friendly_picking_sequence(alloc, current_order, item_categories[category], agent_category_capacities, category)
        envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities,callback=callback,category=category,iteration=iter)
        logger.info(f'Envy graph after  RR -> {envy_graph.edges}')
        if not nx.is_directed_acyclic_graph(envy_graph):
            logger.info("Cycle removal started ")
            envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories,
                                              agent_category_capacities,callback=callback,category=category,iteration=iter)
            logger.info('Cycle removal ended successfully ')
        current_order = list(nx.topological_sort(envy_graph))
        logger.info(f"Topological sort -> {current_order} \n***************************** ")
    logger.info(f'alloc after per-category RR ->{alloc.bundles}')

def capped_round_robin(alloc: AllocationBuilder,item_categories: dict[str,list], agent_category_capacities: dict[str,dict[str,int]],
                             initial_agent_order: list, target_category: str):
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
    #logger.info(f'**********\nRunning capped_round_robin\n**********')

    #input validation
    helper_validate_item_categories(item_categories)
    helper_validate_duplicate(initial_agent_order)
    helper_validate_duplicate(
        [item for category in item_categories.keys() for item in item_categories[category]])
    helper_validate_capacities(agent_category_capacities=agent_category_capacities)
    helper_validate_valuations(agent_item_valuations=alloc.instance._valuations)
    if target_category not in item_categories:
        raise ValueError(f"Target category mistyped or not found: {target_category}")
    # end validation
    # no need for envy graphs whatsoever
    current_order = initial_agent_order
    logger.info(f'**********\nRunning Capped Round Robin.  initial_agent_order -> {initial_agent_order}')
    helper_categorization_friendly_picking_sequence(alloc, current_order, item_categories[target_category], agent_category_capacities,
                                                    target_category=target_category)  # this is RR without wrapper
    logger.info(f'alloc after CRR -> {alloc.bundles}')

def two_categories_capped_round_robin(alloc: AllocationBuilder,item_categories: dict[str,list], agent_category_capacities: dict[str,dict[str,int]],
                             initial_agent_order: list, target_category_pair: tuple[str]):
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
    #logger.info(f'**********\nRunning two_categories_capped_round_robin')

    #validate input
    helper_validate_item_categories(item_categories)
    helper_validate_duplicate(initial_agent_order)
    helper_validate_duplicate(
        [item for category in item_categories.keys() for item in item_categories[category]])
    helper_validate_capacities(agent_category_capacities=agent_category_capacities)
    helper_validate_valuations(agent_item_valuations=alloc.instance._valuations)
    if not all(item in item_categories for item in target_category_pair):
        raise ValueError(
            f"Not all elements of the tuple {target_category_pair} are in the categories list {list(item_categories.keys())}.")
    #end validation
    current_order = initial_agent_order
    logger.info(f'**********\nRunning two_categories_capped_round_robin, initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating cagetory {target_category_pair[0]}')
    helper_categorization_friendly_picking_sequence(alloc, current_order, item_categories[target_category_pair[0]], agent_category_capacities,
                                                    target_category=target_category_pair[0])  #calling CRR on first category
    logger.info(f'alloc after CRR#{target_category_pair[0]} ->{alloc.bundles}')
    current_order.reverse()  #reversing order
    logger.info(f'reversed initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating cagetory {target_category_pair[1]}')
    helper_categorization_friendly_picking_sequence(alloc, current_order, item_categories[target_category_pair[1]], agent_category_capacities,
                                                    target_category=target_category_pair[1])  # calling CRR on second category
    logger.info(f'alloc after CRR#{target_category_pair[1]} ->{alloc.bundles}')


def per_category_capped_round_robin(alloc: AllocationBuilder,item_categories: dict[str,list], agent_category_capacities: dict[str,dict[str,int]],
                             initial_agent_order: list,callback:callable=None):
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
        :param initial_agent_order: a list representing the order we start with in the algorithm

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
            >>> order=['Agent2','Agent3','Agent1']
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
    #logger.info(f'**********\nRunning per-category-CRR\n**********')

    #validate input
    helper_validate_item_categories(item_categories)
    helper_validate_duplicate(initial_agent_order)
    helper_validate_duplicate(
        [item for category in item_categories.keys() for item in item_categories[category]])
    helper_validate_capacities(agent_category_capacities=agent_category_capacities)
    helper_validate_valuations(is_identical=True, agent_item_valuations=alloc.instance._valuations)
    #end validation

    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    logger.info(f'**********\nRunning Per-Category Capped Round Robin, initial_agent_order->{initial_agent_order}')
    iter=0
    for category in item_categories.keys():
        iter+=1
        helper_categorization_friendly_picking_sequence(alloc=alloc, agent_order=current_order,
                                                        items_to_allocate=item_categories[category],
                                                        agent_category_capacities=agent_category_capacities,
                                                        target_category=category)

        envy_graph = helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func,
                                              item_categories=item_categories,
                                              agent_category_capacities=agent_category_capacities,category=category,iteration=iter,callback=callback)

        current_order = list(nx.topological_sort(envy_graph))
        logger.info(f'alloc after RR in category ->{category} is ->{alloc.bundles}.\n Envy graph nodes->{envy_graph.nodes} edges->{envy_graph.edges}.\ntopological sort->{current_order}')
    logger.info(f'allocation after termination of algorithm4 -> {alloc.bundles}')


def iterated_priority_matching(alloc: AllocationBuilder, item_categories: dict[str,list], agent_category_capacities: dict[str,dict[str,int]],callback:callable=None):
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
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':0},'Agent3':{'m1':0,'m2':0,'m3':0}}
            >>> #divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1': ['m1', 'm3'], 'Agent2': ['m2'], 'Agent3': []}

            >>> # Example 3 ( 3 agents , 3 categories , with common interests, and remainder unallocated items at the end )
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3','m4','m5','m6']
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':0,'c2':0,'c3':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':0,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':0,'m2':0,'m3':0,'m4':0,'m5':0,'m6':1}}
            >>> #divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)# m3 remains unallocated ....
            {'Agent1': ['m1', 'm5', 'm6'], 'Agent2': ['m2', 'm4'], 'Agent3': []}
   """
    logger.info(f'**********\nRunning Iterated priority matching')
    #validate input
    helper_validate_item_categories(item_categories)
    helper_validate_duplicate(
        [item for category in item_categories.keys() for item in item_categories[category]])
    helper_validate_capacities(agent_category_capacities=agent_category_capacities)
    helper_validate_valuations(agent_item_valuations=alloc.instance._valuations, is_binary=True)

    #end validation
    #logger.info("Running Iterated Priority Matching")
    envy_graph = nx.DiGraph()
    envy_graph.add_nodes_from(alloc.remaining_agents())  # adding agent nodes (no edges involved yet)
    current_order = list(alloc.remaining_agents())  # in this algorithm no need for initial_agent_order
    valuation_func = alloc.instance.agent_item_value

    for category in item_categories.keys():
        maximum_capacity = max(
            [agent_category_capacities[agent][category] for agent in
             agent_category_capacities.keys()])# for the sake of inner iteration
        logger.info(f'\nCategory {category}, Th=max(kih) is -> {maximum_capacity}')
        remaining_category_agent_capacities = {
            agent: agent_category_capacities[agent][category] for agent in agent_category_capacities if
            agent_category_capacities[agent][category] != 0
        }  # dictionary of the agents paired with capacities with respect to the current category we're dealing with

        # remaining_category_items = helper_update_item_list(alloc, category, item_categories)  # items we're dealing with with respect to the category
        remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[category]]
        current_agent_list = helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)  #  items we're dealing with with respect to the constraints
        logger.info(f'remaining_category_items before priority matching in category:{category}-> {remaining_category_items}')
        logger.info(f'current_agent_list before priority matching in category:{category} -> {current_agent_list}')
        for i in range(maximum_capacity):  # as in papers we run for the length of the maximum capacity out of all agents for the current category
            # Creation of agent-item graph
            agent_item_bipartite_graph = helper_create_agent_item_bipartite_graph(
                agents=current_agent_list,  # remaining agents
                items=[item for item in alloc.remaining_items() if item in item_categories[category]],
                # remaining items
                valuation_func=valuation_func,
              # remaining agents with respect to the order
            )  # building the Bi-Partite graph
            if callback:
                callback(helper_generate_bipartite_graph_base64(agent_item_bipartite_graph,iteration=i, category=category))

            # Creation of envy graph
            # updating envy graph with respect to matching (first iteration we get no envy, cause there is no matching)
            envy_graph = helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func,
                                                  item_categories=item_categories,
                                                  agent_category_capacities=agent_category_capacities,callback=callback,category=category,iteration=i)
            #topological sort (papers prove graph is always a-cyclic)
            topological_sort = list(nx.topological_sort(envy_graph))
            logger.info(f'topological sort is -> {topological_sort}')
            current_order = current_order if not topological_sort else topological_sort
            # Perform priority matching
            helper_priority_matching(agent_item_bipartite_graph, current_order, alloc,
                                     remaining_category_agent_capacities)  # deals with eliminating finished agents from agent_category_capacities
            logger.info(f'allocation after priority matching in category:{category} & i:{i} -> {alloc.bundles}')
            remaining_category_items = helper_update_item_list(alloc, category,
                                                        item_categories)  # important to update the item list after priority matching.
            current_agent_list = helper_update_ordered_agent_list(current_order,
                                                                  remaining_category_agent_capacities)  # important to update the item list after priority matching.
            logger.info(f'current_item_list after priority matching in category:{category} & i:{i} -> {remaining_category_items}')
            logger.info(f'current_agent_list after priority matching in category:{category} & i:{i} -> {current_agent_list}')

        agents_with_remaining_capacities = [agent for agent,capacity in remaining_category_agent_capacities.items() if capacity>0]
        logger.info(f'remaining_category_agent_capacities of agents capable of carrying arbitrary item ->{remaining_category_agent_capacities}')
        logger.info(f'Using round-robin to allocate the items that were not allocated in the priority matching ->{remaining_category_items}')
        if  remaining_category_items and remaining_category_agent_capacities:
            helper_categorization_friendly_picking_sequence(alloc, agents_with_remaining_capacities, item_categories[category], agent_category_capacities={agent:{category:remaining_category_agent_capacities[agent]} for agent in remaining_category_agent_capacities.keys()}, target_category=category)
    logger.info(f'FINAL ALLOCATION IS -> {alloc.bundles}')



 # helper functions section :


def helper_envy(source: str, target: str, bundles: dict[str, set or list], val_func: callable, item_categories: dict[str,list],
                agent_category_capacities: dict[str,dict[str,int]]):
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
        Example 1: example of 2 agents in which they have different feasiblity constraints(different caps)
        >>> bundles = {'agent1': {'m1'}, 'agent2': {'m2','m3','m4'}}
        >>> items=[]
        >>> item_categories = {'c1':['m1','m2','m3','m4']}
        >>> agent_category_capacities = {'agent1': {'c1': 1}, 'agent2': {'c1': 4}}
        >>> valuations={'agent1': {'m1': 0, 'm2': 0, 'm3': 0, 'm4': 0},'agent2': {'m1': 10, 'm2': 3, 'm3': 2, 'm4': 1}}
        >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
        >>> alloc.bundles=bundles
        >>> val_func = lambda agent, item : valuations[agent][item]
        >>> envy_graph = helper_update_envy_graph(curr_bundles=bundles,valuation_func=val_func,item_categories=item_categories,agent_category_capacities=agent_category_capacities)
        >>> helper_envy('agent1', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent2', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True

        Example 2: example of a cycle between 2 agents ,with envy-check before and after elimination of cycles
        >>> bundles = {'agent1': {'m1', 'm2'}, 'agent2': {'m3', 'm4'}}
        >>> items=['m1','m2','m3','m4']
        >>> item_categories = {'c1':['m1','m2','m3','m4']}
        >>> agent_category_capacities = {'agent1': {'c1': 2}, 'agent2': {'c1': 2}}
        >>> valuations={'agent1': {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4},'agent2': {'m1': 4, 'm2': 3, 'm3': 2, 'm4': 1}}
        >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
        >>> alloc.bundles=bundles
        >>> val_func = lambda agent, item: {'agent1': {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4},'agent2': {'m1': 4, 'm2': 3, 'm3': 2, 'm4': 1}}[agent][item]
        >>> envy_graph = helper_update_envy_graph(curr_bundles=bundles,valuation_func=val_func,item_categories=item_categories,agent_category_capacities=agent_category_capacities)
        >>> helper_envy('agent1', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> helper_envy('agent2', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc=alloc,valuation_func=val_func,agent_category_capacities=agent_category_capacities,item_categories=item_categories)
        >>> helper_envy('agent1', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent2', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False

        Example 3: example of a double cycle between 3 agents ,with envy-check before and after elimination of cycles
        >>> bundles = {'agent1': {'m1'}, 'agent2': {'m2'}, 'agent3': {'m3'}}
        >>> items=['m1','m2','m3']
        >>> item_categories = {'c1':['m1','m2','m3']}
        >>> agent_category_capacities = {'agent1': {'c1': 1}, 'agent2': {'c1': 1},'agent3': {'c1': 1}}
        >>> valuations = {'agent1': {'m1': 5, 'm2': 6,'m3': 5}, 'agent2': {'m1': 6, 'm2': 5 ,'m3':6},'agent3': {'m1': 5, 'm2': 6 ,'m3':5}}
        >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
        >>> alloc.bundles=bundles
        >>> val_func = lambda agent, item: valuations[agent][item]
        >>> envy_graph = helper_update_envy_graph(curr_bundles=bundles,valuation_func=val_func,item_categories=item_categories,agent_category_capacities=agent_category_capacities)
        >>> helper_envy('agent1', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> helper_envy('agent2', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> helper_envy('agent2', 'agent3', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> helper_envy('agent3', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        True
        >>> helper_envy('agent3', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent1', 'agent3', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc=alloc,valuation_func=val_func,agent_category_capacities=agent_category_capacities,item_categories=item_categories)
        >>> helper_envy('agent1', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent2', 'agent1', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent2', 'agent3', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        >>> helper_envy('agent3', 'agent2', alloc.bundles, val_func, item_categories, agent_category_capacities)
        False
        """
    #logger.info(f'**********\nRunning helper_envy\n**********')
    #validate input
    if isinstance(bundles, dict):
        for key, val in bundles.items():
            if not isinstance(key, str) or not isinstance(val, (list, set)):
                raise ValueError(f"Bundles not structured properly.{bundles} type(va;) -> {type(val)}")
        if not callable(val_func):
            raise ValueError("val_func must be callable.")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Source and target not structured properly.")
        if source not in bundles or target not in bundles:
            raise ValueError(f"Source or target agents are mistyped/not found in {bundles}.")
        helper_validate_item_categories(item_categories)
        helper_validate_capacities(agent_category_capacities)
    #end validation

    val = val_func
    source_bundle_val = sum(list(val(source, current_item) for current_item in bundles[source]))
    logger.debug(f'    source {source} full-bundle value -> {source_bundle_val}')
    copy = bundles[target].copy()
    target_feasible_bundle_val = 0
    target_feasible_bundle=[]
    sorted(copy, key=lambda x: val(source, x), reverse=True)  # sort target items  in the perspective of the envier in desc order
    for category in item_categories.keys():# for each category
        candidates = [item for item in copy if
                      item in item_categories[category]]  # assumes items sorted in reverse based on source agent's valuation (maximum comes first)
        curr_best_subset=candidates[:agent_category_capacities[source][category]]
        target_feasible_bundle.append(curr_best_subset)
        target_feasible_bundle_val += sum(val(source, x) for x in curr_best_subset)# take as much as source agent cant carry (Kih)
        logger.debug(f'    best feasible sub_bundle for {source} from {target} in category {category} is -> {candidates[:agent_category_capacities[source][category]]} and overall value(including all categories till now) is -> {target_feasible_bundle_val}')

    logger.debug(f'    source{source} bundle is -> {bundles[source]} and its value is -> {source_bundle_val}\n target {target} best feasible bundle in the perspective of {source} is -> {target_feasible_bundle} and its value is -> {target_feasible_bundle_val}')
    logger.debug(f'     Does {source} envy {target} ? -> {target_feasible_bundle_val > source_bundle_val}')
    return target_feasible_bundle_val > source_bundle_val

def helper_categorization_friendly_picking_sequence(alloc:AllocationBuilder, agent_order:list, items_to_allocate:list, agent_category_capacities:dict[str,dict[str,int]],
                                                    target_category:str):
    """
    This is Round Robin algorithm with respect to categorization (works on each category separately when called)
    it was copied from picking_sequence.py and modified to align with our task

    :param alloc: the current allocation in a form of AllocationBuilder instance
    :param agent_order: a specific order of agents in which to start with
    :param target_category: the category we're welling to run round robin on it currently
    :param agent_category_capacities: a dictionary mapping agents to their capacities per category
    :param items_to_allocate: the desired items to be allocated with Round Robin method

    Examples :
    >>> # Example 1 : basic example
    >>> agent_order=['agent1','agent2','agent3']
    >>> items=['m1','m2','m3']
    >>> valuations={'agent1':{'m1':8,'m2':4,'m3':2},'agent2':{'m1':8,'m2':4,'m3':2},'agent3':{'m1':8,'m2':4,'m3':2}}
    >>> item_categories={'c1':['m1','m2','m3']}
    >>> agent_category_capacities={'agent1': {'c1': 2}, 'agent2': {'c1': 1}, 'agent3': {'c1': 0}}
    >>> target_category='c1'
    >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories[target_category],agent_category_capacities,target_category)
    >>> alloc.sorted()
    {'agent1': ['m1', 'm3'], 'agent2': ['m2'], 'agent3': []}

    >>> # Example 2 :
    >>> agent_order=['agent1','agent2','agent3']
    >>> items=['m1','m2','m3']
    >>> valuations={'agent1':{'m1':8,'m2':4,'m3':10},'agent2':{'m1':8,'m2':4,'m3':2},'agent3':{'m1':8,'m2':4,'m3':2}}
    >>> item_categories={'c1':['m1','m2','m3']}
    >>> agent_category_capacities={'agent1': {'c1': 1}, 'agent2': {'c1': 2}, 'agent3': {'c1': 1}}
    >>> target_category='c1'
    >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories[target_category],agent_category_capacities,target_category)
    >>> alloc.sorted()
    {'agent1': ['m3'], 'agent2': ['m1'], 'agent3': ['m2']}

    >>> # Example 3 :
    >>> agent_order=['agent1','agent2','agent3']
    >>> items=['m1','m2','m3','m4','m5','m6','m7']
    >>> valuations={'agent1':{'m1':1,'m2':2,'m3':3,'m4':4,'m5':5,'m6':6,'m7':7},'agent2':{'m1':7,'m2':6,'m3':5,'m4':4,'m5':3,'m6':2,'m7':1},'agent3':{'m1':1,'m2':2,'m3':2,'m4':1,'m5':1,'m6':1,'m7':1}}
    >>> item_categories={'c1':['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6'],'c4':['m7']}
    >>> agent_category_capacities={'agent1': {'c1': 3,'c2':0,'c3':0,'c4':1}, 'agent2': {'c1': 2,'c2':0,'c3':0,'c4':1}, 'agent3': {'c1': 1,'c2':2,'c3':1,'c4':1}}
    >>> target_category='c1'
    >>> alloc=AllocationBuilder(instance=Instance(valuations=valuations,items=items))
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories[target_category],agent_category_capacities,target_category)
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories['c2'],agent_category_capacities,target_category='c2')
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories['c3'],agent_category_capacities,target_category='c3')
    >>> helper_categorization_friendly_picking_sequence(alloc,agent_order,item_categories['c4'],agent_category_capacities,target_category='c4')
    >>> alloc.sorted()
    {'agent1': ['m3', 'm7'], 'agent2': ['m1'], 'agent3': ['m2', 'm4', 'm5', 'm6']}
    """
    #logger.info(f"**********\nRunning categorization_friendly_picking sequence with\nalloc->{alloc.bundles}, agent order -> {agent_order},items to allocate ->{items_to_allocate} , agent category capacities ->{agent_category_capacities} , target category ->{target_category}\n**********")
    #validate input
    helper_validate_duplicate(agent_order)
    helper_validate_duplicate(items_to_allocate)
    helper_validate_capacities(agent_category_capacities)
    if not isinstance(target_category, str):
        raise ValueError("target_category must be of type str!")
    categories=list(set([category for agent,dict in agent_category_capacities.items() for category in dict.keys()]))
    logger.info(f"target category is ->{target_category} ,agent_category_capacities are -> {agent_category_capacities}")
    if target_category not in categories:
        raise ValueError(f"Target category mistyped or not found: {target_category}")

    #end validation
    if agent_order is None:
        agent_order = [agent for agent in alloc.remaining_agents() if agent_category_capacities[agent][target_category] > 0]

    remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
                                           agent_category_capacities.keys()}
    #logger.info(f"agent_category_capacities-> {agent_category_capacities}")
    remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
    logger.info(f"helper_categorization_friendly_picking_sequence: ")
    logger.info(f'remaining_category_items -> {remaining_category_items} & remaining agent capacities {remaining_category_agent_capacities}')
    logger.info(f"Agent order is -> {agent_order}")
    remaining_agents_with_capacities = {agent for agent,capacity in remaining_category_agent_capacities.items() if capacity>0}# all the agents with non zero capacities in our category
    for agent in cycle(agent_order):# agent order cant be changed , it would still consist of expired agents but we're handling it in our own ways
        logger.info("    Looping agent %s, remaining capacity %s", agent, remaining_category_agent_capacities[agent])
        if agent not in remaining_agents_with_capacities or not remaining_agents_with_capacities: # means no agents left that are feasible to get items
            if not remaining_agents_with_capacities:
                logger.info(f'No agents left due to either:\n 1) reached maximum capacity\n 2) already has copy of item and cant carry more than 1 copy \n breaking out of loop!')
                break
            else: # means only pass to other agent and there is still agents to carry items
               continue


        if remaining_category_agent_capacities[agent] <= 0:
            remaining_agents_with_capacities.discard(agent)
            logger.info(f'{agent} removed from loop since he has no capacity!')
            if len(remaining_agents_with_capacities) == 0:
                logger.info(f'No more agents with capacity')
                break
            continue

        potential_items_for_agent = set(remaining_category_items).difference(alloc.bundles[agent]) # in case difference is empty means already has a duplicate of the item(legal) / there is no items left
        logger.info(f'potential set of items to be allocated to {agent} are -> {potential_items_for_agent}')
        if len(potential_items_for_agent) == 0: # still has capacity, but no items to aquire (maybe no items left maybe already has copy of item)
            logger.info(f'No potential items for agent {agent}')
            logger.info(f'remaining_agents_with_capacities is -> {remaining_agents_with_capacities},agent order is -> {agent_order}')
            if agent in remaining_agents_with_capacities:    # need to remove agent from our loop ,even if he still has capacity !
                logger.info(f'{agent} still has capacity but already has copy of the item')
                #del remaining_category_agent_capacities[agent]
                remaining_agents_with_capacities.discard(agent)
                logger.info(f'{agent} removed from loop')
                if len(remaining_agents_with_capacities) == 0:
                    logger.info(f'No more agents with capacity,breaking loop!')
                    break
                continue # otherwise pick the next agent !
        # safe to assume agent has capacity & has the best item to pick
        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.instance.agent_item_value(agent, item))
        logger.info(f'picked best item for {agent} -> item -> {best_item_for_agent}')
        alloc.give(agent, best_item_for_agent)# this handles capacity of item and capacity of agent !
        remaining_category_agent_capacities[agent] -= 1
        remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
        if len(remaining_category_items) == 0:
            logger.info(f'No more items in category')
            break
        logger.info(f'remaining_category_items -> {remaining_category_items} & remaining agents {remaining_category_agent_capacities}')


def helper_update_envy_graph(curr_bundles: dict, valuation_func: callable, item_categories: dict[str,list],
                             agent_category_capacities: dict[str,dict[str,int]],callback:callable=None,category:str='',iteration:int=0)-> DiGraph:
    """
    simply a helper function to update the envy-graph based on given params
    :param curr_bundles: the current allocation
    :param valuation_func: simply a callable(agent,item)->value
    :param envy_graph: the envy-graph: of course the graph we're going to modify
    :param item_categories: a dictionary mapping items to categorize for the ease of checking:
    :param agent_category_capacities: a dictionary where keys are agents and values are dictionaries mapping categories to capacities.

    >>> #Example 1 :
    >>> valuation_func= lambda  agent,item: {'Agent1':{'m1':100,'m2':0},'Agent2':{'m1':100,'m2':0}}[agent][item]
    >>> bundle={'Agent1':['m1'],'Agent2':['m2']}
    >>> item_categories={'c1':['m1','m2']}
    >>> agent_category_capacities= {'Agent1':{'c1':1},'Agent2':{'c1':1}}
    >>> graph = helper_update_envy_graph(valuation_func=valuation_func, item_categories=item_categories, agent_category_capacities=agent_category_capacities, curr_bundles=bundle )
    >>> graph.has_edge('Agent2','Agent1')
    True
    >>> graph.has_edge('Agent1','Agent2')
    False
    >>> #Example 2 : cycle between 2 agents !
    >>> valuation_func= lambda  agent,item: {'Agent1':{'m1':99,'m2':100},'Agent2':{'m1':100,'m2':99}}[agent][item]
    >>> bundle={'Agent1':['m1'],'Agent2':['m2']}
    >>> item_categories={'c1':['m1','m2']}
    >>> agent_category_capacities= {'Agent1':{'c1':1},'Agent2':{'c1':1}}
    >>> graph = helper_update_envy_graph(valuation_func=valuation_func, item_categories=item_categories, agent_category_capacities=agent_category_capacities, curr_bundles=bundle )
    >>> graph.has_edge('Agent2','Agent1')
    True
    >>> graph.has_edge('Agent1','Agent2')
    True

    >>> #Example 3 : cycle between 2 agents and cycle-removal
    >>> graph=DiGraph()
    >>> valuations={'Agent1':{'m1':99,'m2':100},'Agent2':{'m1':100,'m2':99}}
    >>> items=['m1','m2']
    >>> valuation_func= lambda  agent,item: valuations[agent][item]
    >>> bundle={'Agent1':['m1'],'Agent2':['m2']}
    >>> item_categories={'c1':['m1','m2']}
    >>> agent_category_capacities= {'Agent1':{'c1':1},'Agent2':{'c1':1}}
    >>> graph = helper_update_envy_graph(valuation_func=valuation_func, item_categories=item_categories, agent_category_capacities=agent_category_capacities, curr_bundles=bundle )
    >>> graph.has_edge('Agent2','Agent1')
    True
    >>> graph.has_edge('Agent1','Agent2')
    True
    >>> graph = helper_remove_cycles(graph, alloc=AllocationBuilder(instance=Instance(items=items,valuations=valuations)),valuation_func=valuation_func,agent_category_capacities=agent_category_capacities,item_categories=item_categories)
    >>> graph.has_edge('Agent2','Agent1')
    False
    >>> graph.has_edge('Agent1','Agent2')
    False
    """
    #logger.info(f'**********\nRunning helper_update_envy_graph\n**********')
    logger.info(f"  Creating envy graph for curr_bundles -> {curr_bundles}")
    envy_graph = DiGraph()
    envy_graph.add_nodes_from(curr_bundles.keys())
    # validate input
    if isinstance(curr_bundles, dict):
        for key, val in curr_bundles.items():
            if not isinstance(key, str) or not isinstance(val, (list, set)):
                raise ValueError(f"Bundles not structured properly.{curr_bundles} type(va;) -> {type(val)}")
        if not callable(valuation_func):
            raise ValueError("valuation_func must be callable.")
        if not isinstance(envy_graph, nx.DiGraph):
            raise ValueError("envy_graph must be of type nx.DiGraph.")
        helper_validate_item_categories(item_categories)
        helper_validate_capacities(agent_category_capacities)
    #end validation
    for agent1, bundle1 in curr_bundles.items():
        for agent2, bundle_agent2 in curr_bundles.items():
            if agent1 is not agent2:  # make sure w're not comparing same agent to himself
                # make sure to value with respect to the constraints of feasibility
                # since in algo 1 its always feasible because everyone has equal capacity we dont pay much attention to it
                if helper_envy(source=agent1, target=agent2, bundles=curr_bundles, val_func=valuation_func,
                               item_categories=item_categories, agent_category_capacities=agent_category_capacities):
                    #print(f"{agent1} envies {agent2}")  # works great .
                    # we need to add edge from the envier to the envyee
                    envy_graph.add_edge(agent1, agent2)
    if callback:# no need to capture image in each iteration ...
        callback(helper_generate_directed_graph_base64(envy_graph, category=category, iteration=iteration))
    logger.info(f"  Done updating/building envy-graph. envy_graph.edges after update -> {envy_graph.edges}")
    return envy_graph

# def visualize_graph(envy_graph):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 6))
#     nx.draw(envy_graph, with_labels=True)
#     plt.title('Basic Envy Graph')
#     plt.show()
#

def helper_remove_cycles(envy_graph:nx.DiGraph, alloc:AllocationBuilder, valuation_func:callable, item_categories:dict[str,list], agent_category_capacities:dict[str,dict[str,int]],callback:callable=None,iteration:int=-1,category:str='')->DiGraph:
    """
        Removes cycles from the envy graph by updating the bundles.

        :param envy_graph: The envy graph (a directed graph).
        :param alloc: An AllocationBuilder instance for managing allocations.
        :param valuation_func: Function to determine the value of an item for an agent.
        :param item_categories: A dictionary of categories, with each category paired with a list of items.
        :param agent_category_capacities: A dictionary of dictionaries mapping agents to their capacities for each category.

        >>> import networkx as nx
        >>> from fairpyx import Instance, AllocationBuilder
        >>> valuations = {'Agent1': {'Item1': 3, 'Item2': 5,'Item3': 1}, 'Agent2': {'Item1': 2, 'Item2': 6 ,'Item3':10}, 'Agent3': {'Item1': 4, 'Item2': 1,'Item3':2.8}}
        >>> items = ['Item1', 'Item2','Item3']
        >>> instance = Instance(valuations=valuations, items=items)
        >>> alloc = AllocationBuilder(instance)
        >>> alloc.give('Agent1', 'Item1')
        >>> alloc.give('Agent2', 'Item2')
        >>> alloc.give('Agent3', 'Item3')
        >>> item_categories = {'c1': ['Item1', 'Item2','Item3']}
        >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}, 'Agent3': {'c1': 1}}
        >>> envy_graph = nx.DiGraph()
        >>> valuation_callable=alloc.instance.agent_item_value
        >>> def valuation_func(agent, item): return valuation_callable(agent,item)
        >>> envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities)
        >>> list(envy_graph.edges)
        [('Agent1', 'Agent2'), ('Agent2', 'Agent3'), ('Agent3', 'Agent1')]
        >>> not nx.is_directed_acyclic_graph(envy_graph)
        True
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
        >>> list(nx.simple_cycles(envy_graph))
        []
        >>> list(envy_graph.edges)
        []
        >>> alloc.sorted()
        {'Agent1': ['Item2'], 'Agent2': ['Item3'], 'Agent3': ['Item1']}

        Example 2: 3 Agents double cycle
         >>> valuations = {'Agent1': {'Item1': 5, 'Item2': 6,'Item3': 5}, 'Agent2': {'Item1': 6, 'Item2': 5 ,'Item3':6},'Agent3': {'Item1': 5, 'Item2': 6 ,'Item3':5}}
        >>> items = ['Item1', 'Item2','Item3']
        >>> instance = Instance(valuations=valuations, items=items)
        >>> alloc = AllocationBuilder(instance)
        >>> alloc.give('Agent1', 'Item1')
        >>> alloc.give('Agent2', 'Item2')
        >>> alloc.give('Agent3', 'Item3')
        >>> item_categories = {'c1': ['Item1', 'Item2','Item3']}
        >>> agent_category_capacities = {'Agent1': {'c1': 3}, 'Agent2': {'c1': 3},'Agent3': {'c1': 3}}
        >>> envy_graph = nx.DiGraph()
        >>> valuation_callable=alloc.instance.agent_item_value
        >>> def valuation_func(agent, item): return valuation_callable(agent,item)
        >>> envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities)
        >>> list(envy_graph.edges)
        [('Agent1', 'Agent2'), ('Agent2', 'Agent1'), ('Agent2', 'Agent3'), ('Agent3', 'Agent2')]
        >>> not nx.is_directed_acyclic_graph(envy_graph)
        True
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
        >>> list(nx.simple_cycles(envy_graph))
        []
        >>> list(envy_graph.edges)
        [('Agent3', 'Agent1')]
        >>> alloc.sorted()
        {'Agent1': ['Item2'], 'Agent2': ['Item1'], 'Agent3': ['Item3']}

         Example 3: 2 Agents bundle switching
        >>> valuations = {'Agent1': {'Item1': 1, 'Item2': 1,'Item3': 10}, 'Agent2': {'Item1': 10, 'Item2': 10 ,'Item3':1}}
        >>> items = ['Item1', 'Item2','Item3']
        >>> instance = Instance(valuations=valuations, items=items)
        >>> alloc = AllocationBuilder(instance)
        >>> alloc.give('Agent1', 'Item1')
        >>> alloc.give('Agent1', 'Item2')
        >>> alloc.give('Agent2', 'Item3')
        >>> item_categories = {'c1': ['Item1', 'Item2','Item3']}
        >>> agent_category_capacities = {'Agent1': {'c1': 3}, 'Agent2': {'c1': 3}}
        >>> envy_graph = nx.DiGraph()
        >>> valuation_callable=alloc.instance.agent_item_value
        >>> def valuation_func(agent, item): return valuation_callable(agent,item)
        >>> envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities)
        >>> list(envy_graph.edges)
        [('Agent1', 'Agent2'), ('Agent2', 'Agent1')]
        >>> not nx.is_directed_acyclic_graph(envy_graph)
        True
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
        >>> list(nx.simple_cycles(envy_graph))
        []
        >>> list(envy_graph.edges)
        []
        >>> alloc.sorted()
        {'Agent1': ['Item3'], 'Agent2': ['Item1', 'Item2']}



        Example 4: 2 cycle of size 3
        >>> valuations = {'Agent1': {'Item1': 1, 'Item2': 2,'Item3': 1}, 'Agent2': {'Item1': 1, 'Item2': 1 ,'Item3':3},'Agent3': {'Item1': 10, 'Item2': 1 ,'Item3':5}}
        >>> items = ['Item1', 'Item2','Item3']
        >>> instance = Instance(valuations=valuations, items=items)
        >>> alloc = AllocationBuilder(instance)
        >>> alloc.give('Agent1', 'Item1')
        >>> alloc.give('Agent2', 'Item2')
        >>> alloc.give('Agent3', 'Item3')
        >>> item_categories = {'c1': ['Item1'],'c2':['Item2'],'c3':['Item3']}
        >>> agent_category_capacities = {'Agent1': {'c1': 1,'c2': 1,'c3': 1}, 'Agent2': {'c1': 1,'c2': 1,'c3': 1},'Agent3': {'c1': 1,'c2': 1,'c3': 1}}
        >>> envy_graph = nx.DiGraph()
        >>> valuation_callable=alloc.instance.agent_item_value
        >>> def valuation_func(agent, item): return valuation_callable(agent,item)
        >>> envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities)
        >>> list(envy_graph.edges)
        [('Agent1', 'Agent2'), ('Agent2', 'Agent3'), ('Agent3', 'Agent1')]
        >>> not nx.is_directed_acyclic_graph(envy_graph)
        True
        >>> envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
        >>> list(nx.simple_cycles(envy_graph))
        []
        >>> list(envy_graph.edges)
        []
        >>> alloc.sorted()
        {'Agent1': ['Item2'], 'Agent2': ['Item3'], 'Agent3': ['Item1']}
        """
    #TODO add example with cycle of length 3

    #start validation
    if not callable(valuation_func):
        raise ValueError("valuation_func must be callable.")
    if not isinstance(envy_graph, nx.DiGraph):
        raise ValueError("envy_graph must be of type nx.DiGraph.")
    helper_validate_item_categories(item_categories)
    helper_validate_capacities(agent_category_capacities)
    #end validation
    while not nx.is_directed_acyclic_graph(envy_graph):
        try:
            cycles = list(nx.simple_cycles(envy_graph))
            num_cycles = len(cycles)
            logger.info(f"Number of cycles detected: {num_cycles}")

            cycle = nx.find_cycle(envy_graph, orientation='original')
            agents_in_cycle = [edge[0] for edge in cycle]
            logger.info(f"Detected cycle: {agents_in_cycle}")

            # Copy the bundle of the first agent in the cycle
            temp_val = alloc.bundles[agents_in_cycle[0]].copy()
            logger.info(f"Initial temp_val (copy of first agent's bundle): {temp_val}")

            # Perform the swapping
            logger.info(
                f"    Before swapping: { {i: alloc.bundles[i] for i in agents_in_cycle} }"
            )
            for i in range(len(agents_in_cycle) - 1):
                current_agent = agents_in_cycle[i]
                next_agent = agents_in_cycle[(i + 1) % len(agents_in_cycle)]
                logger.info(
                    f"Before swapping: {current_agent} -> {alloc.bundles[current_agent]}, {next_agent} -> {alloc.bundles[next_agent]}")

                # Swap the bundles
                alloc.bundles[current_agent] = alloc.bundles[next_agent]
                alloc.bundles[next_agent] = temp_val
                logger.info(f"After swapping: { {i: alloc.bundles[i] for i in agents_in_cycle} }")

            # Update the envy graph after swapping
            envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories,
                                                  agent_category_capacities)
            logger.info(f"  Updated envy graph. is Graph acyclic ?? {nx.is_directed_acyclic_graph(envy_graph)}")

        except nx.NetworkXNoCycle:
            logger.info("No more cycles detected")
            break
        except Exception as e:
            logger.error(f"Error during cycle removal: {e}")
            break

    if not nx.is_directed_acyclic_graph(envy_graph):
        logger.warning("Cycle removal failed to achieve an acyclic graph.")
        raise RuntimeError("Cycle removal failed to achieve an acyclic graph.")

    logger.info("Cycle removal process ended successfully")
    return envy_graph

def helper_update_ordered_agent_list(current_order: list, remaining_category_agent_capacities: dict) -> list:
    """
       Update the ordered list of agents based on remaining category agent capacities.

       Args:
           current_order (list): Current order of agents.
           remaining_category_agent_capacities (dict): A dictionary with category capacities for only remaining agents
           (Assume agents with 0 capacity aren't included in it).

       Returns:
           list: The updated list of agents ordered by remaining capacities.

    >>> #Example 1: trivial example
    >>> current_order = ['Abed', 'Yousef']
    >>> remaining_category_agent_capacities = {'Abed': 1, 'Yousef': 2}
    >>> helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)
    ['Abed', 'Yousef']
    >>> #Example 2: example with agent with no remaining capacity
    >>> current_order = ['Abed', 'Yousef','Noor']
    >>> remaining_category_agent_capacities = {'Abed': 1, 'Yousef': 2}
    >>> helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)
    ['Abed', 'Yousef']
    >>> #Example 3: example to test that function maintains order
    >>> current_order = ['Abed', 'Yousef','Noor']
    >>> remaining_category_agent_capacities = {'Abed': 1, 'Noor': 2}
    >>> helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)
    ['Abed', 'Noor']
    """
    #logger.info(f'**********\nRunning helper_update_ordered_agent_list\n**********')
    #validate input
    helper_validate_duplicate(current_order)
    temp = {'catx': remaining_category_agent_capacities}
    helper_validate_capacities(temp)
    #end validation
    current_agent_list = [agent for agent in current_order if agent in remaining_category_agent_capacities.keys()]
    logger.info(f'current_agent_list->{current_agent_list}')
    return current_agent_list


def helper_update_item_list(alloc: AllocationBuilder, category: str, item_categories: dict[str,list]) -> list:
    """
        Update the item list for a given category and allocation.

        Args:
            alloc: Current allocationBuilder.
            category: The category to update.
            item_categories : A dictionary mapping categories to items .

        Returns:
            list: The updated allocation list with items for the specified category.

            >>> # Example 1: no items remain
            >>> instance= Instance(valuations={'Agent1': {'m1': 1, 'm2': 0}, 'Agent2': {'m1': 0, 'm2': 1}}, items=['m1', 'm2'])
            >>> alloc= AllocationBuilder(instance=instance)
            >>> alloc.give('Agent1','m1')
            >>> category = 'c1'
            >>> item_categories = {'c1':['m1'], 'c2':['m2']}
            >>> helper_update_item_list(alloc, category, item_categories)
            []
            >>> #Example 2: remaining item
            >>> instance = Instance(valuations={'Agent1': {'m1': 1, 'm2': 0,'m3':0}, 'Agent2': {'m1': 0, 'm2': 1,'m3':0}}, items=['m1', 'm2','m3'])
            >>> alloc= AllocationBuilder(instance=instance)
            >>> alloc.give('Agent1','m1')
            >>> category = 'c1'
            >>> item_categories = {'c1':['m1','m3'], 'c2':['m2']}
            >>> helper_update_item_list(alloc, category, item_categories)
            ['m3']
            >>> alloc.give('Agent1','m3')
            >>> helper_update_item_list(alloc, category, item_categories)
            []


            >>> #Example 3: remaining items
            >>> instance = Instance(valuations={'Agent1': {'m1': 1, 'm2': 0,'m3':0,'m4':0}, 'Agent2': {'m1': 0, 'm2': 1,'m3':0,'m4':1}}, items=['m1', 'm2','m3','m4'])
            >>> alloc= AllocationBuilder(instance=instance)
            >>> alloc.give('Agent1','m1')
            >>> category = 'c1'
            >>> item_categories = {'c1':['m1','m3','m4'], 'c2':['m2']}
            >>> helper_update_item_list(alloc, category, item_categories)
            ['m3', 'm4']
            >>> helper_update_item_list(alloc, 'c2', item_categories)
            ['m2']
            >>> alloc.give('Agent2','m2')
            >>> helper_update_item_list(alloc, 'c2', item_categories)
            []
        """
    #logger.info(f'**********\nRunning helper_update_item_list\n**********')
    #validate input
    helper_validate_item_categories(item_categories)
    if not isinstance(category, str):
        raise ValueError("category must be of type str!")
    if category not in item_categories:
        raise ValueError(f"Category mistyped or not found: {category}")
    #end validation
    current_item_list = [item for item in alloc.remaining_items() if item in item_categories[category]]
    logger.info(f'current_item_list->{current_item_list}')
    return current_item_list


def helper_priority_matching(agent_item_bipartite_graph:nx.Graph, current_order:list, alloc:AllocationBuilder, remaining_category_agent_capacities:dict[str,int]):
    """
    Performs priority matching based on the agent-item bipartite graph and the current order of agents.

    :param agent_item_bipartite_graph: A bipartite graph with agents and items as nodes, and edges with weights representing preferences.
    :param current_order: The current order of agents for matching.
    :param alloc: An AllocationBuilder instance to manage the allocation process.
    :param remaining_category_agent_capacities: A dictionary mapping agents to their remaining capacities for the category.
    >>> # Example 1 : simple introductory example
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
    >>> helper_priority_matching(agent_item_bipartite_graph, current_order, alloc, remaining_category_agent_capacities)
    >>> alloc.sorted()
    {'Agent1': ['Item1'], 'Agent2': ['Item2']}

    >>> # Example 2 : 2 agents common interest (the one who's first wins !)
    >>> import networkx as nx
    >>> from fairpyx import Instance, AllocationBuilder
    >>> current_order = ['Agent1', 'Agent2']
    >>> items=['Item1', 'Item2']
    >>> valuations={'Agent1': {'Item1': 1, 'Item2': 0}, 'Agent2': {'Item1': 1, 'Item2': 0}}
    >>> instance = Instance(valuations=valuations, items=items)
    >>> alloc = AllocationBuilder(instance)
    >>> remaining_category_agent_capacities = {'Agent1': 1, 'Agent2': 1}
    >>> agent_item_graph=helper_create_agent_item_bipartite_graph(agents=current_order,items=items,valuation_func=lambda agent,item:valuations[agent][item])
    >>> helper_priority_matching(agent_item_graph, current_order, alloc, remaining_category_agent_capacities)
    >>> alloc.sorted()
    {'Agent1': ['Item1'], 'Agent2': []}

    >>> # Example 3 : 3 agents common interest in all items
    >>> import networkx as nx
    >>> from fairpyx import Instance, AllocationBuilder
    >>> current_order = ['Agent1', 'Agent2','Agent3']
    >>> items=['Item1', 'Item2','Item3']
    >>> valuations={'Agent1': {'Item1': 1, 'Item2': 1,'Item3':1}, 'Agent2': {'Item1': 1, 'Item2': 1,'Item3':1},'Agent3': {'Item1': 1, 'Item2': 1,'Item3':1}}
    >>> instance = Instance(valuations=valuations, items=items)
    >>> alloc = AllocationBuilder(instance)
    >>> remaining_category_agent_capacities = {'Agent1': 4, 'Agent2': 4,'Agent3': 4}
    >>> agent_item_graph=helper_create_agent_item_bipartite_graph(agents=current_order,items=items,valuation_func=lambda agent,item:valuations[agent][item])
    >>> helper_priority_matching(agent_item_graph, current_order, alloc, remaining_category_agent_capacities)
    >>> alloc.sorted() in [{'Agent1': ['Item3'], 'Agent2': ['Item2'], 'Agent3': ['Item1']} , {'Agent1': ['Item1'], 'Agent2': ['Item2'], 'Agent3': ['Item3']} , {'Agent1': ['Item1'], 'Agent2': ['Item3'], 'Agent3': ['Item2']}]
    True
    """
    #logger.info(f'**********\nRunning helper_priority_matching\n**********')
    #validate input
    if not isinstance(agent_item_bipartite_graph ,nx.Graph):
        raise ValueError("agent_item_bipartite_graph must be of type nx.Graph.")
    helper_validate_duplicate(current_order)
    helper_validate_capacities({'catx': remaining_category_agent_capacities})
    #end validation
    matching=nx.max_weight_matching(agent_item_bipartite_graph)
    # we used max weight matching in which (agents are getting high weights in desc order 2^n,2^n-1.......1)
    logger.info(f'matching is -> {matching}')
    for match in matching:
        if match[0] in current_order: # previously was .startsWith('A') but not necessarily
            if ((match[0], match[1]) not in alloc.remaining_conflicts) and match[0] in remaining_category_agent_capacities.keys():
                alloc.give(match[0], match[1], logger)
                remaining_category_agent_capacities[match[0]] -= 1
                if remaining_category_agent_capacities[match[0]] <= 0:
                    del remaining_category_agent_capacities[match[0]]
            #else do nothing ....
        else:#meaning match[1] in current_order
            if  ((match[1], match[0]) not in alloc.remaining_conflicts) and match[1] in remaining_category_agent_capacities.keys():
                alloc.give(match[1], match[0], logger)
                remaining_category_agent_capacities[match[1]] -= 1
                if remaining_category_agent_capacities[match[1]] <= 0:
                    del remaining_category_agent_capacities[match[1]]


def helper_create_agent_item_bipartite_graph(agents:list, items:list, valuation_func:callable):
    """
    Creates an agent-item bipartite graph with edges weighted based on agent preferences.

    :param agents: List of agents.
    :param items: List of items.
    :param valuation_func: Function to determine the value of an item for an agent.
    :return: A bipartite graph with agents and items as nodes, and edges with weights representing preferences.

    >>> import networkx as nx
    >>> #Example 1: simple graph
    >>> agents = ['Agent1', 'Agent2']
    >>> items = ['Item1', 'Item2']
    >>> valuation_func=lambda agent,item:{'Agent1': {'Item1': 3, 'Item2': 1}, 'Agent2': {'Item1': 2, 'Item2': 4}}[agent][item]
    >>> bipartite_graph = helper_create_agent_item_bipartite_graph(agents, items,valuation_func)
    >>> sorted(bipartite_graph.edges(data=True))
    [('Agent1', 'Item1', {'weight': 4}), ('Agent1', 'Item2', {'weight': 4}), ('Agent2', 'Item1', {'weight': 2}), ('Agent2', 'Item2', {'weight': 2})]

    >>> #Example 2: 3 agent graph (with unsual order,notice the weights with respect to initial_order)
    >>> agents = ['Agent1', 'Agent3','Agent2']
    >>> items = ['Item1', 'Item2','Item3']
    >>> valuation_func=lambda agent,item:{'Agent1': {'Item1': 3, 'Item2': 1,'Item3': 0}, 'Agent2': {'Item1': 2, 'Item2': 0,'Item3': 0},'Agent3': {'Item1': 0, 'Item2': 0,'Item3':1}}[agent][item]
    >>> bipartite_graph = helper_create_agent_item_bipartite_graph(agents, items,valuation_func)
    >>> sorted(bipartite_graph.edges(data=True))
    [('Agent1', 'Item1', {'weight': 8}), ('Agent1', 'Item2', {'weight': 8}), ('Agent2', 'Item1', {'weight': 2}), ('Agent3', 'Item3', {'weight': 4})]

    >>> #Example 3: 3 agent graph
    >>> agents = ['Agent1', 'Agent2','Agent3']
    >>> items = ['Item1', 'Item2','Item3']
    >>> valuation_func=lambda agent,item:{'Agent1': {'Item1': 0, 'Item2': 0,'Item3': 0}, 'Agent2': {'Item1': 0, 'Item2': 0,'Item3': 0},'Agent3': {'Item1': 1, 'Item2': 1,'Item3':1}}[agent][item]
    >>> bipartite_graph = helper_create_agent_item_bipartite_graph(agents, items,valuation_func)
    >>> sorted(bipartite_graph.edges(data=True))
    [('Agent3', 'Item1', {'weight': 2}), ('Agent3', 'Item2', {'weight': 2}), ('Agent3', 'Item3', {'weight': 2})]
    """
    #logger.info(f'**********\nRunning helper_create_agent_item_bipartite_graph\n**********')
    #validate input
    helper_validate_duplicate(agents)
    helper_validate_duplicate(items)
    if not callable(valuation_func):
        raise ValueError("valuation_func must be callable.")
    #end
    agent_item_bipartite_graph = nx.Graph()
    agent_item_bipartite_graph.add_nodes_from(agents, bipartite=0)
    agent_item_bipartite_graph.add_nodes_from(items, bipartite=1)
    n=len(agents)
    logger.info(f'ordered agent list ->{agents}')
    weight = 2**n
    for agent in agents:
        logger.info(f'{agent} weight is ->{weight}')
        for item in items:
            if valuation_func(agent, item) != 0:
                agent_item_bipartite_graph.add_edge(agent, item, weight=weight)
        n -= 1
        weight = 2**n
    logger.info(f'bipartite graph ->{agent_item_bipartite_graph}')
    return agent_item_bipartite_graph

def helper_validate_valuations(agent_item_valuations: dict[str, dict[str, int]], is_identical: bool = False, is_binary: bool = False):
    #logger.info(f'**********\nRunning helper_validate_valuations\n**********')

    if  isinstance(agent_item_valuations,dict):# to check that the agent_category_capacities is indeed dict[str,dict[str,int]]
        for key,value in agent_item_valuations.items():
            if not isinstance(key,str) or not isinstance(value,dict):
                raise ValueError(f"agent_item_valuations {agent_item_valuations} isn't structured correctly")
            for inner_key,inner_value in value.items():
                if not isinstance(inner_key,str) or not isinstance(inner_value,(int,np.int32,np.int64,float)):
                    raise ValueError(f"agent_item_valuations {agent_item_valuations} isn't structured correctly, inner value type is {type(inner_value)}")

        if is_identical:
            # Check for identical valuations
            first_agent_values = next(iter(agent_item_valuations.values()))
            for agent, items in agent_item_valuations.items():
                if items != first_agent_values:
                    raise ValueError(f"Valuations for agent {agent} are not identical.")

        # Check if there are negative valuations
        negative_values = [value for agent in agent_item_valuations for value in agent_item_valuations[agent].values() if value < 0]
        if negative_values:
            raise ValueError(f"Negative valuations found: {negative_values}")
        if is_binary:
            if any(value not in [0, 1] for agent in agent_item_valuations for value in
                   agent_item_valuations[agent].values()):
                raise ValueError("Non-binary values found in agent item valuations.")
    else:
        raise ValueError(f"agent_item_valuations {agent_item_valuations} isn't structured correctly")



def helper_validate_capacities(agent_category_capacities: dict[str, dict[str, int]], is_identical: bool = False):
    #logger.info(f'**********\nRunning helper_validate_capacities\n**********')

    if  isinstance(agent_category_capacities,dict):# to check that the agent_category_capacities is indeed dict[str,dict[str,int]]
        for key,value in agent_category_capacities.items():
            if not isinstance(key,str) or not isinstance(value,dict):
                raise ValueError(f"agent_category_capacities {agent_category_capacities} isn't structured correctly")
            for inner_key,inner_value in value.items():
                if not isinstance(inner_key,str) or not isinstance(inner_value,int):
                    raise ValueError(f"agent_category_capacities {agent_category_capacities} isn't structured correctly")

        if is_identical:
            # Check for identical capacities
            first_agent_capacities = next(iter(agent_category_capacities.values()))
            for agent, capacities in agent_category_capacities.items():
                if capacities != first_agent_capacities:
                    raise ValueError(f"Capacities for {agent}={capacities} are not identical with {list(agent_category_capacities.keys())[0]}={first_agent_capacities}.")

        # Check if there are negative capacities
        negative_capacities = [value for agent in agent_category_capacities for value in agent_category_capacities[agent].values() if value < 0]
        if negative_capacities:
            raise ValueError(f"Negative capacities found: {negative_capacities}")
    else:
        raise ValueError(f"agent_category_capacities {agent_category_capacities} isn't structured correctly")

def helper_validate_duplicate(list_of_items:list):
    #logger.info(f'**********\nRunning helper_validate_duplicates\n**********')
    if  isinstance(list_of_items,list):
        if len(list_of_items) != len(set(list_of_items)):
            raise ValueError(f"Duplicate items found in the list: {list_of_items}.")
    else:
        raise ValueError(f"the input {list_of_items} isn't of type list, only list is allowed.")

def helper_validate_item_categories(item_categories:dict[str, list]):
    #logger.info(f'**********\nRunning helper_validate_item_categories\n**********')
    if isinstance(item_categories, dict):
        for category, items in item_categories.items():
            if not isinstance(category, str) or not isinstance(items, list):
                raise ValueError(f"item categories not structured properly!!!")
    else:
        raise ValueError(f"item categories is supposed to be dict[str,list] but u entered {type(item_categories)}")


# def helper_generate_directed_graph_base64(graph, seed=42,category:str='',iteration:int=None,text:str=None):
#     #logger.info(f'**********\nRunning helper_generate_directed_graph_base64\n**********')
#
#     plt.figure()
#     plt.title('Envy Graph',fontsize=16)
#     additional_text=f'category -> {category} iteration -> {iteration}' if text is None else text
#     plt.figtext(0.5, 0.85, additional_text, wrap=True, horizontalalignment='center', fontsize=10)
#     pos = nx.spring_layout(graph, seed=seed)  # Use a seed for reproducibility
#     nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, arrows=True)
#
#     img_bytes = io.BytesIO()
#     plt.savefig(img_bytes, format='png')
#     plt.close()
#     img_bytes.seek(0)
#
#     base64_image = base64.b64encode(img_bytes.read()).decode('utf-8')
#     print("Generated image data:", base64_image[:100])  # Print the first 100 characters of the image data
#     return base64_image
#
#
# def helper_generate_bipartite_graph_base64(graph,iteration:int,category:str,text:str=None):
#     #logger.info(f'**********\nRunning helper_generate_bipartite_graph_base64\n**********')
#     plt.figure()
#     plt.title('Agent-Item Bipartite Graph', fontsize=16)
#     additional_text=f'category -> {category} iteration -> {iteration}' if text is None else text
#     plt.figtext(0.5, 0.85, additional_text, wrap=True, horizontalalignment='center', fontsize=10)
#     try:
#         top_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
#         bottom_nodes = set(graph) - top_nodes
#
#         # Create fixed positions
#         pos = {}
#         pos.update((node, (1, index)) for index, node in enumerate(top_nodes))  # x=1 for top_nodes
#         pos.update((node, (2, index)) for index, node in enumerate(bottom_nodes))  # x=2 for bottom_nodes
#
#         # Assign colors to the nodes based on their group
#         color_map = ['red' if node in top_nodes else 'blue' for node in graph.nodes]
#     except nx.NetworkXError:
#         # Fallback to spring layout if there's an error
#         pos = nx.spring_layout(graph)
#         # Assign default colors if layout falls back
#         color_map = ['red' for node in graph.nodes]
#
#     nx.draw(graph, pos, with_labels=True, node_color=color_map, edge_color='gray', node_size=500, font_size=10)
#
#     # Adjust the layout to make sure the title is visible
#     plt.tight_layout(rect=[0, 0, 1, 0.90])
#
#     img_bytes = io.BytesIO()
#     plt.savefig(img_bytes, format='png')
#     plt.close()
#     img_bytes.seek(0)
#
#     return base64.b64encode(img_bytes.read()).decode('utf-8')

def helper_generate_directed_graph_base64(graph, seed=42, category: str = '', iteration: int = None, text: str = None):
    plt.figure(figsize=(14, 10), dpi=100)  # Figure size and DPI for quality
    plt.title('Envy Graph', fontsize=24, pad=60)  # Increased padding to separate title from additional text

    # Additional text with better placement
    additional_text = f'Category: {category} | Iteration: {iteration}' if text is None else text
    plt.figtext(0.5, 0.88, additional_text, wrap=True, horizontalalignment='center', fontsize=16)  # Lowered text position

    pos = nx.spring_layout(graph, seed=seed)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16, arrows=True)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')  # Save with tight bounding box for better layout
    plt.close()
    img_bytes.seek(0)

    base64_image = base64.b64encode(img_bytes.read()).decode('utf-8')
    return base64_image


def helper_generate_bipartite_graph_base64(graph, iteration: int, category: str, text: str = None):
    plt.figure(figsize=(14, 10), dpi=100)  # Figure size and DPI for quality
    plt.title('Agent-Item Bipartite Graph', fontsize=24, pad=60)  # Increased padding to separate title from additional text

    # Additional text with better placement
    additional_text = f'Category: {category} | Iteration: {iteration}' if text is None else text
    plt.figtext(0.5, 0.88, additional_text, wrap=True, horizontalalignment='center', fontsize=16)  # Lowered text position

    try:
        top_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
        bottom_nodes = set(graph) - top_nodes

        pos = {}
        pos.update((node, (1, index)) for index, node in enumerate(top_nodes))  # x=1 for top_nodes
        pos.update((node, (2, index)) for index, node in enumerate(bottom_nodes))  # x=2 for bottom_nodes

        color_map = ['red' if node in top_nodes else 'blue' for node in graph.nodes]
    except nx.NetworkXError:
        pos = nx.spring_layout(graph)
        color_map = ['red' for node in graph.nodes]

    nx.draw(graph, pos, with_labels=True, node_color=color_map, edge_color='gray', node_size=2000, font_size=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')  # Save with tight bounding box for better layout
    plt.close()
    img_bytes.seek(0)

    return base64.b64encode(img_bytes.read()).decode('utf-8')
def helper_configure_logger():
    #logger.info(f'**********\nRunning helper_configure_logger\n**********')
    # Create a string stream to capture logs
    log_stream = io.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(message)s')
    logger.addHandler(stream_handler)
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    return log_stream

def helper_get_logs(log_stream):
    #logger.info(f'**********\nRunning helper_get_logs\n**********')
    return log_stream.getvalue()

if __name__ == "__main__":
    import doctest, sys
    import time
    
    print("\n", doctest.testmod(), "\n")
    sys.exit(1)
    
    #
    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    # logger.addHandler(logging.FileHandler('fairpy.log'))
    #
    # order = ['Agent14', 'Agent2', 'Agent16', 'Agent3', 'Agent6', 'Agent12', 'Agent8', 'Agent15', 'Agent19', 'Agent4',
    #          'Agent13', 'Agent9', 'Agent5', 'Agent11', 'Agent17', 'Agent7', 'Agent1', 'Agent10', 'Agent18']
    # items = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
    # item_categories = {'c1': ['m5'], 'c2': ['m1'], 'c3': ['m4'], 'c4': ['m10'], 'c5': ['m8'], 'c6': ['m9'],
    #                    'c7': ['m6'], 'c8': ['m2'], 'c9': ['m3'], 'c10': ['m7']}
    #
    # equal_capacity = {'c1': 10, 'c2': 11, 'c3': 11, 'c4': 11, 'c5': 10, 'c6': 11, 'c7': 10, 'c8': 10, 'c9': 11,
    #                   'c10': 10}
    # agent_category_capacities = {
    #     'Agent1': equal_capacity,
    #     'Agent2': equal_capacity,
    #     'Agent3': equal_capacity,
    #     'Agent4': equal_capacity,
    #     'Agent5': equal_capacity,
    #     'Agent6': equal_capacity,
    #     'Agent7': equal_capacity,
    #     'Agent8': equal_capacity,
    #     'Agent9': equal_capacity,
    #     'Agent10': equal_capacity,
    #     'Agent11': equal_capacity,
    #     'Agent12': equal_capacity,
    #     'Agent13': equal_capacity,
    #     'Agent14': equal_capacity,
    #     'Agent15': equal_capacity,
    #     'Agent16': equal_capacity,
    #     'Agent17': equal_capacity,
    #     'Agent18': equal_capacity,
    #     'Agent19': equal_capacity,
    # }
    #
    # valuations = {'Agent1': {'m1': 1, 'm2': 0, 'm3': 1, 'm4': 1, 'm5': 0, 'm6': 1, 'm7': 1, 'm8': 1, 'm9': 0, 'm10': 0},
    #               'Agent2': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 1, 'm5': 0, 'm6': 1, 'm7': 0, 'm8': 1, 'm9': 0, 'm10': 0},
    #               'Agent3': {'m1': 0, 'm2': 0, 'm3': 1, 'm4': 1, 'm5': 1, 'm6': 1, 'm7': 1, 'm8': 0, 'm9': 1, 'm10': 0},
    #               'Agent4': {'m1': 1, 'm2': 0, 'm3': 1, 'm4': 0, 'm5': 1, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 0, 'm10': 1},
    #               'Agent5': {'m1': 1, 'm2': 1, 'm3': 0, 'm4': 0, 'm5': 1, 'm6': 0, 'm7': 0, 'm8': 1, 'm9': 0, 'm10': 0},
    #               'Agent6': {'m1': 0, 'm2': 1, 'm3': 0, 'm4': 1, 'm5': 0, 'm6': 1, 'm7': 0, 'm8': 1, 'm9': 1, 'm10': 0},
    #               'Agent7': {'m1': 0, 'm2': 1, 'm3': 0, 'm4': 1, 'm5': 0, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 0, 'm10': 0},
    #               'Agent8': {'m1': 1, 'm2': 1, 'm3': 0, 'm4': 1, 'm5': 0, 'm6': 1, 'm7': 1, 'm8': 0, 'm9': 1, 'm10': 0},
    #               'Agent9': {'m1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 1, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 0, 'm10': 0},
    #               'Agent10': {'m1': 1, 'm2': 1, 'm3': 0, 'm4': 0, 'm5': 1, 'm6': 0, 'm7': 1, 'm8': 1, 'm9': 1,
    #                           'm10': 1},
    #               'Agent11': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 1, 'm5': 1, 'm6': 1, 'm7': 1, 'm8': 0, 'm9': 0,
    #                           'm10': 1},
    #               'Agent12': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 0, 'm5': 0, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 1,
    #                           'm10': 0},
    #               'Agent13': {'m1': 1, 'm2': 0, 'm3': 1, 'm4': 0, 'm5': 0, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 0,
    #                           'm10': 0},
    #               'Agent14': {'m1': 0, 'm2': 1, 'm3': 0, 'm4': 1, 'm5': 0, 'm6': 0, 'm7': 1, 'm8': 1, 'm9': 1,
    #                           'm10': 1},
    #               'Agent15': {'m1': 0, 'm2': 0, 'm3': 1, 'm4': 0, 'm5': 1, 'm6': 1, 'm7': 1, 'm8': 1, 'm9': 0,
    #                           'm10': 0},
    #               'Agent16': {'m1': 1, 'm2': 0, 'm3': 0, 'm4': 1, 'm5': 1, 'm6': 1, 'm7': 0, 'm8': 0, 'm9': 0,
    #                           'm10': 1},
    #               'Agent17': {'m1': 1, 'm2': 0, 'm3': 1, 'm4': 0, 'm5': 0, 'm6': 0, 'm7': 1, 'm8': 1, 'm9': 0,
    #                           'm10': 1},
    #               'Agent18': {'m1': 0, 'm2': 1, 'm3': 0, 'm4': 0, 'm5': 1, 'm6': 0, 'm7': 0, 'm8': 0, 'm9': 0,
    #                           'm10': 0},
    #               'Agent19': {'m1': 1, 'm2': 1, 'm3': 1, 'm4': 1, 'm5': 1, 'm6': 1, 'm7': 1, 'm8': 1, 'm9': 0,
    #                           'm10': 0}}
    #
    # inst = Instance(valuations=valuations, items=items)
    # print(inst)
    # divide(algorithm=per_category_round_robin, instance=inst,
    #        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    #        initial_agent_order=order)

    from IPython.display import display, Image
    import base64
    images_data = []


    def store_visualization(img_base64):  # used to get us the images from fairpyx !
        images_data.append(img_base64)


    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    from PIL import Image


    def display_all_images_with_matplotlib(images_data):
        for img_base64 in images_data:
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data))
            plt.imshow(img)
            plt.axis('off')  # Hide axes for a cleaner look
            plt.show()

    #logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    order = ['Agent1', 'Agent8', 'Agent5', 'Agent7', 'Agent2', 'Agent9', 'Agent4', 'Agent3', 'Agent10', 'Agent6']
    items = ['m1']
    item_categories = {'c1': ['m1'], 'c2': [], 'c3': [], 'c4': [], 'c5': [], 'c6': [], 'c7': [], 'c8': [], 'c9': [],
                       'c10': []}
    agent_category_capacities = {
        'Agent1': {'c1': 20, 'c2': 15, 'c3': 8, 'c4': 1, 'c5': 2, 'c6': 10, 'c7': 1, 'c8': 11, 'c9': 4, 'c10': 12},
        'Agent2': {'c1': 19, 'c2': 3, 'c3': 1, 'c4': 1, 'c5': 5, 'c6': 6, 'c7': 7, 'c8': 9, 'c9': 18, 'c10': 16},
        'Agent3': {'c1': 5, 'c2': 10, 'c3': 11, 'c4': 2, 'c5': 2, 'c6': 8, 'c7': 10, 'c8': 4, 'c9': 7, 'c10': 12},
        'Agent4': {'c1': 15, 'c2': 19, 'c3': 1, 'c4': 15, 'c5': 4, 'c6': 13, 'c7': 11, 'c8': 12, 'c9': 5, 'c10': 7},
        'Agent5': {'c1': 5, 'c2': 16, 'c3': 4, 'c4': 13, 'c5': 5, 'c6': 9, 'c7': 15, 'c8': 16, 'c9': 4, 'c10': 16},
        'Agent6': {'c1': 14, 'c2': 17, 'c3': 18, 'c4': 6, 'c5': 10, 'c6': 4, 'c7': 1, 'c8': 6, 'c9': 1, 'c10': 18},
        'Agent7': {'c1': 19, 'c2': 5, 'c3': 3, 'c4': 17, 'c5': 4, 'c6': 3, 'c7': 11, 'c8': 14, 'c9': 17, 'c10': 8},
        'Agent8': {'c1': 10, 'c2': 1, 'c3': 11, 'c4': 19, 'c5': 12, 'c6': 3, 'c7': 3, 'c8': 4, 'c9': 4, 'c10': 19},
        'Agent9': {'c1': 15, 'c2': 4, 'c3': 18, 'c4': 19, 'c5': 15, 'c6': 10, 'c7': 2, 'c8': 5, 'c9': 11, 'c10': 12},
        'Agent10': {'c1': 9, 'c2': 12, 'c3': 3, 'c4': 20, 'c5': 17, 'c6': 1, 'c7': 1, 'c8': 7, 'c9': 20, 'c10': 15}}
    valuations = {'Agent1': {'m1': 100}, 'Agent2': {'m1': 100}, 'Agent3': {'m1': 100}, 'Agent4': {'m1': 100},
                  'Agent5': {'m1': 100}, 'Agent6': {'m1': 100}, 'Agent7': {'m1': 100}, 'Agent8': {'m1': 100},
                  'Agent9': {'m1': 100}, 'Agent10': {'m1': 100}}
    sum_agent_category_capacities = {agent: sum(cap.values()) for agent, cap in agent_category_capacities.items()}
    instance = Instance(valuations=valuations, items=items, agent_capacities=sum_agent_category_capacities)
    # print(instance)
    start_time = time.perf_counter()
    divide(algorithm=per_category_capped_round_robin, instance=instance, item_categories=item_categories,
           agent_category_capacities=agent_category_capacities, initial_agent_order=order)
    end_time = time.perf_counter()
    print(f'time taken{end_time - start_time}')
    # with callback (storing graph images)
    start_time = time.perf_counter()
    divide(algorithm=per_category_capped_round_robin, instance=instance, item_categories=item_categories,
           agent_category_capacities=agent_category_capacities, initial_agent_order=order, callback=store_visualization)
    end_time = time.perf_counter()
    print(f'time taken{end_time - start_time}')
    display_all_images_with_matplotlib(images_data)
