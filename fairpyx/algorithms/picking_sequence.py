"""
Allocate course seats using a picking sequence.

Three interesting special cases of a picking-sequence are: round-robin, balanced round-robin, and serial dictatorship.

Programmer: Erel Segal-Halevi
Since: 2023-06
"""

from itertools import cycle
from fairpyx import Instance, AllocationBuilder

import logging

logger = logging.getLogger(__name__)


# TODO ask Erel : 1) give function in AllocationBuilder tracks agent_capacity in the classic single dimensional
#  way (as if its {'agent1':1,.....}) , YES WE PASSED THEM AS ARGUMENTS to our algorithm per_category_round_robin
#  (alloc:AllocationBuilder, item_categories:dict, agent_category_capacities: dict) but still when calling
#  alloc.give(yadayada...) it treats the builtin attribute which is 1d when ours is more
#  {'agent1':{category1:2, ....},.....} (2 keys , 1 for agent-category , and other for-category-capacity)
def per_category_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict):
    """
    per category round-robin is an allocation algorithm which guarantees EF1 (envy-freeness up to 1 good) allocation
    under settings in which agent-capacities are equal across all agents,
    no capacity-inequalities are allowed since this algorithm doesnt provie a cycle-prevention mechanism
    TLDR: same partition constriants , same capacities , may have different valuations across agents  -> EF1 allocation

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
    :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
    paired with a dictionary of category-capacity.

    >>> # Example 1
    >>> from fairpyx import  divide
    >>> order=(1,2)
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    >>> valuations = {'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
    >>>{'Agent1':['m1','m3'],'Agent2':['m2']}

    >>> # Example 2
    >>> from fairpyx import  divide
    >>> order=(1,3,2)
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2','m3']}
    >>> agent_category_capacities = {'Agent1': {'c1':3}, 'Agent2': {'c1':3},'Agent3': {'c1':3}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':5},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':5,'m2':6,'m3':5}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
    >>> {'Agent1':['m2'],'Agent2':['m1'],'Agent3':['m3']}


     >>> # Example 3  (4 agents ,4 items)
    >>> from fairpyx import  divide
    >>> order=(1,2,3,4)
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1}, 'Agent2': {'c1':1,'c2':1},'Agent3': {'c1':1,'c2':1}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':5},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':5,'m2':6,'m3':5}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
    >>> {'Agent1':['m1'],'Agent2':['m2'],'Agent3':['m3'],'Agent4':['m4']} #TODO ask Erel if i should take treat it as this or each allocation categorized for each agent ? like{'Agent1':{'c1':['m1'}...}....}
    """

    pass


def capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict):
    """
    CRR (capped round-robin) algorithm
    TLDR: single category , may have differnt capacities , maye have different valuations -> F-EF1 (feasible envy-freeness up to 1 good) allocation

        :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.

        >>> # Example 1 (2 agents 1 of them with capacity of 0)
        >>> from fairpyx import  divide
        >>> order=(1,2)
        >>> items=['m1']
        >>> item_categories = {'c1': ['m1']}
        >>> agent_category_capacities = {'Agent1': {'c1':0}, 'Agent2': {'c1':1}}
        >>> valuations = {'Agent1':{'m1':0},'Agent2':{'m1':420}}
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
        >>>{'Agent1':None,'Agent2':['m1']}

        >>> # Example 2 (3 agents , 4 items)
        >>> from fairpyx import  divide
        >>> order=(1,2,3)
        >>> items=['m1','m2','m3','m4']
        >>> item_categories = {'c1': ['m1', 'm2','m3','m4']}
        >>> agent_category_capacities = {'Agent1': {'c1':2}, 'Agent2': {'c1':2},'Agent3': {'c1':2}}
        >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1},'Agent3':{'m1':1,'m2':1,'m3':1}}
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
        >>> {'Agent1':['m1','m4'],'Agent2':['m2'],'Agent3':['m3']}


         >>> # Example 3  (to show that F-EF (feasible envy-free) is sometimes achievable in good scenarios)
        >>> from fairpyx import  divide
        >>> order=(1,2)
        >>> items=['m1','m2']
        >>> item_categories = {'c1': ['m1', 'm2']}
        >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':1},'Agent3': {'c1':1}}
        >>> valuations = {'Agent1':{'m1':10,'m2':5},'Agent2':{'m1':5,'m2':10}}
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
        >>> {'Agent1':['m1'],'Agent2':['m2']} #TODO ask Erel if i should take treat it as this or each allocation categorized for each agent ? like{'Agent1':{'c1':['m1'}...}....}
        """
    pass


def picking_sequence(alloc: AllocationBuilder, agent_order: list):
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
    logger.info("\nPicking-sequence with items %s , agents %s, and agent-order %s", alloc.remaining_item_capacities,
                alloc.remaining_agent_capacities, agent_order)
    for agent in cycle(agent_order):
        if alloc.isdone():
            break
        if not agent in alloc.remaining_agent_capacities:
            continue
        potential_items_for_agent = set(alloc.remaining_items()).difference(alloc.bundles[agent])
        if len(potential_items_for_agent) == 0:
            logger.info("Agent %s cannot pick any more items: remaining=%s, bundle=%s", agent,
                        alloc.remaining_item_capacities, alloc.bundles[agent])
            alloc.remove_agent_from_loop(agent)
            continue
        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.effective_value(agent, item))
        alloc.give(agent, best_item_for_agent, logger)


def serial_dictatorship(alloc: AllocationBuilder, agent_order: list = None):
    """
    Allocate the given items to the given agents using the serial_dictatorship protocol, in the given agent-order.
    :param agents a list of Agent objects.
    :param agent_order (optional): a list of indices of agents. The agents will pick items in this order.
    :param items (optional): a list of items to allocate. Default is allocate all items.

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 10, "c2": 8, "c3": 6}
    >>> s2 = {"c1": 6, "c2": 8, "c3": 10}
    >>> agent_capacities = {"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3}      # 10 seats required
    >>> course_capacities = {"c1": 2, "c2": 3, "c3": 4}                       # 9 seats available
    >>> valuations = {"Alice": s1, "Bob": s1, "Chana": s2, "Dana": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(serial_dictatorship, instance=instance)
    {'Alice': ['c1', 'c2'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c3']}
    """
    if agent_order is None: agent_order = alloc.remaining_agents()
    agent_order = sum([alloc.remaining_agent_capacities[agent] * [agent] for agent in agent_order], [])
    picking_sequence(alloc, agent_order)


def round_robin(alloc: AllocationBuilder, agent_order: list = None):
    """
    Allocate the given items to the given agents using the round-robin protocol, in the given agent-order.
    :param agents a list of Agent objects.
    :param agent_order (optional): a list of indices of agents. The agents will pick items in this order.
    :param items (optional): a list of items to allocate. Default is allocate all items.

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 10, "c2": 8, "c3": 6}
    >>> s2 = {"c1": 6, "c2": 8, "c3": 10}
    >>> agent_capacities = {"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3}      # 10 seats required
    >>> course_capacities = {"c1": 2, "c2": 3, "c3": 4}                       # 9 seats available
    >>> valuations = {"Alice": s1, "Bob": s1, "Chana": s2, "Dana": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(round_robin, instance=instance)
    {'Alice': ['c1', 'c2'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c3']}
    """
    if agent_order is None: agent_order = list(alloc.remaining_agents())
    picking_sequence(alloc, agent_order)


def bidirectional_round_robin(alloc: AllocationBuilder, agent_order: list = None):
    """
    Allocate the given items to the given agents using the bidirectional-round-robin protocol (ABCCBA), in the given agent-order.
    :param agents a list of Agent objects.
    :param agent_order (optional): a list of indices of agents. The agents will pick items in this order.
    :param items (optional): a list of items to allocate. Default is allocate all items.
    :return a list of bundles; each bundle is a list of items.

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 10, "c2": 8, "c3": 6}
    >>> s2 = {"c1": 6, "c2": 8, "c3": 10}
    >>> agent_capacities = {"Alice": 2, "Bob": 3, "Chana": 2, "Dana": 3}      # 10 seats required
    >>> course_capacities = {"c1": 2, "c2": 3, "c3": 4}                       # 9 seats available
    >>> valuations = {"Alice": s1, "Bob": s1, "Chana": s2, "Dana": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(bidirectional_round_robin, instance=instance)
    {'Alice': ['c1', 'c3'], 'Bob': ['c1', 'c2', 'c3'], 'Chana': ['c2', 'c3'], 'Dana': ['c2', 'c3']}
    """
    if agent_order is None: agent_order = alloc.remaining_agents()
    picking_sequence(alloc, list(agent_order) + list(reversed(agent_order)))


round_robin.logger = picking_sequence.logger = serial_dictatorship.logger = logger

### MAIN

if __name__ == "__main__":
    import doctest, sys

    print("\n", doctest.testmod(), "\n")

    # sys.exit()

    # logger.addHandler(logging.StreamHandler(sys.stdout))
    # logger.setLevel(logging.INFO)

    # from fairpyx.adaptors import divide_random_instance

    # print("\n\nRound robin:")
    # divide_random_instance(algorithm=round_robin, 
    #                        num_of_agents=30, num_of_items=10, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)
    # print("\n\nBidirectional round robin:")
    # divide_random_instance(algorithm=bidirectional_round_robin, 
    #                        num_of_agents=30, num_of_items=10, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12], 
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)
