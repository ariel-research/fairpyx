from fairpyx import Instance, AllocationBuilder


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
    >>> order=[1,2,3,4]
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':5},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':5,'m2':6,'m3':5}}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
    >>> {'Agent1':['m1'],'Agent2':['m2'],'Agent3':['m3'],'Agent4':['m4']} #TODO ask Erel if i should take treat it as this or each allocation categorized for each agent ? like{'Agent1':{'c1':['m1'}...}....}
    """

    pass


def capped_round_robin(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict, order: list):
    """
    this is Algorithm 2
    CRR (capped round-robin) algorithm
    TLDR: single category , may have differnt capacities , maye have different valuations -> F-EF1 (feasible envy-freeness up to 1 good) allocation

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
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order = order)
        >>>{'Agent1':None,'Agent2':['m1']}

        >>> # Example 2 (3 agents , 4 items)
        >>> from fairpyx import  divide
        >>> order=[1,2,3]
        >>> items=['m1','m2','m3','m4']
        >>> item_categories = {'c1': ['m1', 'm2','m3','m4']}
        >>> agent_category_capacities = {'Agent1': {'c1':2}, 'Agent2': {'c1':2},'Agent3': {'c1':2}}
        >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':1},'Agent3':{'m1':1,'m2':1,'m3':1}}
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
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
        >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities,order=order)
        >>> {'Agent1':[],'Agent2':['m1'],'Agent3':['m3','m4'],'Agent4':['m2','m5','m6']}

        """
    pass
#TODO finish to algorithm 5 by 15/4