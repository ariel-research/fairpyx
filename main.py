import doctest
import pytest
import fairpyx as fpx
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.adaptors import divide, divide_random_instance

example_instance= instance = Instance(
        valuations={'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}})
example_item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
example_agent_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
def per_category_round_robin(alloc:AllocationBuilder,**kwargs):
    """
        This dummy algorithm gives one item to the first agent, and all items to the second agent.

        >>> divide(per_category_round_robin, instance=example_instance,agent_capacities=example_agent_capacities,item_categories=example_item_categories)
        {'Agent1': [], 'Agent2': []}
        """
#TODO for now our impl is empty only for the same of showing the structure of the work
    remaining_agents = list(alloc.remaining_agents())  # `remaining_agents` returns the list of agents with remaining capacities.
    remaining_items = list(alloc.remaining_items())


# def per_category_round_robin_2(alloc: AllocationBuilder, **kwargs):
#     """
#         This dummy algorithm gives one item to the first agent, and all items to the second agent.
#
#         >>> divide(per_category_round_robin_2, instance=example_instance,agent_capacities=example_agent_capacities,item_categories=example_item_categories)
#         {'Agent1': ['m1'], 'Agent2': ['m2','m3']}
#         """
#     # TODO for now our impl is empty only for the same of showing the structure of the work
#     remaining_agents = list(
#         alloc.remaining_agents())  # `remaining_agents` returns the list of agents with remaining capacities.
#     remaining_items = list(alloc.remaining_items())
#     alloc.give(remaining_agents[0],remaining_items[0])
#     alloc.give_bundle(remaining_agents[1],remaining_items)
#     #TODO this fails because we're not using the same agent_capacity anymore (which is bound with instance)


def example1():
    # TODO dont pass item_capacities as its automatically in the constructor -> constant_function(1)
    instance = Instance(
        valuations={'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
        # no need for item_capacity since by default its 1 (which is what we want)
        # TODO agent capacity has been changed its now 2d not 1d , thats why we pass it as kwarg when invoking divide
        # TODO regarding conflicts ask Erel
        #  (i believe there is no conflicts since each item is only of capacity 1 )

                        )
    # all the necessary parameters (TODO there might be optional ones)
    items = set()
    items.add('m1')
    items.add('m2')
    items.add('m3')
    item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    agent_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    # in Biswas&Barman we assume same capacities, so we can avoid conflicts when switching bundles
    # along envy-cycle in the envy-graph
    invoke_divide(instance,agent_capacities=agent_capacities,item_categories=item_categories) #TODO all the optional kwargs go here

def invoke_divide_random_instance():
    divide_random_instance(algorithm=per_category_round_robin()
                           ,num_of_agents=30
                           ,num_of_items=10
                           ,agent_capacity_bounds=[10, 10]
                           ,item_capacity_bounds=[1,1]
                           ,item_base_value_bounds=[1, 100]
                           ,item_subjective_ratio_bounds=[0.5, 1.5]
                           ,normalized_sum_of_values=100
                           ,random_seed=1
                           )# TODO num of categories ... , agent_capacities (this is not suppoered by this function )
                            # TODO ask Erel , since this function invokes Instance.random_uniform() , should i extend it by inheritance or simply an outer function for the extra kwargs ?



def invoke_divide(instance:Instance,**kwargs):
    #print(instance)
    #print(kwargs)
    divide(algorithm=per_category_round_robin,instance=instance,**kwargs) ## TODO remember the allocation returns from here (its alloc.sorted())


def test_per_category_round_robin():
    expected_result = {'Agent1': [], 'Agent2': []}
    result = divide(algorithm=per_category_round_robin, instance=example_instance,
                    agent_capacities=example_agent_capacities, item_categories=example_item_categories)
    assert result == expected_result
if __name__ == '__main__':
    #example1()
    print(doctest.testmod())
