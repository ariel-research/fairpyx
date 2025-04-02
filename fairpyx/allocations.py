"""
Utils for post-processing allocations.
"""

import numpy as np
from collections import defaultdict
from fairpyx import Instance 

# The following constant is used as an item value, to indicate that this item must not be allocated to the agent.
FORBIDDEN_ALLOCATION = -np.inf


def validate_allocation(instance:Instance, allocation:dict, title:str="", allow_multiple_copies:bool=False):
    """
    Validate that the given allocation is feasible for the given input-instance.
    Checks agent capacities, item capacities, and uniqueness of items.

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 4, "Bob": 8}, 
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 3}, 
    ...   item_weights     = {"c1": 4, "c2": 3, "c3": 2}, 
    ...   valuations       = {"Alice": {"c1": 11, "c2": 22, "c3": 33}, "Bob": {"c1": 33, "c2": 44, "c3": 55}})
    >>> validate_allocation(instance, allocation = {"Alice": ["c1", "c2"]})
    >>> validate_allocation(instance, allocation = {"Alice": ["c1", "c2", "c3"]})
    Traceback (most recent call last):
    ...
    ValueError: : Agent Alice has capacity 4, but received more items: ['c1', 'c2', 'c3'] with total weight: 9.
    >>> validate_allocation(instance, allocation = {"Alice": ["c2", "c2"]})
    Traceback (most recent call last):
    ...
    ValueError: : Agent Alice received two or more copies of the same item. Bundle: ['c2', 'c2'].
    >>> validate_allocation(instance, allocation = {"Alice": ["c1", "c2"], "Bob": ["c2","c3"]})
    >>> validate_allocation(instance, allocation = {"Alice": ["c1", "c2"], "Bob": ["c2","c1"]})
    Traceback (most recent call last):
    ...
    ValueError: : Item c1 has capacity 1, but is given to more agents: ['Alice', 'Bob'].
    >>> validate_allocation(instance, allocation = {"Alice": ["c2"], "Bob": ["c2","c3"]})
    Traceback (most recent call last):
    ...
    ValueError: : Wasteful allocation:
    Item c3 has remaining capacity: 3>['Bob'].
    Agent Alice has remaining capacity: 4>['c2'].
    Agent Alice values Item c3 at 33.
    """

    ### validate agent capacity and uniqueness:
    agents_below_their_capacity = []
    for agent,bundle in allocation.items():
        agent_capacity = instance.agent_capacity(agent)
        bundle_weights = [instance.item_weight(item) for item in bundle]
        bundle_total_weight = sum(bundle_weights)
        if bundle_total_weight > agent_capacity and bundle_total_weight - max(bundle_weights) >= agent_capacity: 
            raise ValueError(f"{title}: Agent {agent} has capacity {agent_capacity}, but received more items: {bundle} with total weight: {bundle_total_weight}.")
        if (not allow_multiple_copies) and len(set(bundle))!=len(bundle):
            raise ValueError(f"{title}: Agent {agent} received two or more copies of the same item. Bundle: {bundle}.")
        if bundle_total_weight < agent_capacity:
            agents_below_their_capacity.append(agent)

    ### validate item capacity:
    map_item_to_list_of_owners = defaultdict(list)
    items_below_their_capacity = []
    for agent,bundle in allocation.items():
        for item in bundle:
            map_item_to_list_of_owners[item].append(agent)
    for item,list_of_owners in map_item_to_list_of_owners.items():
        item_capacity = instance.item_capacity(item)
        if len(list_of_owners) > item_capacity:
            raise ValueError(f"{title}: Item {item} has capacity {item_capacity}, but is given to more agents: {list_of_owners}.")
        if len(list_of_owners) < item_capacity:
            items_below_their_capacity.append(item)

    ### validate no waste:
    for agent in agents_below_their_capacity:
        for item in items_below_their_capacity:
            bundle = allocation[agent]
            value = instance.agent_item_value(agent,item)
            if item not in bundle and value>0:
                item_message = f"Item {item} has remaining capacity: {instance.item_capacity(item)}>{map_item_to_list_of_owners[item]}."
                agent_message = f"Agent {agent} has remaining capacity: {instance.agent_capacity(agent)}>{bundle}."
                value_message = f"Agent {agent} values Item {item} at {value}."
                raise ValueError(f"{title}: Wasteful allocation:\n{item_message}\n{agent_message}\n{value_message}")


def rounded_allocation(allocation_matrix:dict, digits:int):
    return {agent:{item:np.round(allocation_matrix[agent][item],digits) for item in allocation_matrix[agent].keys()} for agent in allocation_matrix.keys()}


def allocation_is_fractional(allocation:dict)->bool:
    """
    Weak check if the given allocation is fractional.
    The check is made only on one arbitrary bundle.

    >>> allocation_is_fractional({"agent1": {"item1": 0.3, "item2": 0.4}})
    True
    >>> allocation_is_fractional({"agent1": ["item1", "item2"]})
    False
    """
    arbitrary_bundle = next(iter(allocation.values()))
    if isinstance(arbitrary_bundle,list):
        return False
    elif isinstance(arbitrary_bundle,dict):
        arbitrary_value = next(iter(arbitrary_bundle.values()))
        if isinstance(arbitrary_value,float):
            return True
    raise ValueError(f"Bundle format is unknown: {arbitrary_bundle}")


class AllocationBuilder:
    """
    A class for incrementally constructing an allocation.

    Whenever an item is given to an agent (via the 'give' method),
    the class automatically updates the 'remaining_item_capacities' and the 'remaining_agent_capacities'.
    It also updates the 'remaining_conflicts' by adding a conflict between the agent and the item, so that it is not assigned to it anymore.

    Once you finish adding items, use "sorted" to get the final allocation (where each bundle is sorted alphabetically).

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3}, 
    ...   item_capacities  = {"c1": 4, "c2": 5}, 
    ...   item_weights     = {"c1": 2, "c2": 4},
    ...   valuations       = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}})
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Bob', 'c1')
    >>> alloc.remaining_agent_capacities
    {'Alice': 2, 'Bob': 1}
    >>> alloc.remaining_item_capacities
    {'c1': 3, 'c2': 5}
    >>> alloc.remaining_conflicts
    {('Bob', 'c1')}

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3}, 
    ...   item_capacities  = {"c1": 4, "c2": 5}, 
    ...   item_weights     = {"c1": 2, "c2": 4},
    ...   valuations       = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}},
    ...   agent_conflicts  = {"Alice": ["c2"]},
    ...   item_conflicts  = {"c1": ["c2"]})
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.remaining_conflicts
    {('Alice', 'c2')}
    >>> alloc.give('Bob', 'c1')    
    >>> sorted(alloc.remaining_conflicts)
    [('Alice', 'c2'), ('Bob', 'c1'), ('Bob', 'c2')]
    """
    def __init__(self, instance:Instance):
        self.instance = instance
        self.allow_multiple_copies = False
        self.remaining_agent_capacities = {agent: instance.agent_capacity(agent) for agent in instance.agents if instance.agent_capacity(agent) > 0}
        self.remaining_item_capacities = {item: instance.item_capacity(item) for item in instance.items if instance.item_capacity(item) > 0}
        self.remaining_conflicts = {(agent,item) for agent in self.remaining_agents() for item in self.instance.agent_conflicts(agent)}
        self.bundles = {agent: set() for agent in instance.agents}    # Each bundle is a set, since each agent can get at most one seat in each course

    def set_allow_multiple_copies(self, flag):
        self.allow_multiple_copies = flag
        if flag:
            self.bundles = {agent: list() for agent in self.instance.agents}

    def isdone(self)->bool:
        """
        Return True if either all items or all agents have exhausted their capacity - so we are done.
        """
        return len(self.remaining_item_capacities) == 0 or len(self.remaining_agent_capacities) == 0

    def remaining_items(self)->list:
        """
        Return the items with positive remaining capacity.
        """
        return self.remaining_item_capacities.keys()

    def remaining_items_for_agent(self, agent)->list:
        """
        Return the items with positive remaining capacity, that are available for the agent
        (== the agent does not already have them, and there are no item-conflicts or agent-conflicts)
        """
        return [item for item in self.remaining_items() if (agent,item) not in self.remaining_conflicts]

    def remaining_agents(self)->list:
        """
        Return the agents with positive remaining capacity.
        """
        return self.remaining_agent_capacities.keys()

    def remaining_instance(self)->Instance:
        """
        Construct a fresh input instance, based on the remaining agents and items.
        """
        return Instance(
            valuations=self.instance.agent_item_value,                 # base valuations are the same as in the original instance
            agent_capacities=self.remaining_agent_capacities,          # agent capacities may be smaller than in the original instance
            agent_entitlements=self.instance.agent_entitlement,        # agent entitlement is the same as in the original instance
            agent_conflicts=self.instance.agent_conflicts,             # agent conflicts are the same as in the original instance
            agents=self.remaining_agents(),                            # agent list may be smaller than in the original instance
            item_capacities=self.remaining_item_capacities,            # item capacities may be smaller than in the original instance 
            item_weights=self.instance._item_weights,                    # base item weights are the same as in the original instance
            item_conflicts=self.instance.item_conflicts,               # item conflicts are the same as in the original instance   
            items=self.remaining_items())                              # item list may be smaller than in the original instance 
    
    def agent_bundle_value(self, agent:any, bundle:list)->float:
        return self.instance.agent_bundle_value(agent,bundle)

    def effective_value(self, agent:any, item:any)->float:
        """
        Return the agent's value for the item, if there is no conflict;
        otherwise, returns -infinity.
        """
        if (agent,item) in self.remaining_conflicts:
            return FORBIDDEN_ALLOCATION
        else:
            return self.instance.agent_item_value(agent,item)
        
    def remove_item_from_loop(self, item:any):
        """
        Remove the given item from further consideration by the allocation algorithm.
        """
        del self.remaining_item_capacities[item]

    def remove_agent_from_loop(self, agent:any):
        """
        Remove the given agent from further consideration by the allocation algorithm (the agent keeps holding his items)
        """
        del self.remaining_agent_capacities[agent]

    def give(self, agent:any, item:any, logger=None):
        if agent not in self.remaining_agent_capacities:
            raise ValueError(f"Agent {agent} has no remaining capacity for item {item}")
        if item not in self.remaining_item_capacities:
            raise ValueError(f"Item {item} has no remaining capacity for agent {agent}")
        if (agent,item) in self.remaining_conflicts:
            raise ValueError(f"Agent {agent} is not allowed to take item {item} due to a conflict")
        if self.allow_multiple_copies:
            self.bundles[agent].append(item)
        else:
            self.bundles[agent].add(item)
        if logger is not None:
            logger.info("Agent %s takes item %s with value %s", agent, item, self.instance.agent_item_value(agent, item))
        
        # Update capacities:
        self.remaining_agent_capacities[agent] -= self.instance.item_weight(item)
        if self.remaining_agent_capacities[agent] <= 0:
            self.remove_agent_from_loop(agent)
        self.remaining_item_capacities[item] -= 1
        if self.remaining_item_capacities[item] <= 0:
            self.remove_item_from_loop(item)
        if not self.allow_multiple_copies:
            self._update_conflicts(agent,item)

    def give_bundle(self, agent:any, new_bundle:list, logger=None):
        for item in new_bundle:
            self.give(agent, item, logger=None)
        if logger is not None:
            logger.info("Agent %s takes bundle %s with value %s", agent, new_bundle, self.agent_bundle_value(agent, new_bundle))


    def give_bundles(self, new_bundles:dict, logger=None):
        """
        Add an entire set of bundles to this allocation.
        NOTE: No validity check is done - use at your own risk!
        """
        map_agent_to_num_of_items = {agent: len(bundle) for agent,bundle in new_bundles.items()}
        map_agent_to_weights_of_items = {agent: sum([self.instance.item_weight(item) for item in bundle]) for agent,bundle in new_bundles.items()}
        map_item_to_num_of_owners = {item: 0 for item in self.instance.items}
        for agent,bundle in new_bundles.items():
            for item in bundle:
                map_item_to_num_of_owners[item] += 1

        for agent,num_of_items in map_agent_to_num_of_items.items():
            if num_of_items==0: continue
            if agent not in self.remaining_agent_capacities or self.remaining_agent_capacities[agent]<num_of_items:
                raise ValueError(f"Agent {agent} has no remaining capacity for {num_of_items} new items")
            self.remaining_agent_capacities[agent] -= map_agent_to_weights_of_items[agent]
            if self.remaining_agent_capacities[agent] <= 0:
                self.remove_agent_from_loop(agent)

        for item,num_of_owners in map_item_to_num_of_owners.items():
            if num_of_owners==0: continue
            if item not in self.remaining_item_capacities or self.remaining_item_capacities[item]<num_of_owners:
                raise ValueError(f"Item {item} has no remaining capacity for {num_of_owners} new agents")
            self.remaining_item_capacities[item] -= num_of_owners
            if self.remaining_item_capacities[item] <= 0:
                self.remove_item_from_loop(item)

        for agent,bundle in new_bundles.items():
            self.bundles[agent].update(bundle)
            self._update_conflicts(agent,item)

    def _update_conflicts(self, receiving_agent:any, received_item:any):
        """
        Update the list of agent-item conflicts after giving `received_item` to `receiving_agent`:
        * `receiving_agent` has a new conflict with `received_item`, as cannot get the same item twice.
        * `receiving_agent` has a new conflict with any item in conflict with `received_item`, as cannot get both of them at the same time.
        """
        self.remaining_conflicts.add( (receiving_agent,received_item) )
        for conflicting_item in self.instance.item_conflicts(received_item):
            self.remaining_conflicts.add( (receiving_agent,conflicting_item) )

    def sorted(self):
        return {agent: sorted(bundle) for agent,bundle in self.bundles.items()}


if __name__ == "__main__":
    import doctest
    print(doctest.testmod(optionflags=doctest.ELLIPSIS+doctest.NORMALIZE_WHITESPACE))