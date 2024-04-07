import pytest
from fairpyx.instances import Instance
from fairpyx.algorithms import picking_sequence
from fairpyx import divide
import numpy as np
import pytest


def random_uniform_extended(num_of_agents: int, num_of_items: int,
                            num_of_categories: int,
                            agent_capacity_bounds: tuple[int, int],
                            item_capacity_bounds: tuple[int, int],
                            item_base_value_bounds: tuple[int, int],
                            item_subjective_ratio_bounds: tuple[float, float],
                            normalized_sum_of_values: int,
                            agent_name_template="s{index}", item_name_template="c{index}",
                            random_seed: int = None,
                            equal_capacities: bool = False):
    inst = Instance(
        valuations={"Agent1": {'item1': 1}})  # helping varialbe (for the sake of calling inner # non-static-methods)
    result_instance = inst.random_uniform(num_of_agents=
                                          num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=
                                          agent_capacity_bounds, item_capacity_bounds=item_capacity_bounds,
                                          item_base_value_bounds=item_base_value_bounds
                                          , item_subjective_ratio_bounds=item_subjective_ratio_bounds
                                          , normalized_sum_of_values=normalized_sum_of_values,
                                          agent_name_template=agent_name_template
                                          , item_name_template=item_name_template, random_seed=random_seed)
    if random_seed is not None:
        random_seed = np.random.randint(1, 2 ** 31)
    np.random.seed(random_seed)

    # TODO make ranmdomize categories
    category_string_template = "Category:{cat}"
    categories = {category_string_template.format(cat=cat): [] for cat in range(num_of_categories)}
    if not equal_capacities:
        agent_capacities_2d = {
            agent: {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for category in
                    categories} for agent
            in result_instance.agents}
    else:
        random_capacity=np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1)
        agent_capacities_2d = {
            agent: {category: random_capacity for category in
                    categories} for agent
            in result_instance.agents}
    # print(f"and categories are :{categories}\n and agent capacities are : {agent_capacities_2d}") # TODO remove after finish
    #NOTE: to make sure a case of empty category doesnt exist we do a 1 run on all the  categories and fill them with random item
    temporary_items=list(result_instance.items).copy()

    for cat in categories:
        random_item = np.random.choice(temporary_items)
        categories[cat].append(random_item)
        temporary_items.remove(random_item)
    for item in temporary_items:
        random_category = np.random.choice(list(categories.keys()))
        categories[random_category].append(item)

    return result_instance, agent_capacities_2d, categories


def random_instance(equal_capacities):  # todo add randomization for arguments .
    random_num_of_agents = np.random.randint(1, 10 + 1)
    random_num_of_items = np.random.randint(1, 10 + 1)
    random_num_of_categories = np.random.randint(1, random_num_of_items + 1)
    random_instance = random_uniform_extended(
        num_of_categories=random_num_of_categories,
        num_of_agents=random_num_of_agents, num_of_items=random_num_of_items,
        agent_capacity_bounds=[1, 20], item_capacity_bounds=[1, 50],
        item_base_value_bounds=[1, 200], item_subjective_ratio_bounds=[0.5, 1.5],
        agent_name_template="agent{index}", item_name_template="item{index}",
        normalized_sum_of_values=1000, equal_capacities=equal_capacities
    )
    return random_instance


# TODO the only logical way to test random instances , is either
# 1) time-complexity-based tests
# 2) validate Envy-freeness up to 1 good in worst case scenario
def is_fef1(alloc: dict,instance:Instance,item_ctegories:dict,agent_category_capacities:dict) -> bool:
    # TODO implement as the definition says for every pair of agents (i,j) vi(xi) >= vi(Xj-{certain item}) we need to
    #  calculate how much items every agent has in his bundle divided by categories and make sure to only count k of
    #  them as k stands for the capacity of agent i for that category for each category im going to loop over  the
    #  items in each agent bundle and check if item in category i'll be making a dict agent_category_items={agent:{
    #  category:[items.....]}} so that way we can calculate things easily by simply len(agent_category_items[agent][
    #  category]) then in case agent i has capacity < agent j capacity we'll only take in consideration the amount of
    #  i's capacity items in j's bundle (if agent i has capacity of k in category c and agent j got in his allocated
    #  bundle more than k items which belong to that category agent i shall only take in consideration k items when
    #  calulating the valuation of items in j's bundle)
    agent_categorized_allocation:dict[dict[list]]
    agent_categorized_allocation = {agent:{category:[] for category in item_ctegories.keys()} for agent in agent_category_capacities.keys()}
    # agent_category_items[0] = {}
    # agent_category_items[0][0] = []
    # print(f"\nitems categories are : {item_ctegories}")
    # print(f"\nagent category capacities : {agent_category_capacities}")
    # print(f"allocation is: {alloc}") # alloc is empty for now , its hard to tell if the upcoming lines of code would achieve it
    # TODO : for the sake of implementing the algorithm im going to change the allocation to something which is non-empty
    #alloc={'agent1': [], 'agent2': [], 'agent3': [], 'agent4': []}
    alloc={agent:[]for agent in agent_category_capacities.keys()} # initializing
    for item_list in item_ctegories.values():
        for item in item_list:
            random_agent=np.random.choice(list(agent_category_capacities.keys()))
            alloc[random_agent].append(item)
    #TODO remove after completing the impl
    print(f"\nitems categories are : {item_ctegories}")
    print(f"\nagent category capacities : {agent_category_capacities}")
    print(
        f"allocation is: {alloc}")  # alloc is empty for now , its hard to tell if the upcoming lines of code would achieve it
    for cat in item_ctegories.keys():
        #print(f"{cat}")
        for agent in alloc.keys():
            for item in alloc[agent]:
               # print(f"{item}")
                if item in item_ctegories[cat]:
                    agent_categorized_allocation[agent][cat].append(item)
    print()
    print(f"agent new 2d dictionary : {agent_categorized_allocation}")
    #TODO so now we have a 2d version of the allocation categorized , to ease the check of F-EF1 ! :)
    for agent_i in agent_categorized_allocation.keys():
        for agent_j in agent_categorized_allocation.keys():
            #TODO here we do the  FEF1 check
           # if(value(best_feasible_subset(first=agent_i,second=agent_j)))
    return False


def test_algorithm_1():
    instance, agent_capacities_2d, categories = random_instance(equal_capacities=True)
    assert is_fef1(divide(algorithm=picking_sequence.per_category_round_robin, instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d),instance=instance
                   ,agent_category_capacities=agent_capacities_2d,item_ctegories=categories) is True


def test_algorithm_2():
    instance, agent_capacities_2d, categories = random_instance(equal_capacities=False)
    assert is_fef1(divide(algorithm=picking_sequence.capped_round_robin, instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d), instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_ctegories=categories) is True


def test_algorithm_3():

    assert False


def test_algorithm_4():

    assert False


def test_algorithm_5():

    assert False


def test_algorithm_6():

    assert False


def test_algorithm_7():

    assert False


if __name__ == "__main__":
    random_instance()
