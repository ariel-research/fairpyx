"""
An implementation of the Pytest of the algorithms  in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2010), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import random

import pytest
from fairpyx.instances import Instance
from fairpyx.algorithms import heterogeneous_matroid_constraints_algorithms
from fairpyx import divide
import numpy as np

# TODO lets do reconstruction here tomorrow ! , less functions ! beautify ! straight to point !

def random_uniform_extended(num_of_agents: int, num_of_items: int,
                            num_of_categories: int,
                            agent_capacity_bounds: tuple[int, int],
                            item_capacity_bounds: tuple[int, int],
                            item_base_value_bounds: tuple[int, int],
                            item_subjective_ratio_bounds: tuple[float, float],
                            normalized_sum_of_values: int,
                            agent_name_template="Agent{index}", item_name_template="m{index}",
                            random_seed: int = None,
                            equal_capacities: bool = False
                            ,equal_valuations: bool = False
                            )->tuple[Instance,dict,dict,list]:
    result_instance = Instance.random_uniform(num_of_agents=
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
    order = [agent for agent in result_instance.agents]
    random.shuffle(order)
    category_string_template = "c{cat}"
    categories = {category_string_template.format(cat=cat): [] for cat in range(1,num_of_categories+1)}
    # check whether equal capacities or doesn't matter
    if not equal_capacities:
        agent_capacities_2d = {
            agent: {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for category in
                    categories} for agent
            in result_instance.agents}
    else:
        random_capacity = np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1)
        agent_capacities_2d = {
            agent: {category: random_capacity for category in
                    categories} for agent
            in result_instance.agents}

    temporary_items = list(result_instance.items).copy()
    for cat in categories:
        random_item = np.random.choice(temporary_items)
        categories[cat].append(random_item)
        temporary_items.remove(random_item)
    for item in temporary_items:
        random_category = np.random.choice(list(categories.keys()))
        categories[random_category].append(item)
    result_instance.agent_capacity={agent:sum(agent_capacities_2d[agent][category]for category in agent_capacities_2d[agent]) for agent in agent_capacities_2d.keys()}
    print(f'wtffffffff {result_instance.agent_capacity}')
    # print(agent_capacities_2d)
    if equal_valuations:
        #check whether equal valuations or doesnt matter
        # case of random different valuations is already handled in random_uniform()
        # we can simply pick 1 and copy his valuations to the others .
        # or regenerate num_of_items random valuations and apply them to all
        random_valuation=np.random.uniform(low=item_base_value_bounds[0], high=item_base_value_bounds[1] + 1, size=num_of_items)
        # we need to normalize is
        sum_of_valuations = np.sum(random_valuation)
        ratio=normalized_sum_of_values/sum_of_valuations
        normalized_random_values =np.round(random_valuation * normalized_sum_of_values / sum_of_valuations).astype(int)

        normalized_random_valuation = {
            agent: dict(zip(result_instance.items,normalized_random_values
                            ))
            for agent in result_instance.agents
        }# so now we have random valuations generated and applied the same set to everyone so everyone has the same valuations in case of equal_valuations=True
    else:# means the valuations aren't supposed to be equal for every agent
        normalized_random_valuation = {agent:{item: result_instance.agent_item_value(agent,item)for item in result_instance.items}for agent in result_instance.agents} # we simply reconstructed the vaulations dict from the mappings because we have no access to private attributes of result_instance
        # no interest in inner attribute capacities whatsoever because we made our own outer parameter which matches the problems we're solving (categorized)
    item_capacities={item: result_instance.item_capacity(item) for item in result_instance.items}
    if item_base_value_bounds == (0,1):
        # TODO we need to change what we got from random uniform because it doesnt respect Binary valuations
        normalized_random_valuation = { # TODO check this and the actual valuation in the instance
            agent: dict(zip(result_instance.items, np.random.choice([0,1],size=len(result_instance.items))))
            for agent in result_instance.agents
        }
        #print(f"VALUATIONS{normalized_random_valuation}")

    #def __init__(self, valuations:any, agent_capacities:any=None, agent_entitlements:any=None, item_capacities:any=None, agent_conflicts:any=None, item_conflicts:any=None, agents:list=None, items:list=None):
    result_instance=Instance(valuations=normalized_random_valuation,item_capacities=item_capacities,agent_capacities=result_instance.agent_capacity)
    return result_instance, agent_capacities_2d, categories, order


def random_instance(equal_capacities:bool=False, equal_valuations:bool=False,binary_valuations:bool=False,category_count=-1)->tuple[Instance,dict,dict,list]:  # todo add randomization for arguments .
    random_num_of_agents = np.random.randint(1, 10 + 1)
    random_num_of_items = np.random.randint(1, 10 + 1)
    random_num_of_categories = category_count if category_count!=-1 else np.random.randint(1, random_num_of_items + 1)
    item_base_value_bounds = (0,1) if binary_valuations else (1,200)
    random_instance = random_uniform_extended(
        num_of_categories=random_num_of_categories,
        num_of_agents=random_num_of_agents, num_of_items=random_num_of_items,
        agent_capacity_bounds=(1, 20), item_capacity_bounds=(1,random_num_of_agents), # changed to match number of agents in worst case ! no need to have extra items ...
        item_base_value_bounds=item_base_value_bounds, item_subjective_ratio_bounds=(0.5, 1.5),
        agent_name_template="Agent{index}", item_name_template="m{index}",
        normalized_sum_of_values=1000, equal_capacities=equal_capacities, equal_valuations=equal_valuations
    )
    return random_instance

def is_fef1(alloc: dict, instance: Instance, item_categories: dict, agent_category_capacities: dict,
            valuations_func: callable) -> bool:
    """
    description: this function checks if the allocation given satisfies the property of F-EF1 (feasible-envy-freeness up to 1 good)
    param alloc: the allocation
    param instance: the instance which belongs to the allocation
    param item_categories: dictionary of categories paired with lists of items
    param agent_category_capacities: dictionary of agents paired with dict of categories paired with values (capacities for each category)
    param valuations_func: this function directs us to the valuation of each (agent,item)
    """
    agent_bundle_value={agent:sum(valuations_func(agent,item) for item in bundle)for agent,bundle in alloc.items()}
    agent_item_value={agent:{item:valuations_func(agent,item) for item in instance.items}for agent in instance.agents}
    print(f'final bundle before check is {alloc}\n valuations are {agent_bundle_value},\n agent-item value {agent_item_value}\n instance items {instance.items}\n agent capacities {agent_category_capacities}\n item categories {item_categories}')
    agent_categorized_allocation = {agent: {category: [] for category in item_categories.keys()} for agent in
                                    agent_category_capacities.keys()}
    #alloc = alloc_random_filler(agent_category_capacities, alloc, item_categories)
    alloc_2d_converter(agent_categorized_allocation, alloc, item_categories)

    for agent_i in agent_categorized_allocation.keys():
        agent_i_bundle_value = sum(
            valuations_func(agent_i, item) for item in alloc[agent_i])  # the value of agent i bundle
        for agent_j in agent_categorized_allocation.keys():
            if agent_i == agent_j:
                continue  # Skip comparing the same agent
            feasible_agent_j_bundle = []
            for cat, cap in agent_category_capacities[agent_i].items():
                temp_j_list = agent_categorized_allocation[agent_j][cat].copy()
                temp_j_list.sort(key=lambda x: valuations_func(agent_i, x))
                first_k_items = temp_j_list[:cap]
                feasible_agent_j_bundle.extend(
                    first_k_items)  # adding all the time the set of feasible items to the final bundle so we can compare it later with agent_i bundle

            feasible_agent_j_bundle_value = sum(valuations_func(agent_i, item) for item in feasible_agent_j_bundle)
            print(f'j bundle is {feasible_agent_j_bundle} feasible j bundle value {feasible_agent_j_bundle_value}\n agent i bundle {alloc[agent_i]} and agent i bundle value {agent_i_bundle_value}')
            if feasible_agent_j_bundle_value > agent_i_bundle_value:
                # Check if removing any one item can make the bundle less or equal
                found=False
                for current_item in feasible_agent_j_bundle:
                    if feasible_agent_j_bundle_value - valuations_func(agent_i, current_item) <= agent_i_bundle_value:
                        found=True
                        print('found !!!')
                        break # found an item that if we remove the envy is gone
                if found==False:
                    print("false !!!!")
                    return False  # If no such item found, return False
    return True


def alloc_2d_converter(agent_categorized_allocation, alloc, item_ctegories):
    """
    in short this function converts normal allocation dictionary to a more precise form of it in which the items are categorized
    in other words dict[string,dict[string,list]]
    :param agent_categorized_allocation: is a 2d dict in which will be the result
    :param alloc: the allocation we have as a result of invoking divide(algorithm=...,**kwargs)
    :param item_ctegories: is a dict[string,list] in which keys are category names and values are lists of items which belong to that category
    """
    for cat in item_ctegories.keys():
        for agent in alloc.keys():
            for item in alloc[agent]:
                if item in item_ctegories[cat]:
                    agent_categorized_allocation[agent][cat].append(item)


def alloc_random_filler(agent_category_capacities, alloc, item_ctegories)->dict:
    """
    a temporary function to fill the empty allocation , used till we implement the allocation algorithms in heterogeneous_matroid_constraints_algorithms.py
    """
    agent_capacities_copy = agent_category_capacities.copy()
    alloc = {agent: [] for agent in agent_category_capacities.keys()}  # initializing
    for category, item_list in item_ctegories.items():
        for item in item_list:
            random_agent = np.random.choice(list(agent_category_capacities.keys()))
            agent_capacities_copy[random_agent][category] -= 1
            while agent_capacities_copy[random_agent][category] < 0:
                random_agent = np.random.choice(list(agent_category_capacities.keys()))
                agent_capacities_copy[random_agent][category] -= 1
            alloc[random_agent].append(item)
    return alloc


def test_algorithm_1():
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=True)
    #print(f"instance -> {instance},\n agent_capacities -> {agent_capacities_2d},\n categories -> {categories},\n order ->  {order}")
    assert is_fef1(divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_round_robin, instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d,initial_agent_order=order),
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


def test_algorithm_2():
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False,category_count=1)
    alloc=divide(algorithm=heterogeneous_matroid_constraints_algorithms.capped_round_robin, instance=instance,
           item_categories=categories, agent_category_capacities=agent_capacities_2d, initial_agent_order=order)
    #print(alloc)
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


def test_algorithm_3():
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False ,category_count=2)
    alloc=divide(algorithm=heterogeneous_matroid_constraints_algorithms.two_categories_capped_round_robin, instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d, initial_agent_order=order)
    #print(alloc)
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


def test_algorithm_4(): # TODO equal_valuations=True
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False,equal_valuations=True)
    # print(
    #     f"instance -> {instance},\n agent_capacities -> {agent_capacities_2d},\n categories -> {categories},\n order ->  {order}")
    alloc=divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_capped_round_robin, instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d, initial_agent_order=order)
    #print(alloc)
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True #check the is_fef1 function


@pytest.mark.parametrize("run", range(1000))  # Run the test 10 times
def test_algorithm_5(run):  # binary valuations # TODO force it to create instance witn no cyclces in envy graph kind of weird since in binary vals no envy cycle can be imagined
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False,binary_valuations=True)
    alloc=divide(algorithm=heterogeneous_matroid_constraints_algorithms.iterated_priority_matching,
                          instance=instance,
                          item_categories=categories, agent_category_capacities=agent_capacities_2d)
    #print(alloc)
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True

