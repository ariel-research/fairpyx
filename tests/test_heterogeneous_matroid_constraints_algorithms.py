"""
An implementation of the Pytest of the algorithms  in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2010), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import logging
import random

import pytest
from fairpyx.instances import Instance
from fairpyx.algorithms import heterogeneous_matroid_constraints_algorithms
from fairpyx import divide
import numpy as np

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler=logging.FileHandler('test_heterogeneous_matroid_constraints_algorithms.log',mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# TODO lets do reconstruction here tomorrow ! , less functions ! beautify ! straight to point !

def random_instance(equal_capacities: bool = False, equal_valuations: bool = False, binary_valuations: bool = False,
                    category_count=-1,num_of_agents=-1,num_of_items=-1,item_capacity_bounds=(-1,-1)) -> tuple[Instance, dict, dict, list]:
    random_num_of_agents = np.random.randint(1, 10+1) if num_of_agents == -1 else num_of_agents #✅
    random_num_of_items = np.random.randint(1, 10+1) if num_of_items == -1 else num_of_items  #✅
    num_of_categories = category_count if category_count != -1 else np.random.randint(1, random_num_of_items+1)  #✅
    item_base_value_bounds = (0, 1) if binary_valuations else (1, 100)  #✅
    random_instance = random_uniform_extended(
        num_of_categories=num_of_categories,  # in case we state its not random
        num_of_agents=random_num_of_agents,
        num_of_items=random_num_of_items,
        agent_capacity_bounds=(1, 20),
        item_capacity_bounds=(1, random_num_of_agents) if item_capacity_bounds == (-1,-1) else item_capacity_bounds,
        item_base_value_bounds=item_base_value_bounds,
        item_subjective_ratio_bounds=(0.5, 1.5),
        agent_name_template="Agent{index}",
        item_name_template="m{index}",
        normalized_sum_of_values=100,
        equal_capacities=equal_capacities,  # True in case flagged
        equal_valuations=equal_valuations  # True in case flagged
    )
    return random_instance


def random_uniform_extended(num_of_agents: int, num_of_items: int,
                            num_of_categories: int,
                            agent_capacity_bounds: tuple[int, int],
                            item_capacity_bounds: tuple[int, int],
                            item_base_value_bounds: tuple[int, int],
                            item_subjective_ratio_bounds: tuple[float, float],
                            normalized_sum_of_values: int,
                            agent_name_template="Agent{index}", item_name_template="m{index}",
                            random_seed: int = 2136942426,
                            equal_capacities: bool = False
                            , equal_valuations: bool = False
                            ) -> tuple[Instance, dict, dict, list]:
    np.random.seed(random_seed)
    result_instance = Instance.random_uniform(num_of_agents=num_of_agents,
                                              num_of_items=num_of_items,
                                              agent_capacity_bounds=agent_capacity_bounds,
                                              item_capacity_bounds=item_capacity_bounds,
                                              item_base_value_bounds=item_base_value_bounds, # TODO valuation doesnt do me justice too , no respect for equal valuations
                                              item_subjective_ratio_bounds=item_subjective_ratio_bounds, # TODO understand this
                                              normalized_sum_of_values=normalized_sum_of_values,
                                              agent_name_template=agent_name_template,
                                              item_name_template=item_name_template,
                                              random_seed=random_seed)

    initial_agent_order = [agent for agent in result_instance.agents]
    random.shuffle(initial_agent_order) # randomizing initial_agent_order#✅

    category_string_template = "c{cat}"
    categories = {category_string_template.format(cat=cat+1): [] for cat in range(num_of_categories)}
    # check whether equal capacities or doesn't matter
    if not equal_capacities:
        agent_category_capacities = {
            agent: {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1]+1) for category in
                    categories} for agent
            in result_instance.agents}
    else: # equal capacity doesnt require rand each time
        category_capacities = {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1]+1) for
                               category in categories}# this is going to be assigned to each agent
        agent_category_capacities = {agent:category_capacities for agent in result_instance.agents}#✅

    if equal_valuations:
        random_valuation = np.random.uniform(low=item_base_value_bounds[0], high=item_base_value_bounds[1] + 1,
                                             size=num_of_items)
        # we need to normalize is
        sum_of_valuations = np.sum(random_valuation)
        normalized_random_values = np.round(random_valuation * normalized_sum_of_values / sum_of_valuations).astype(
            int)  # TODO change to float if needed

        normalized_random_agent_item_valuation = {
            agent: dict(zip(result_instance.items, normalized_random_values
                            ))
            for agent in result_instance.agents
        }
    else:  # means the valuations aren't supposed to be equal for every agent
        normalized_random_agent_item_valuation = {
            agent: {item: result_instance.agent_item_value(agent, item) for item in result_instance.items} for agent in
            result_instance.agents}  # we simply use what result_instance has (since its private we extract it)

    temporary_items = list(result_instance.items).copy()
    logger.info(f"categories are -> {categories} and items are -> {temporary_items}")
    for category in categories.keys():# start with categories first so we make sure there is no empty categories
        random_item=np.random.choice(temporary_items)
        categories[category].append(random_item)
        temporary_items.remove(random_item)
    for item in temporary_items:# random allocation of items to categories
        random_category = np.random.choice(list(categories.keys()))
        categories[random_category].append(item)

    result_instance.agent_capacity = { # TODO this is where we make capacity as sum of all category capacities per each agent !(useful to prevent algorithm early termination)
        agent: sum(agent_category_capacities[agent][category] for category in agent_category_capacities[agent]) for agent in agent_category_capacities.keys()}

    item_capacities = {item: result_instance.item_capacity(item) for item in result_instance.items}
    #result_instance.item_capacity # TODO looks like a smoother approach to pass a callable since its also public ! for future use if required

    if item_base_value_bounds == (0, 1):# in case of binary valuations (Algorithm5)
        normalized_random_agent_item_valuation = {
            agent: dict(zip(result_instance.items, np.random.choice([0, 1], size=len(result_instance.items))))
            for agent in result_instance.agents
        }

    result_instance = Instance(valuations=normalized_random_agent_item_valuation, item_capacities=item_capacities,
                               agent_capacities=result_instance.agent_capacity)
    return result_instance, agent_category_capacities, categories, initial_agent_order#✅#✅#✅


def agent_categorized_allocation_builder(agent_categorized_allocation, alloc, categories):
    """
    in short this function converts normal allocation dictionary to a more precise form of it in which the items are categorized
    in other words dict[string,dict[string,list]]
    :param agent_categorized_allocation: is a 2d dict in which will be the result
    :param alloc: the allocation we have as a result of invoking divide(algorithm=...,**kwargs)
    :param categories: is a dict[string,list] in which keys are category names and values are lists of items which belong to that category
    """
    for cat in categories.keys():
        for agent in alloc.keys():
            for item in alloc[agent]:
                if item in categories[cat]:
                    agent_categorized_allocation[agent][cat].append(item) # filtering each item to the category they belong to

@pytest.mark.parametrize("run", range(100))  # Run the test 10 times
def test_algorithm_1(run):

    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=True,num_of_agents=4,num_of_items=30,item_capacity_bounds=(1,1)) #since we're doing cycle elemination
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    alloc=divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_round_robin,
                          instance=instance,
                          item_categories=categories,
                          agent_category_capacities=agent_category_capacities,
                          initial_agent_order=initial_agent_order)

    logger.info(f"allocation is ------------->: {alloc}")
    print(f"instance -> {instance},\n agent_capacities -> {agent_category_capacities},\n categories -> {categories},\n initial_agent_order ->  {initial_agent_order}")
    assert is_fef1(alloc,
                          instance=instance,
                          agent_category_capacities=agent_category_capacities,
                          item_categories=categories,
                          valuations_func=instance.agent_item_value) is True

    logger.info("Finished processing data")


@pytest.mark.parametrize("run", range(100))  # Run the test 10 times
def test_algorithm_2(run):# TODO show Erel the video of the problem (the item caps affecting fairness) (no wonder each algorithm using RR is gonna be failing in some instances where item capacity affects the situation)
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False, category_count=1,item_capacity_bounds=(1,1))
    # logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.capped_round_robin, instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities, initial_agent_order=initial_agent_order)
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


@pytest.mark.parametrize("run", range(100))  # Run the test 10 times
def test_algorithm_3(run):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False, category_count=2,num_of_items=3,item_capacity_bounds=(1,1)) # TODO somehow it runs stupid with only 1 item !!! WHYYYYYYY
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")

    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.two_categories_capped_round_robin,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities, initial_agent_order=initial_agent_order)
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


@pytest.mark.parametrize("run", range(100))  # Run the test 10 times
def test_algorithm_4(run):

    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False, equal_valuations=True)
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_capped_round_robin,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_capacities_2d, initial_agent_order=order)
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True  #check the is_fef1 function


@pytest.mark.parametrize("run", range(100))  # Run the test 10 times
def test_algorithm_5(
        run):  # binary valuations # TODO force it to create instance witn no cyclces in envy graph kind of weird since in binary vals no envy cycle can be imagined
    instance, agent_capacities_2d, categories, order = random_instance(equal_capacities=False, binary_valuations=True,item_capacity_bounds=(1,1))
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.iterated_priority_matching,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_capacities_2d)
    logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_capacities_2d, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


"""
            INSTANCE 
  Represents an instance of the fair course-allocation problem.
    Exposes the following functions:
     * agent_capacity:       maps an agent name/index to its capacity (num of seats required). {'Agent1':10,......'Agentk':10} -> we need to make sure each agent gets the sum of capacities for each capacity[agent][category]
     * item_capacity:        maps an item  name/index to its capacity (num of seats allocated). {'m1':10,......} ✅✅✅✅✅
     * agent_conflicts:      maps an agent  name/index to a set of items that conflict with it (- cannot be allocated to this agent). {'Agent1':{'m1',.....'mk'}} ✅✅✅✅✅ handled in alloc.give()
     * item_conflicts:       maps an item  name/index to a set of items that conflict with it (- cannot be allocated together). ❌❌❌❌❌❌❌❌ misconception we dont support this in our case , CHECK USAGES AND TELL IT DOESNT AFFECT
     * agent_item_value:     maps an agent,item pair to the agent's value for the item. ✅✅✅✅✅ callable , supported ✅✅✅
     * agents: an enumeration of the agents (derived from agent_capacity).
     * items: an enumeration of the items (derived from item_capacity).
     
            ALLOCATIONBUILDER 
    A class for incrementally constructing an allocation.

    Whenever an item is given to an agent (via the 'give' method),
    the class automatically updates the 'remaining_item_capacities' and the 'remaining_agent_capacities'.
    It also updates the 'remaining_conflicts' by adding a conflict between the agent and the item, so that it is not assigned to it anymore.

    Once you finish adding items, use "sorted" to get the final allocation (where each bundle is sorted alphabetically).
    self.instance = instance
        self.remaining_agent_capacities = {agent: instance.agent_capacity(agent) for agent in instance.agents if instance.agent_capacity(agent) > 0}
        self.remaining_item_capacities = {item: instance.item_capacity(item) for item in instance.items if instance.item_capacity(item) > 0}
        self.remaining_conflicts = {(agent,item) for agent in self.remaining_agents() for item in self.instance.agent_conflicts(agent)}
        self.bundles = {agent: set() for agent in instance.agents}    # Each bundle is a set, since each agent can get at most one seat
        
        
        ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
        modifications 
        1) we must also have a container mapping agent to container of categories mapped to capacities (agent_category_capacities) {'Agent1':{'c1':10,'c2':5}}
        2) (category_items) mapping each category to items {'c1':['m1',......'mk']}
        ❌❌❌❌❌❌❌❌ item_conflicts must not be used ! , we dont support it 
        ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
        in our papers we have : 
        1) numerical/binary valuations -> make sure to make it an option in the instance generator 
        2) 1/2/k>2  categories -> make sure to make it an option in the instance generator
        3) identical/different valuations -> make sure to make it an option in the instance generator
        4) categorizations is shared between all ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌ 
        NO CHANCE SOMEONE SEES ITEMS DIVIDED IN DIFFERENT FORM AT LEAST TILL ALGORITHM 5 ❌❌❌❌❌❌❌
        
        
     """
def is_fef1(alloc: dict, instance: Instance, item_categories: dict, agent_category_capacities: dict,
            valuations_func: callable) -> bool:
    agent_categorized_allocation = {agent: {category: [] for category in item_categories.keys()} for agent in
                                    agent_category_capacities.keys()}
    agent_categorized_allocation_builder(agent_categorized_allocation, alloc, item_categories)  # Fill the items according to the category they belong to
    logger.info(f"categorized allocation is -> {agent_categorized_allocation} \n agent categorized capacities are -> {agent_category_capacities} \n categorize items  are {item_categories}")
    for agent_i in agent_categorized_allocation.keys():
        agent_i_bundle_value = sum(valuations_func(agent_i, item) for item in alloc[agent_i])  # The value of agent i bundle
        logger.info(f"Checking agent {agent_i}, bundle value {agent_i_bundle_value}")

        for agent_j in agent_categorized_allocation.keys():
            if agent_i == agent_j:
                continue  # Skip comparing the same agent

            feasible_agent_j_bundle = []
            for cat, cap in agent_category_capacities[agent_i].items():
                temp_j_alloc = agent_categorized_allocation[agent_j][cat].copy()  # Sub_allocation per category
                temp_j_alloc.sort(key=lambda x: valuations_func(agent_i, x), reverse=True)  # Sort based on values descending order
                first_k_items = temp_j_alloc[:cap]
                feasible_agent_j_bundle.extend(first_k_items)  # Adding all the time the set of feasible items to the final bundle

            feasible_agent_j_bundle_value = sum(valuations_func(agent_i, item) for item in feasible_agent_j_bundle)  # The value of the best feasible bundle in the eyes of agent i
            logger.info(f"Agent {agent_i} vs Agent {agent_j}, feasible bundle value {feasible_agent_j_bundle_value}")

            if feasible_agent_j_bundle_value > agent_i_bundle_value:
                # Check if removing any one item can make the bundle less or equal
                found = False  # Helping flag
                for current_item in feasible_agent_j_bundle:
                    if feasible_agent_j_bundle_value - valuations_func(agent_i, current_item) <= agent_i_bundle_value:
                        found = True
                        logger.info(f"Removing item {current_item} resolves envy")
                        break  # Found an item that if we remove the envy is gone
                if not found:
                    logger.error(f"Failed F-EF1 check for agent {agent_i} with bundle {alloc[agent_i]} against agent {agent_j} with bundle {feasible_agent_j_bundle}")
                    return False  # If no such item found, return False
    return True  # Whoever reaches here means there is no envy

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #pytest.main(["-s", __file__])
    pass