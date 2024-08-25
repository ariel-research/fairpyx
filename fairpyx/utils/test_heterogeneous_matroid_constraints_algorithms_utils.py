import logging

import numpy as np

from fairpyx import Instance

logger = logging.getLogger(__name__)


def random_instance(equal_capacities: bool = False, equal_valuations: bool = False, binary_valuations: bool = False,
                    category_count=-1, num_of_agents=-1, num_of_items=-1, item_capacity_bounds=(-1, -1),
                    random_seed_num: int = -1,agent_capacity_bounds:tuple[int,int]=(-1,-1)) -> tuple[Instance, dict, dict, list]:
    random_seed_num = np.random.randint(1, 2 ** 31) if random_seed_num == -1 else random_seed_num
    np.random.seed(random_seed_num)
    random_num_of_agents = np.random.randint(1, 10 + 1) if num_of_agents == -1 else num_of_agents  #✅
    random_num_of_items = np.random.randint(1, 10 + 1) if num_of_items == -1 else num_of_items  #✅
    num_of_categories = category_count if category_count != -1 else np.random.randint(1, random_num_of_items + 1)  #✅
    item_base_value_bounds = (0, 1) if binary_valuations else (1, 100)  #✅
    random_instance = random_uniform_extended(
        num_of_categories=num_of_categories,  # in case we state its not random
        num_of_agents=random_num_of_agents,
        num_of_items=random_num_of_items,
        agent_capacity_bounds=(1, 20) if agent_capacity_bounds == (-1, -1) else agent_capacity_bounds,
        item_capacity_bounds=(1, random_num_of_agents) if item_capacity_bounds == (-1, -1) else item_capacity_bounds,
        item_base_value_bounds=item_base_value_bounds,
        item_subjective_ratio_bounds=(1, 2),
        agent_name_template="Agent{index}",
        item_name_template="m{index}",
        normalized_sum_of_values=100,
        equal_capacities=equal_capacities,  # True in case flagged
        equal_valuations=equal_valuations,  # True in case flagged
        random_seed=random_seed_num
    )
    return random_instance


def random_uniform(num_of_agents: int, num_of_items: int,
                   agent_capacity_bounds: tuple[int, int],
                   item_capacity_bounds: tuple[int, int],
                   item_base_value_bounds: tuple[int, int],
                   item_subjective_ratio_bounds: tuple[float, float],
                   normalized_sum_of_values: int,
                   agent_name_template="s{index}", item_name_template="c{index}",
                   random_seed: int = None,
                   ):
    """
    Generate a random instance by drawing values from uniform distributions.
    """
    if random_seed is None:
        random_seed = np.random.randint(1, 2 ** 31)
    logger.info("Random seed: %d", random_seed)
    agents = [agent_name_template.format(index=i + 1) for i in range(num_of_agents)]
    items = [item_name_template.format(index=i + 1) for i in range(num_of_items)]
    agent_capacities = {agent: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for agent
                        in agents}
    item_capacities = {item: np.random.randint(item_capacity_bounds[0], item_capacity_bounds[1] + 1) for item in
                       items}
    base_values = normalized_valuation(random_valuation(num_of_items, item_base_value_bounds),
                                       normalized_sum_of_values)
    valuations = {
        agent: dict(zip(items, normalized_valuation(
            base_values * random_valuation(num_of_items, item_subjective_ratio_bounds),
            normalized_sum_of_values
        )))
        for agent in agents
    }
    return Instance(valuations=valuations, agent_capacities=agent_capacities, item_capacities=item_capacities)


def random_valuation(numitems: int, item_value_bounds: tuple[float, float]) -> np.ndarray:
    """
    >>> r = random_valuation(10, [30, 40])
    >>> len(r)
    10
    >>> all(r>=30)
    True
    """
    return np.random.uniform(low=item_value_bounds[0], high=item_value_bounds[1] + 1, size=numitems)


def normalized_valuation(raw_valuations: np.ndarray, normalized_sum_of_values: float):
    epsilon = 1e-10
    raw_sum_of_values = sum(raw_valuations) if sum(raw_valuations) != 0 else epsilon # not going to affect since values are 0 and k*0=0
    return np.round(raw_valuations * normalized_sum_of_values / raw_sum_of_values).astype(int)


def random_uniform_extended(num_of_agents: int, num_of_items: int,
                            num_of_categories: int,
                            agent_capacity_bounds: tuple[int, int],
                            item_capacity_bounds: tuple[int, int],
                            item_base_value_bounds: tuple[int, int],
                            item_subjective_ratio_bounds: tuple[float, float],
                            normalized_sum_of_values: int,
                            random_seed: int,
                            agent_name_template="Agent{index}", item_name_template="m{index}",
                            equal_capacities: bool = False
                            , equal_valuations: bool = False
                            ) -> tuple[Instance, dict, dict, list]:
    #logger.info(f'{np.random.randint(100)}') works fine gives same number !
    result_instance = random_uniform(num_of_agents=num_of_agents,
                                     num_of_items=num_of_items,
                                     agent_capacity_bounds=agent_capacity_bounds,
                                     item_capacity_bounds=item_capacity_bounds,
                                     item_base_value_bounds=item_base_value_bounds,
                                     item_subjective_ratio_bounds=item_subjective_ratio_bounds,
                                     normalized_sum_of_values=normalized_sum_of_values,
                                     agent_name_template=agent_name_template,
                                     item_name_template=item_name_template,
                                     random_seed=random_seed)
    initial_agent_order = [agent for agent in result_instance.agents]
    np.random.shuffle(initial_agent_order)  # randomizing initial_agent_order#✅

    category_string_template = "c{cat}"
    categories = {category_string_template.format(cat=cat + 1): [] for cat in range(num_of_categories)}
    # check whether equal capacities or doesn't matter
    if not equal_capacities:
        agent_category_capacities = {
            agent: {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for category in
                    categories} for agent
            in result_instance.agents}
    else:  # equal capacity doesnt require rand each time
        category_capacities = {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for
                               category in categories}  # this is going to be assigned to each agent
        agent_category_capacities = {agent: category_capacities for agent in result_instance.agents}  #✅
    if equal_valuations:
        random_valuation = np.random.randint(low=item_base_value_bounds[0], high=item_base_value_bounds[1] + 1,
                                             size=num_of_items)
        print(f"random valuation is -> {random_valuation}")
        # we need to normalize is
        sum_of_valuations = np.sum(random_valuation)
        normalized_sum_of_values=sum_of_valuations if item_base_value_bounds == (0, 1) else normalized_sum_of_values
        if item_base_value_bounds != (0, 1):
            normalized_random_values = np.round(random_valuation * normalized_sum_of_values / sum_of_valuations).astype(
                int) # change to float if needed
        else:#binary valuations , no need for normalization ! since we are constrained
            normalized_random_values = np.round(random_valuation).astype(
                int)  # change to float if needed

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
    for category in categories.keys():  # start with categories first so we make sure there is no empty categories
        if len(temporary_items) > 0:
            random_item = np.random.choice(temporary_items)
            categories[category].append(random_item)
            temporary_items.remove(random_item)
        else:
            break
    for item in temporary_items:  # random allocation of items to categories
        random_category = np.random.choice(list(categories.keys()))
        categories[random_category].append(item)

    result_instance.agent_capacity = {
        # this is where we make capacity as sum of all category capacities per each agent !(useful to prevent algorithm early termination)
        agent: sum(agent_category_capacities[agent][category] for category in agent_category_capacities[agent]) for
        agent in agent_category_capacities.keys()}

    item_capacities = {item: result_instance.item_capacity(item) for item in result_instance.items}

    if item_base_value_bounds == (0, 1):  # in case of binary valuations (Algorithm5)
        if equal_valuations:
            pass
        else:
            normalized_random_agent_item_valuation = {
                agent: dict(zip(result_instance.items, np.random.choice([0, 1], size=len(result_instance.items))))
                for agent in result_instance.agents
            }

    result_instance = Instance(valuations=normalized_random_agent_item_valuation, item_capacities=item_capacities,
                               agent_capacities=result_instance.agent_capacity)
    return result_instance, agent_category_capacities, categories, initial_agent_order  #✅#✅#✅


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
                    agent_categorized_allocation[agent][cat].append(
                        item)  # filtering each item to the category they belong to


def is_fef1(alloc: dict, instance: Instance, item_categories: dict, agent_category_capacities: dict,
            valuations_func: callable) -> bool:
    agent_categorized_allocation = {agent: {category: [] for category in item_categories.keys()} for agent in
                                    agent_category_capacities.keys()}
    agent_categorized_allocation_builder(agent_categorized_allocation, alloc,
                                         item_categories)  # Fill the items according to the category they belong to
    logger.info(
        f"categorized allocation is -> {agent_categorized_allocation} \n agent categorized capacities are -> {agent_category_capacities} \n categorize items  are {item_categories}")
    for agent_i in agent_categorized_allocation.keys():
        agent_i_bundle_value = sum(
            valuations_func(agent_i, item) for item in alloc[agent_i])  # The value of agent i bundle
        logger.info(f"Checking agent {agent_i}, bundle value {agent_i_bundle_value}")

        for agent_j in agent_categorized_allocation.keys():
            if agent_i == agent_j:
                continue  # Skip comparing the same agent

            feasible_agent_j_bundle = []
            for cat, cap in agent_category_capacities[agent_i].items():
                temp_j_alloc = agent_categorized_allocation[agent_j][cat].copy()  # Sub_allocation per category
                temp_j_alloc.sort(key=lambda x: valuations_func(agent_i, x),
                                  reverse=True)  # Sort based on values descending order
                first_k_items = temp_j_alloc[:cap]
                feasible_agent_j_bundle.extend(
                    first_k_items)  # Adding all the time the set of feasible items to the final bundle

            feasible_agent_j_bundle_value = sum(valuations_func(agent_i, item) for item in
                                                feasible_agent_j_bundle)  # The value of the best feasible bundle in the eyes of agent i
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
                    logger.error(
                        f"Failed F-EF1 check for agent {agent_i} with bundle {alloc[agent_i]} against agent {agent_j} with bundle {feasible_agent_j_bundle}")
                    return False  # If no such item found, return False
    return True  # Whoever reaches here means there is no envy
