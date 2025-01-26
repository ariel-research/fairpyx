"""
An implementation of the Pytest of the algorithms  in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2010), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import logging

import pytest
from fairpyx.algorithms import heterogeneous_matroid_constraints_algorithms
from fairpyx import divide
from fairpyx.utils.test_heterogeneous_matroid_constraints_algorithms_utils import random_instance, is_fef1

logger = logging.getLogger(__name__)

TIMES_TO_RUN = 1

@pytest.mark.parametrize("run", range(TIMES_TO_RUN))  # Run the test 10 times
def test_algorithm_1(run):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(
        equal_capacities=True, num_of_agents=4, num_of_items=30, random_seed_num=0)  #,item_capacity_bounds=(1,1)#since we're doing cycle elemination
    logger.info(f'TEST NUMBER {run}')
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_round_robin,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)

    logger.info(f"allocation is ------------->: {alloc}")
    #print(f"instance -> {instance},\n agent_capacities -> {agent_category_capacities},\n categories -> {categories},\n initial_agent_order ->  {initial_agent_order}")
    assert is_fef1(alloc,
                   instance=instance,
                   agent_category_capacities=agent_category_capacities,
                   item_categories=categories,
                   valuations_func=instance.agent_item_value) is True

    logger.info("Finished processing data")


@pytest.mark.parametrize("run", range(TIMES_TO_RUN))  # Run the test 10 times
def test_algorithm_2(run):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False,
                                                                                           category_count=1,
                                                                                           item_capacity_bounds=(1, 1),
                                                                                           random_seed_num=0)
    # logger.info(f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.capped_round_robin, instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order,target_category='c1')
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


@pytest.mark.parametrize("run", range(TIMES_TO_RUN))  # Run the test 10 times
def test_algorithm_3(run):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False,
                                                                                           category_count=2,
                                                                                           item_capacity_bounds=(1, 1),
                                                                                           random_seed_num=0)
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")

    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.two_categories_capped_round_robin,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order,target_category_pair=('c1', 'c2'))
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True


@pytest.mark.parametrize("run", range(TIMES_TO_RUN))  # Run the test 10 times
def test_algorithm_4(run):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False,
                                                                                           equal_valuations=True,
                                                                                           random_seed_num=0)
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.per_category_capped_round_robin,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True  #check the is_fef1 function


@pytest.mark.parametrize("run", range(TIMES_TO_RUN))  # Run the test 10 times
def test_algorithm_5(
        run):  # binary valuations
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False,
                                                                                           binary_valuations=True,
                                                                                           item_capacity_bounds=(1, 1),
                                                                                           random_seed_num=0)
    alloc = divide(algorithm=heterogeneous_matroid_constraints_algorithms.iterated_priority_matching,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities)
    logger.info(
        f"Starting to process data: {instance} \n categories are -> {categories} \n initial_agent_order is -> {initial_agent_order} \n -> agent_category_capacities are -> {agent_category_capacities}\n *********************************************************************************** ")
    logger.info(f"allocation is ------------->: {alloc}")
    assert is_fef1(alloc,
                   instance=instance
                   , agent_category_capacities=agent_category_capacities, item_categories=categories,
                   valuations_func=instance.agent_item_value) is True
    


if __name__ == "__main__":
    pytest.main(["-v", __file__])
