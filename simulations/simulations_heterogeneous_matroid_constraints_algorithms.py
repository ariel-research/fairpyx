from tests.test_heterogeneous_matroid_constraints_algorithms import random_instance
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import *
import experiments_csv
import logging

experiments_csv.logger.setLevel(logging.INFO)
#Experiment 1 : single  category friendly ->>[per_category_round_robin,capped_round_robin,per_category_capped_round_robin]
expr = experiments_csv.Experiment('results/', 'experiment1.csv')
input_ranges = {
    'equal_capacities': [True],
    'category_count': [1],
    'item_capacity_bounds': [(1, 1)],
    'random_seed_num': [0],
    'algorithm': [per_category_round_robin, capped_round_robin,
                  per_category_capped_round_robin]
}


def experiment1(equal_capacities: bool, category_count: int, item_capacity_bounds: any, random_seed_num: int,
                algorithm: callable):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(
        equal_capacities=equal_capacities,
        category_count=category_count,
        item_capacity_bounds=item_capacity_bounds, random_seed_num=random_seed_num)
    alloc = AllocationBuilder(instance)
    kawrgs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order}
    algorithm(**kawrgs)
    return {'allocation': alloc.bundles}


expr.run_with_time_limit(experiment1,input_ranges,time_limit=60)
experiments_csv.single_plot_results('results/experiment1.csv',filter={},x_field='category_count',y_field='runtime',z_field='algorithm',save_to_file='results/photos/experiment1.png')


# Experiment 2 : many cateogry friendly(minimum 2 , assuming algo 2 handles it)  ->> [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin]
expr = experiments_csv.Experiment('results/', 'experiment2.csv')
input_ranges = {
    'equal_capacities': [True,False],
    'category_count': range(2,20),
    'item_capacity_bounds': [(1, 100)],
    'random_seed_num': [0],
    'algorithm': [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin]
}


def experiment2(equal_capacities: bool, category_count: int, item_capacity_bounds: any, random_seed_num: int,
                algorithm: callable):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(
        equal_capacities=equal_capacities,
        category_count=category_count,
        item_capacity_bounds=item_capacity_bounds, random_seed_num=random_seed_num)
    alloc = AllocationBuilder(instance)
    kawrgs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order}
    algorithm(**kawrgs)
    return {'allocation': alloc.bundles}

expr.run_with_time_limit(experiment2,input_ranges,time_limit=60)
experiments_csv.single_plot_results('results/experiment2.csv',filter={},x_field='category_count',y_field='runtime',z_field='algorithm',save_to_file='results/photos/experiment2.png')


# Experiment 3 : binary valuation friendly(single category)->> [per_category_round_robin,capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
expr = experiments_csv.Experiment('results/', 'experiment3.csv')
input_ranges = {
    'equal_capacities': [True,False],
    'category_count': range(1,1),
    'item_capacity_bounds': [(1, 100)],
    'random_seed_num': [0],
    'binary_valuations':[True],
    'algorithm': [per_category_round_robin,capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
}


def experiment3(equal_capacities: bool, category_count: int, item_capacity_bounds: any, random_seed_num: int,
                algorithm: callable,binary_valuations:bool):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance( binary_valuations=binary_valuations,
        equal_capacities=equal_capacities,
        category_count=category_count,
        item_capacity_bounds=item_capacity_bounds, random_seed_num=random_seed_num)
    alloc = AllocationBuilder(instance)
    kawrgs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order} if algorithm.__name__ != 'iterated_priority_matching' else {
        'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
        }
    algorithm(**kawrgs)
    return {'allocation': alloc.bundles}

expr.run_with_time_limit(experiment3,input_ranges,time_limit=60)
experiments_csv.single_plot_results('results/experiment3.csv',filter={},x_field='category_count',y_field='runtime',z_field='algorithm',save_to_file='results/photos/experiment3.png')


## Experiment 4 : binary valuation friendly(many categories)->> [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
expr = experiments_csv.Experiment('results/', 'experiment4.csv')
input_ranges = {
    'equal_capacities': [True,False],
    'category_count': range(10,20),
    'item_capacity_bounds': [(1, 100)],
    'random_seed_num': [0],
    'binary_valuations':[True],
    'algorithm': [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
}
def experiment4(equal_capacities: bool, category_count: int, item_capacity_bounds: any, random_seed_num: int,
                algorithm: callable,binary_valuations:bool):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance( binary_valuations=binary_valuations,
        equal_capacities=equal_capacities,
        category_count=category_count,
        item_capacity_bounds=item_capacity_bounds, random_seed_num=random_seed_num)
    alloc = AllocationBuilder(instance)
    kawrgs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order} if algorithm.__name__!='iterated_priority_matching' else {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              }
    algorithm(**kawrgs)
    return {'allocation': alloc.bundles}

expr.run_with_time_limit(experiment4,input_ranges,time_limit=60)
experiments_csv.single_plot_results('results/experiment4.csv',filter={},x_field='category_count',y_field='runtime',z_field='algorithm',save_to_file='results/photos/experiment4.png')





















# algo 1 : different number of categories , same capacity
# algo 2 : 1 category , different capacities
# algo 3 : 2 categories
# algo 4 : many categories , differnt capacities
# algo 5 : many categories , diffeent capacities , binary valuations
#all - > [per_category_round_robin,capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
# Experiment 1 : single  category friendly ->>[per_category_round_robin,capped_round_robin,per_category_capped_round_robin]
# Experiment 2 : many cateogry friendly(minimum 2 , assuming algo 2 handles it)  ->> [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin]
# Experiment 3 : binary valuation friendly(single category)->> [per_category_round_robin,capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
# Experiment 4 : binary valuation friendly(many categories)->> [capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
