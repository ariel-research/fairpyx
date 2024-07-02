import experiments_csv
import logging

from fairpyx import rounded_allocation
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import *
from fairpyx.utils.test_utils import stringify
from tests.test_heterogeneous_matroid_constraints_algorithms import *
from fairpyx.algorithms.fractional_egalitarian import *

def compare_heterogeneous_matroid_constraints_algorithms_egalitarian_utilitarian():  #egalitarian: prioritizes the poor
    """
    we have 5 algorithms , for the sake of common support we give an input range respected by all algorithms
    meaning : single category , binary valuations , equal valuations , equal capacities
    we make number of items/number of agents variable
    """
    expr=experiments_csv.Experiment('results/', 'egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms.csv')
    #expr.clear_previous_results()
    input_ranges_1 = {
        'equal_capacities': [True],
        'equal_valuations': [True],
        'binary_valuations': [True],
        'num_of_items':range(10,20),
        'category_count': [2],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': [0],
        'num_of_agents': range(10,20),
        'algorithm': [per_category_round_robin, capped_round_robin,
                      per_category_capped_round_robin, two_categories_capped_round_robin, iterated_priority_matching]
    }
    input_ranges_2 = {
        'equal_capacities': [True],
        'equal_valuations': [True,False],
        'binary_valuations': [False],
        'num_of_items': range(10,20),
        'category_count': [2],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': [0],
        'num_of_agents': range(10,20),
        'algorithm': [per_category_round_robin, capped_round_robin
                      , two_categories_capped_round_robin]
    } # equal capacities for the sake of the compatibility of input with the implemented egalitarian and utilitarian algorithms
    # we also need to consider giving each agent a capacity which is >= number of items
    input_ranges_3 = {
        'equal_capacities': [True],
        'equal_valuations': [True],
        'binary_valuations': [False],
        'num_of_items': range(10,20),
        'category_count': [2],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': [0],
        'num_of_agents': range(10,20),
        'algorithm': [
                      per_category_capped_round_robin]
    }
    expr.run_with_time_limit(run_experiment,input_ranges_1,5)
    expr.run_with_time_limit(run_experiment, input_ranges_2, 5)
    expr.run_with_time_limit(run_experiment, input_ranges_3, 5)
def run_experiment(equal_capacities:bool,equal_valuations:bool,binary_valuations:bool,category_count:int,item_capacity_bounds:int,random_seed_num:int,num_of_agents:int,algorithm:callable,num_of_items:int):
    # Mapping of algorithms to their specific argument sets
    algo_args = {
        per_category_round_robin: {'alloc', 'agent_category_capacities', 'item_categories', 'initial_agent_order'},
        capped_round_robin: {'alloc', 'item_categories', 'agent_category_capacities',
                             'initial_agent_order', 'target_category'},
        two_categories_capped_round_robin: {'alloc', 'item_categories', 'agent_category_capacities', 'initial_agent_order','target_category_pair'},
        per_category_capped_round_robin: {'alloc', 'item_categories', 'agent_category_capacities', 'initial_agent_order'},
        iterated_priority_matching: {'alloc', 'item_categories', 'agent_category_capacities'},
    }

    instance, agent_category_capacities, categories, initial_agent_order = random_instance(
        equal_capacities=equal_capacities,
        equal_valuations=equal_valuations,
        binary_valuations=binary_valuations,
        category_count=category_count,
        item_capacity_bounds=(1, item_capacity_bounds), random_seed_num=random_seed_num, num_of_agents=num_of_agents,num_of_items=num_of_items,agent_capacity_bounds=(num_of_items,num_of_items+1))
    alloc = AllocationBuilder(instance)
    kwargs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order, 'target_category_pair': ('c1', 'c2'), 'target_category': 'c1'}

    # Extract the set of required arguments for the chosen algorithm
    required_args = algo_args.get(algorithm, set())

    # Filter kwargs to include only those required by the chosen algorithm
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_args}
    #Egalitarian algorithm
    valuation_matrix = [[instance.agent_item_value(agent, item) for item in instance.items] for agent in
                        instance.agents]# simply forming valuations as in matrix so we can run fractional egalitarian algorithm
    not_rounded_egal=fractional_egalitarian_allocation(Instance(valuation_matrix),normalize_utilities=False)
    epsilon = 1 # replaced because a number (0,1) isnt a good apprach since it gives a very large number when being divided by
    min_egalitarian_algorithm_value=min(not_rounded_egal)#egalitarian value
    min_egalitarian_algorithm_value_denominator=min_egalitarian_algorithm_value if min_egalitarian_algorithm_value!=0 else min_egalitarian_algorithm_value+epsilon
    #experiments_csv.logger.info(f'valuation_matrix -> {valuation_matrix} \n and egalitarian allocation ->{not_rounded_egal} \n min value of it is -> {min_egalitarian_algorithm_value}')
    #running current algorithm with the appropriate arguments
    algorithm(**filtered_kwargs)
    min_algorithm_bundle_value=min(alloc.agent_bundle_value(agent,bundle) for agent,bundle in alloc.bundles.items())# to compare with egalitarian algorithm
    # Utilitarian algorithm
    alloc_utilitarian=AllocationBuilder(instance)
    utilitarian_matching(alloc_utilitarian)
    utilitarian_bundle_sum=sum(alloc_utilitarian.agent_bundle_value(agent,bundle)for agent,bundle in alloc_utilitarian.bundles.items())
    utilitarian_bundle_sum_denominator=utilitarian_bundle_sum if utilitarian_bundle_sum>0 else utilitarian_bundle_sum+epsilon # epsilon=1 to prevent big numbers
    current_algorithm_bundle_sum=sum(alloc.agent_bundle_value(agent,bundle)for agent,bundle in alloc.bundles.items())
    #experiments_csv.logger.info(f'utilitarian algorithm bundle sum is ->{utilitarian_bundle_sum} \n current algorithm {algorithm.__name__} bundle sum is ->{current_algorithm_bundle_sum}')




    return {'egalitarian_algorithm_min_value':min_egalitarian_algorithm_value,'current_algorithm_min_value':min_algorithm_bundle_value,'ratio_egalitarian':min_algorithm_bundle_value/min_egalitarian_algorithm_value_denominator,'utilitarian_algorithm_sum':utilitarian_bundle_sum,'current_algorithm_sum':current_algorithm_bundle_sum,'ratio_utilitarian':current_algorithm_bundle_sum/utilitarian_bundle_sum_denominator}

if __name__ == '__main__':
    #experiments_csv.logger.setLevel(logging.INFO)
    #compare_heterogeneous_matroid_constraints_algorithms_egalitarian_utilitarian()
    #experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms.csv',filter={},x_field='num_of_agents',y_field='ratio_egalitarian',z_field='algorithm',save_to_file='results/egalitarian_comparison_heterogeneous_constraints_algorithms.png') # egalitarian ratio plot
    #experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms.csv',filter={},x_field='num_of_agents',y_field='ratio_utilitarian',z_field='algorithm',save_to_file='results/utilitarian_comparison_heterogeneous_constraints_algorithms.png') # utilitarian ratio plot
    #experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms.csv',filter={},x_field='num_of_agents',y_field='runtime',z_field='algorithm',save_to_file='results/runtime_comparison_heterogeneous_constraints_algorithms.png') # runtime plot
    pass

