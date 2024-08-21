import experiments_csv

from fairpyx.algorithms.fractional_egalitarian import fractional_egalitarian_allocation
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import *
from fairpyx.utils.test_heterogeneous_matroid_constraints_algorithms_utils import *


def compare_heterogeneous_matroid_constraints_algorithms_egalitarian_utilitarian():  #egalitarian: prioritizes the poor
    """
    this function contains 6 input ranges in which
    input_ranges_intersection: is an input range which is acceptable to all the algorithms combined
    input_ranges_algorithm_1 : per-category RR same capacities
    input_ranges_algorithm_2 :CRR -> single category
    input_ranges_algorithm_3 : back and forth CRR -> 2 categories
    input_ranges_algorithm_4 : per-category CRR -> equal valuations
    input_ranges_algorithm_5 : iterated priority-matching -> binary valuations
    """
    expr=experiments_csv.Experiment('results/', 'egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms_bigData.csv')
    expr.clear_previous_results()#TODO close after saving results
    input_ranges_intersection = { #an input range which is appropriate for all the algorithms
        'equal_capacities': [True],
        'equal_valuations': [True],
        'binary_valuations': [True],
        'num_of_items':range(10,20),
        'category_count': [2],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(10,100),
        'algorithm': [per_category_round_robin, capped_round_robin,
                      per_category_capped_round_robin, two_categories_capped_round_robin, iterated_priority_matching,
        egalitarian_algorithm,utilitarian_algorithm]
    }
    input_ranges_algorithm_1 = {#same capacities else doesn't matter
        'equal_capacities': [True],
        'equal_valuations': [True,False],
        'binary_valuations': [True,False],
        'num_of_items': range(10,20),
        'category_count': [10],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(1,100),
        'algorithm': [per_category_round_robin,
        egalitarian_algorithm,utilitarian_algorithm,iterated_maximum_matching]
    } # equal capacities for the sake of the compatibility of input with the implemented egalitarian and utilitarian algorithms
    # we also need to consider giving each agent a capacity which is >= number of items
    input_ranges_algorithm_2 = {# crr -> single category
        'equal_capacities': [True,False],
        'equal_valuations': [True,False],
        'binary_valuations': [True,False],
        'num_of_items': range(10,20),
        'category_count': [1],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(1,100),
        'algorithm': [
                      capped_round_robin,egalitarian_algorithm,utilitarian_algorithm]
    }
    input_ranges_algorithm_3 = { # back and forth crr -> 2 categories
        'equal_capacities': [True,False],
        'equal_valuations': [True,False],
        'binary_valuations': [True,False],
        'num_of_items': range(10, 20),
        'category_count': [2],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(1, 100),
        'algorithm': [
            two_categories_capped_round_robin,utilitarian_algorithm,egalitarian_algorithm]
    }
    input_ranges_algorithm_4 = { # per-category crr -> equal valuations
        'equal_capacities': [True,False],
        'equal_valuations': [True],
        'binary_valuations': [True,False],
        'num_of_items': range(10, 20),
        'category_count': [10],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(1, 100),
        'algorithm': [
            per_category_capped_round_robin,egalitarian_algorithm,utilitarian_algorithm]
    }
    input_ranges_algorithm_5 = {# iterated priority-matching -> binary valuations
        'equal_capacities': [True,False],
        'equal_valuations': [True,False],
        'binary_valuations': [True],
        'num_of_items': range(10, 20),
        'category_count': [10],
        'item_capacity_bounds': range(1, 1 + 1),
        'random_seed_num': range(0,10),
        'num_of_agents': range(1, 100),
        'algorithm': [
            iterated_priority_matching,egalitarian_algorithm,utilitarian_algorithm]
    }
    expr.run_with_time_limit(run_experiment,input_ranges_intersection,10)
    expr.run_with_time_limit(run_experiment, input_ranges_algorithm_1, 10)
    expr.run_with_time_limit(run_experiment, input_ranges_algorithm_2, 10)
    expr.run_with_time_limit(run_experiment, input_ranges_algorithm_3, 10)
    expr.run_with_time_limit(run_experiment, input_ranges_algorithm_4, 10)
    expr.run_with_time_limit(run_experiment, input_ranges_algorithm_5, 10)
def run_experiment(equal_capacities:bool,equal_valuations:bool,binary_valuations:bool,category_count:int,item_capacity_bounds:int,random_seed_num:int,num_of_agents:int,algorithm:callable,num_of_items:int):
    # Mapping of algorithms to their specific argument sets
    algo_args = {
        per_category_round_robin: {'alloc', 'agent_category_capacities', 'item_categories', 'initial_agent_order'},
        capped_round_robin: {'alloc', 'item_categories', 'agent_category_capacities',
                             'initial_agent_order', 'target_category'},
        two_categories_capped_round_robin: {'alloc', 'item_categories', 'agent_category_capacities', 'initial_agent_order','target_category_pair'},
        per_category_capped_round_robin: {'alloc', 'agent_category_capacities', 'item_categories', 'initial_agent_order'},
        iterated_priority_matching: {'alloc', 'item_categories', 'agent_category_capacities'},
        egalitarian_algorithm:{'instance'},
        utilitarian_algorithm:{'instance'},
        iterated_maximum_matching:{'alloc'}

    }
    #print(f'algorithm{algorithm.__name__} , binary valuations ->{binary_valuations}')
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(
        equal_capacities=equal_capacities,
        equal_valuations=equal_valuations,
        binary_valuations=binary_valuations,
        category_count=category_count,
        item_capacity_bounds=(1, item_capacity_bounds), random_seed_num=random_seed_num, num_of_agents=num_of_agents,num_of_items=num_of_items,agent_capacity_bounds=(num_of_items,num_of_items+1))
    alloc = AllocationBuilder(instance)
    kwargs = {'alloc': alloc, 'agent_category_capacities': agent_category_capacities, 'item_categories': categories,
              'initial_agent_order': initial_agent_order, 'target_category_pair': ('c1', 'c2'), 'target_category': 'c1','instance':instance}

    # Extract the set of required arguments for the chosen algorithm
    required_args = algo_args.get(algorithm, set())

    # Filter kwargs to include only those required by the chosen algorithm
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_args}
    current_algorithm_bundle_sum = 0
    current_algorithm_bundle_min_value = float('inf') # useless since there is no comparison could be replaced with 0
    if algorithm.__name__=='egalitarian_algorithm':
        #egalitarian algorithm
        current_algorithm_bundle_sum, current_algorithm_bundle_min_value = egalitarian_algorithm(instance)
    elif algorithm.__name__=='utilitarian_algorithm':
        # Utilitarian algorithm
        current_algorithm_bundle_sum,current_algorithm_bundle_min_value = utilitarian_algorithm(instance)
    # our algorithm
    else:# one of our algorithms then !
        print(f'filtered kwargs->{filtered_kwargs["alloc"].instance._valuations}')
        algorithm(**filtered_kwargs)
        current_algorithm_bundle_min_value=min(alloc.agent_bundle_value(agent,bundle) for agent,bundle in alloc.bundles.items())# to compare with egalitarian algorithm
        current_algorithm_bundle_sum=sum(alloc.agent_bundle_value(agent,bundle)for agent,bundle in alloc.bundles.items())# to compare with utilitarian



    return {'current_algorithm_bundle_min_value':current_algorithm_bundle_min_value,'current_algorithm_bundle_sum':current_algorithm_bundle_sum}


def utilitarian_algorithm(instance):
    alloc_utilitarian = AllocationBuilder(instance)# to make sure we're using a fresh copy of allocation
    utilitarian_matching(alloc_utilitarian)

    agent_bundle_values = [
        alloc_utilitarian.agent_bundle_value(agent, bundle)
        for agent, bundle in alloc_utilitarian.bundles.items()
    ]

    utilitarian_bundle_sum = sum(agent_bundle_values)
    min_utilitarian_value = min(agent_bundle_values)

    return utilitarian_bundle_sum, min_utilitarian_value


def egalitarian_algorithm(instance):
    # Step 1: Form the valuation matrix
    valuation_matrix = [
        [instance.agent_item_value(agent, item) for item in instance.items]
        for agent in instance.agents
    ]

    # Step 2: Compute the fractional egalitarian allocation
    not_rounded_egal = fractional_egalitarian_allocation(
        Instance(valuation_matrix), normalize_utilities=False
    )

    # Step 3: Multiply the fractions by the original valuation matrix
    not_rounded_egalitarian_bundle_matrix = [
        [
            not_rounded_egal[agent][item] * valuation_matrix[agent][item]
            for item in range(len(instance.items))
        ]
        for agent in range(len(instance.agents))
    ]

    # Step 4: Calculate the total value each agent receives from their allocation
    agent_total_values = [
        sum(not_rounded_egalitarian_bundle_matrix[agent])
        for agent in range(len(instance.agents))
    ]

    # Step 5: Find the minimum value among these totals
    min_egalitarian_algorithm_value = min(agent_total_values)

    # Step 6: Calculate the total sum of all allocations (for comparison)
    total_sum = sum(agent_total_values)

    return total_sum, min_egalitarian_algorithm_value

if __name__ == '__main__':
    #experiments_csv.logger.setLevel(logging.INFO)
    #compare_heterogeneous_matroid_constraints_algorithms_egalitarian_utilitarian()
    experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms_bigData.csv',filter={},x_field='num_of_agents',y_field='current_algorithm_bundle_min_value',z_field='algorithm',save_to_file='results/egalitarian_comparison_heterogeneous_constraints_algorithms_bigData.png') # egalitarian ratio plot
    experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms_bigData.csv',filter={},x_field='num_of_agents',y_field='current_algorithm_bundle_sum',z_field='algorithm',save_to_file='results/utilitarian_comparison_heterogeneous_constraints_algorithms_bigData.png') # utilitarian ratio plot
    experiments_csv.single_plot_results('results/egalitarian_utilitarian_comparison_heterogeneous_constraints_algorithms_bigData.csv',filter={},x_field='num_of_agents',y_field='runtime',z_field='algorithm',save_to_file='results/runtime_comparison_heterogeneous_constraints_algorithms_bigData.png') # runtime plot
    pass
#
