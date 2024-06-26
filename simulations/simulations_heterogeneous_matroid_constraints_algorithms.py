from tests.test_heterogeneous_matroid_constraints_algorithms import random_instance
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import *
import experiments_csv

input_ranges={
    'equal_capacities':True,
    'category_count':[1],
    'item_capacity_bounds':(1, 1),
    'random_seed_num':0,
    'algorithm':[per_category_round_robin,capped_round_robin,two_categories_capped_round_robin,per_category_capped_round_robin,iterated_priority_matching]
}
def start_experiment(equal_capacities:bool,category_count:int,item_capacity_bounds:tuple[int,int],random_seed_num:int,algorithm:callable):
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=equal_capacities,
                                                                                           category_count=category_count,
                                                                                           item_capacity_bounds=item_capacity_bounds,random_seed_num=random_seed_num)
    algorithm({'alloc':AllocationBuilder(instance),'agent_category_capacities':agent_category_capacities,'item_categories':categories,'initial_agent_order':initial_agent_order})


if __name__ == '__main__':
    ex=experiments_csv.Experiment()
    ex.run(start_experiment,input_ranges)