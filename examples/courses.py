#' # Fair item allocation algorithms

import fairpyx
divide = fairpyx.divide


#' `fairpy` contains various algorithms for fair allocation of course-seats.
#' Before starting the algorithms, let us create some inputs for them.
#' There are two agents (students), "avi" and "beni", each of them requires three seats:
agent_capacities = {"avi": 3, "beni": 3}
#' There are four items (courses), each of them has a different number of seats:
item_capacities  = {"w": 2, "x": 1, "y": 2, "z": 1}
#' Students assign different values to different courses:
valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":5, "y":2, "z":4, "w":3}}
#' Construct the instance:
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
#' Compute the allocation:
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)

#' add item conflicts (- courses that cannot be taken simultaneously):
item_conflicts={"x": ["w"], "w": ["x"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, item_conflicts=item_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)

#' add agent conflicts (- courses that cannot be taken by some agent, e.g. due to missing prerequisites):
agent_conflicts={"avi": ["w"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, agent_conflicts=agent_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)

#' create a random instance:
random_instance = fairpyx.Instance.random_uniform(
    num_of_agents=10, num_of_items=3, agent_capacity_bounds=[2,3], item_capacity_bounds=[5,7], 
    item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.8,1.2],
    normalized_sum_of_values=1000)

#' Try various algorithms:
print(divide(fairpyx.algorithms.round_robin, instance=random_instance))
print(divide(fairpyx.algorithms.bidirectional_round_robin, instance=random_instance))
print(divide(fairpyx.algorithms.serial_dictatorship, instance=random_instance))
print(divide(fairpyx.algorithms.almost_egalitarian_with_donation, instance=random_instance))
