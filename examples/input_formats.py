#' # Input formats

import fairpyx
divide = fairpyx.divide

#' `fairpyx` allows various input formats, so that you can easily use it on your own data,
#' whether for applications or for research.


#' ## Valuations

#' Suppose you want to divide candies among your children.
#' It is convenient to collect their preferences in a dict of dicts:

valuations = {
    "Ami": {"green": 8, "red":7, "blue": 6, "yellow": 5},
    "Tami": {"green": 12, "red":8, "blue": 4, "yellow": 2} }
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations)

#' You can then see the resulting allocation with the agents' real names:

print(allocation) 

#' For research, passing a dict of dicts as a parameter may be too verbose.
#' You can call the same algorithm with only the values, or only the value matrix:

print(divide(fairpyx.algorithms.round_robin, valuations={"Ami": [8,7,6,5], "Tami": [12,8,4,2]}))
print(divide(fairpyx.algorithms.round_robin, valuations=[[8,7,6,5], [12,8,4,2]]))


#' For experiments, you can use a numpy random matrix. The code below generates random values for 5 agents and 12 courses:

import numpy as np
valuations = np.random.randint(1,100,[5,12])
print(valuations)
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations)
print(allocation)


#' ## Capacities

#' There are several input formats for agent capacities. You can set the same capacity to all agents:

allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  
print(allocation)

#' Or different capacities to different agents:

allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=[1,2,3,2,1]) 
print(allocation)

#' Similarly, you can set the same capacity to all items:

allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)

#' Or different capacities to different items:

allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=[1,2,1,2,1,2,1,2,1,2,1,2])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)


#' ## Conflicts

#' You can specify agent_conflicts - a set of items that cannot be allocated to this agent (e.g. due to missing preliminaries):

valuations = {
    "Ami": {"green": 8, "red":7, "blue": 6, "yellow": 5},
    "Tami": {"green": 12, "red":8, "blue": 4, "yellow": 2} }
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations, agent_conflicts={"Ami": ["green", "red", "blue"], "Tami": ["red", "blue", "yellow"]}) 
print(allocation)

#' You can also specify item_conflicts - a set of items that cannot be taken together (e.g. due to overlapping times):

allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations, item_conflicts={"green": ["yellow", "red", "blue"]})
print(allocation)

#' Note that not all algorithms can handle conflicts.

