#' # Input formats

import fairpyx
divide = fairpyx.divide

#' `fairpyx` allows various input formats, so that you can easily use it on your own data,
#' whether for applications or for research.
#' For example, suppose you want to divide candies among your children.
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


#' For experiments, you can use a numpy random matrix:

import numpy as np
valuations = np.random.randint(1,100,[2,4])
print(valuations)
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations)
print(allocation)
