"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
import random
import time
import numpy as np
from fairpyx.instances import Instance

"""
Algorithm 1: Approximate Competitive Equilibrium from Equal Incomes (A-CEEI), finds the best price vector that matches student preferences and course capacities.
"""
def alpha(demands):
    """    
    Calculate the alpha value given the error vector demands.
    
    :param demands (array): Clearing error vector, d(price_vector)
    
    :returns (float) Alpha value which is the RMSE of the error vector
    
    Examples:
    >>> demands = np.array([1., 0., 0.])
    >>> alpha(demands)
    1.0
    
    >>> demands = np.array([20., 0., 0.])
    >>> alpha(demands)
    20.0

    >>> demands = np.array([10., 5., 0.])
    >>> alpha(demands)
    11.180339887498949
    
    >>> demands = np.array([0., 0., 1.])
    >>> alpha(demands)
    1.0
    
    >>> demands = np.array([2., 3., 3.])
    >>> alpha(demands)
    4.69041575982343

    # Edge cases
    >>> demands = np.array([0., 0., 0.])
    >>> alpha(demands)
    0.0

    >>> demands = np.array([200., 0., 200.])
    >>> alpha(demands)
    282.842712474619

    >>> demands = np.array([50., 50., 50.])
    >>> alpha(demands)
    86.60254037844386  
    """

def d(price_vector,instance: Instance):
    """
    Calculate the clearing error vector given the price vector, enrollment matrix, and target capacities.

    :param price_vector (list): Prices for courses
    :param instance (Instance): Instance object.

    :return (list) Clearing error vector 

    :pseudo code
        1. for each course j:
        2.     if price_vector[j] > 0:
        3.         error_vector[j] = sum(valuations[:, j]) - item_capacities[j]
        4.     else:
        5.         error_vector[j] = max(sum(valuations[:, j]) - item_capacities[j], 0)
        6. return error_vector

    :example
    >>> price_vector = [0, 0, 0]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 100, "c2": 150, "c3": 300}, 
    ...   valuations = {
    ...     "Alice": {"c1": 10, "c2": 20, "c3": 30},
    ...     "Bob": {"c1": 40, "c2": 50, "c3": 60},
    ...     "Tom": {"c1": 70, "c2": 80, "c3": 90}
    ... })
    >>> d(price_vector, instance)
    array([20.,  0.,  0.])
    
    

    >>> price_vector = [5, 0, 10]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 20, "c2": 40, "c3": 60}, 
    ...   valuations = {
    ...     "Alice": {"c1": 5, "c2": 10, "c3": 15},
    ...     "Bob": {"c1": 10, "c2": 15, "c3": 20},
    ...     "Tom": {"c1": 15, "c2":20, "c3": 25}
    ... })
    >>> d(price_vector, instance)
    array([10.,  5.,  0.])


    >>> price_vector = [1, 1, 0]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 3, "c2": 3, "c3": 2}, 
    ...   valuations = {
    ...     "Alice": {"c1": 1, "c2": 1, "c3": 1},
    ...     "Bob": {"c1": 1, "c2": 1, "c3": 1},
    ...     "Tom": {"c1": 1, "c2":1, "c3": 1}
    ... })
    >>> d(price_vector, instance)
    array([0., 0., 1.])


    >>> price_vector = [0, 5, 5]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 10, "c2": 12, "c3": 15}, 
    ...   valuations = {
    ...     "Alice": {"c1": 1, "c2": 2, "c3": 3},
    ...     "Bob": {"c1": 4, "c2": 5, "c3": 6},
    ...     "Tom": {"c1": 7, "c2":8, "c3": 9}
    ... })
    >>> d(price_vector, instance)
    array([2., 3., 3.])


    >>> price_vector = [0, 0, 0]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 0, "c2": 0, "c3": 0},
    ...   valuations = {
    ...     "Alice": {"c1": 0, "c2": 0, "c3": 0},
    ...     "Bob": {"c1": 0, "c2": 0, "c3": 0},
    ...     "Tom": {"c1": 0, "c2": 0, "c3": 0}
    ... })
    >>> d(price_vector, instance)
    array([0., 0., 0.])

    >>> price_vector = [10, 0, 5]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 100, "c2": 0, "c3": 100},
    ...   valuations = {
    ...     "Alice": {"c1": 100, "c2": 0, "c3": 100},
    ...     "Bob": {"c1": 100, "c2": 0, "c3": 100},
    ...     "Tom": {"c1": 100, "c2": 0, "c3": 100}
    ... })
    >>> d(price_vector, instance)
    array([100.,   0., 100.])

    >>> price_vector = [0, 0, 0]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 100, "c2": 100, "c3": 100},
    ...   valuations = {
    ...     "Alice": {"c1": 50, "c2": 50, "c3": 50},
    ...     "Bob": {"c1": 50, "c2": 50, "c3": 50},
    ...     "Tom": {"c1": 50, "c2": 50, "c3": 50}
    ... })
    >>> d(price_vector, instance)
    array([50., 50., 50.])
    """
   

def N(price_vector,instance: Instance):
    """
    Generate the neighbors of the price vector p sorted by increasing alpha^2 values.

    :param price_vector (list): Prices for courses
    :param instance (Instance): Instance object.

    :return (list) List of neighbors sorted by increasing alpha^2 values

    :example

    >>> price_vector = [50, 100, 15, 20]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 3, "c2": 1, "c3": 2, "c4": 2},
    ...   valuations = {
    ...     "Alice": {"c1": 90, "c2": 100, "c3": 20, "c4": 40},
    ...     "Bob": {"c1": 100, "c2": 80, "c3": 50, "c4": 60},
    ...     "Tom": {"c1": 50, "c2": 50, "c3": 50, "c4": 50}
    ... })
    >>> N(price_vector, instance)
    [[50, 100, 16, 20], [50, 100, 15, 21], [50, 101, 15, 20], [51, 100, 15, 20], [73, 122, 26, 34], [93, 141, 36, 46], [112, 160, 46, 59], [131, 179, 55, 71], [151, 197, 65, 83], [170, 216, 75, 95], [190, 235, 84, 107], [209, 254, 94, 119], [228, 272, 104, 131], [248, 291, 113, 143], [267, 310, 123, 155], [287, 329, 133, 168]]

   """
    pass

def generate_individual_adjustment_neighbors(price_vector, demands):
    """
    Generate neighbors by adjusting the price of each course individually.

    :param price_vector (list): Prices for courses
    :param demands (list): Clearing error vector

    :return (list) List of neighbors

    :example
    >>> price_vector = [50, 100, 15, 20]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 3, "c2": 1, "c3": 2, "c4": 2},
    ...   valuations = {
    ...     "Alice": {"c1": 90, "c2": 100, "c3": 20, "c4": 40},
    ...     "Bob": {"c1": 100, "c2": 80, "c3": 50, "c4": 60},
    ...     "Tom": {"c1": 50, "c2": 50, "c3": 50, "c4": 50}
    ... })
    >>> demands = d(price_vector,instance) # =[237, 229, 118, 148]
    >>> generate_individual_adjustment_neighbors(price_vector, demands)
    [[73, 122, 26, 34], [93, 141, 36, 46], [112, 160, 46, 59], [131, 179, 55, 71], [151, 197, 65, 83], [170, 216, 75, 95], [190, 235, 84, 107], [209, 254, 94, 119], [228, 272, 104, 131], [248, 291, 113, 143], [267, 310, 123, 155], [287, 329, 133, 168]]
    
    >>> price_vector = [100, 30, 90]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 1},
    ...   valuations = {
    ...     "Alice": {"c1": 10, "c2": 20, "c3": 30},
    ...     "Bob": {"c1": 40, "c2": 50, "c3": 60},
    ...     "Tom": {"c1": 70, "c2": 80, "c3": 90}
    ... })
    >>> demands = d(price_vector,instance) #
    >>> generate_individual_adjustment_neighbors(price_vector, demands)
    [[111, 44, 107], [121, 56, 122], [131, 69, 137], [141, 81, 151], [150, 93, 166], [160, 105, 181], [170, 117, 195], [180, 129, 210], [189, 141, 225], [199, 153, 239], [209, 165, 254], [219, 178, 269]]
    """
    pass
    
def generate_gradient_neighbors(price_vector, demands, item_capacities):
    """
    Generate neighbors by adjusting the price of the most over and under subscribed courses.

    :param price_vector (list): Prices for courses
    :param demands (list): d(price_vector)
    :param item_capacities (list): Target capacities for courses

    :return (list) List of neighbors

    :example
    price_vector = [50, 100, 15, 20]
    instance = Instance(
        item_capacities  = {"c1": 3, "c2": 1, "c3": 2, "c4": 2},
        valuations = {
            "Alice": {"c1": 90, "c2": 100, "c3": 20, "c4": 40},
            "Bob": {"c1": 100, "c2": 80, "c3": 50, "c4": 60},
            "Tom": {"c1": 50, "c2": 50, "c3": 50, "c4": 50}
        })
    demands = d(price_vector,instance) # =[237, 229, 118, 148]
    generate_gradient_neighbors(price_vector, demands, [3, 1, 2, 2])
    [[50, 100, 16, 20], [50, 100, 15, 21], [50, 101, 15, 20], [51, 100, 15, 20]]

    >>> price_vector = [10, 30, 20]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 1},
    ...   valuations = {
    ...     "Alice": {"c1": 10, "c2": 20, "c3": 30},
    ...     "Bob": {"c1": 40, "c2": 50, "c3": 60},
    ...     "Tom": {"c1": 70, "c2": 80, "c3": 90}
    ... })
    >>> demands = d(price_vector,instance) # [119. 148. 179.]
    >>> generate_gradient_neighbors(price_vector, demands, [1, 2, 1])
    [[11, 30, 20], [10, 31, 20], [10, 30, 21]]

    >>> price_vector = [5, 10, 15]
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 3, "c2": 3, "c3": 2},
    ...   valuations = {
    ...     "Alice": {"c1": 1, "c2": 1, "c3": 1},
    ...     "Bob": {"c1": 1, "c2": 1, "c3": 1},
    ...     "Tom": {"c1": 1, "c2": 1, "c3": 1}
    ... })
    >>> demands = d(price_vector,instance) # [0. 0. 1.]
    >>> generate_gradient_neighbors(price_vector, demands, [3, 3, 2])
    [[0, 10, 15], [5, 0, 15], [5, 10, 16]]
    """

    pass


def A_CEEI(instance: Instance, beta = 100, time = 10):
    """
    Perform heuristic search to find the best price vector that matches student preferences and course capacities.

    :param instance (Instance): Instance object.
    :param beta (float): Initial budget.
    :param time (float): Time limit for the search.

    :return (tuple) Tuple containing the best price vector and the best error.

    :pseudo code
    input: β̄ # maximum budget, t # time limit in seconds, x # enrollment matrix, q # target capacities
    Output: p* price vector corresponding to approximate competitive equilibrium with lowest clearing error.

    1: besterror ← ∞  # besterror tracks the best error found over every search start
    2: repeat
    3:     p ← (U[0, 1] · β̄)_Mj_1  # Start the search from a random, reasonable price vector
    4:     searcherror ← alpha(d(p))  # searcherror tracks the best error found in this search start
    5:     τ ← ∅  # τ is the tabu list
    6:     c ← 0  # c tracks the number of steps taken without improving error
    7:     while c < 5 do  # Restart the search if we have not improved our error in five steps
    8:         N ← N(p)  # This requires evaluating the clearing error of each neighbor
    9:         foundnextstep ← false
    10:        repeat
    11:            p̃ ← N.pop()  # Remove the front of the neighbor list
    12:            z ← d(p̃)
    13:            if z < τ then  # If p̃ does not induce demands found in our tabu list, it becomes the next step in our search
    14:                foundnextstep ← true
    15:            end if
    16:        until foundnextstep or N.empty()
    17:        if N.empty() then
    18:            c ← 5  # All neighbors are in the tabu list; force a restart
    19:        else  # p̃ has the next step of the search
    20:            p ← p̃
    21:            τ.append(z)
    22:            currenterror ← alpha^2(z)
    23:            if currenterror < searcherror then
    24:                searcherror ← currenterror
    25:                c ← 0  # We improved our search solution, so reset the step counter
    26:            else  # We did not improve our solution from this search start, so add to the step counter
    27:                c ← c + 1
    28:            end if
    29:            if currenterror < besterror then
    30:                besterror ← currenterror
    31:                p* ← p
    32:            end if
    33:        end if
    34:    end while
    35: until current time > t

    """
    pass    

if __name__ == "__main__":
    import doctest
    doctest.testmod()



