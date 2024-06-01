"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation
Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Naama Shiponi and Ben Dabush
1/6/2024
"""
from fairpyx.instances import Instance
"""
Algorithm 2 : The algorithm makes sure that there are no courses that have more students registered than their capacity.
"""

def remove_oversubscription(price_vector, student_budgets, instance, epsilon, demand_function):
    """
    Perform oversubscription elimination to adjust course prices.

    :param price_vector: Initial price vector (List of floats)
    :param student_budgets: List of student budgets (list of floats)
    :param instance: Instance
    :param epsilon: Small value to determine when to stop binary search
    :param demand_function: Function that takes price vector and returns excess demand vector
    
    :return: Adjusted price vector (list of floats)

    :pseudo code
    Input:  p* heuristic search solution price vector from Algorithm 1,
            ¯p scalar price greater than any budget,
            ε smallerthan budget differences,
            excess demand function d(p) that maps a price vector to the demand of a coursebeyond its maximum capacity.
    Output: Altered p* without oversubscription

    1:  j' ← argMax_j (d_j (p*))  # j' is the most oversubscribed course
    2:  while d_j'(p*) > 0 do
    3:      d* ← d_j'(p*)/2  # Perform binary search on the price of course j' until oversubscription equals (at most) d*
    4:      pl ← p*_j'
    5:      ph ← ¯p
    6:      repeat  # Our target price is always in the interval [pl ,ph], which we progressively shrink inhalf in each iteration of this loop
    7:          p*_j' ← (pl + ph )/2
    8:          if d_j'(p*) > d* then
    9:              pl ← p*_j'
    10:         else
    11:             ph ← p*_j'
    12:         end if
    13:     until ph - pl < ε
    14:     p*_j' ← ph  # Set to the higher price to be sure oversubscription is at most d*
    15:     j' ← argMax_j d_j(p*)  # Find the most oversubscribed course with the new prices
    16: end while

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1}, 
    ...   valuations       = {"Alice": {"c1": 55, "c2": 55, "c3": 100},
    ...                         "Bob": {"c1": 40, "c2": 60, "c3": 40},
    ...                         "Tom": {"c1": 70, "c2": 70, "c3": 100}
    ... })
    >>> price_vector = [40, 50, 50]
    >>> epsilon = 2
    >>> student_budgets = [90,100,110]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, demand_function)
    [84, 98, 108]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1}, 
    ...   valuations       = {"Alice": {"c1": 55, "c2": 55, "c3": 100},
    ...                         "Bob": {"c1": 40, "c2": 60, "c3": 40},
    ...                         "Tom": {"c1": 70, "c2": 70, "c3": 100}
    ... })
    >>> price_vector = [40, 50, 50]
    >>> epsilon = 1
    >>> student_budgets = [90,100,110]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, demand_function)
    [82, 99, 109]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1}, 
    ...   valuations       = {"Alice": {"c1": 55, "c2": 55, "c3": 100},
    ...                         "Bob": {"c1": 40, "c2": 60, "c3": 40},
    ...                         "Tom": {"c1": 70, "c2": 70, "c3": 100}
    ... })
    >>> price_vector = [0, 0, 0]
    >>> epsilon = 2
    >>> student_budgets = [90,100,110]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, demand_function)
    [85, 99, 108]
    """
    pass

def demand_function(p,instance, student_budgets):
    """
    :param p (list): Price vector
    :param instance (Instance)
    :param student_budgets (list of floats): List of student budgets 
  
    :return (list) List of demands for each course.
    
    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"x": 1, "y": 1, "z": 1},
    ...   valuations       = {"Alice": {"x": 55, "y": 55, "z": 100},
    ...                         "Bob": {"x": 40, "y": 60, "z": 40},
    ...                         "Tom": {"x": 70, "y": 70, "z": 100}})
    >>> p = [40, 50, 50]
    >>> student_budgets = [110, 100, 90]
    >>> demand_function(p,instance, student_budgets)
    [1, 3, 2]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"x": 1, "y": 1, "z": 1},
    ...   valuations       = {"Alice": {"x": 55, "y": 55, "z": 100},
    ...                         "Bob": {"x": 40, "y": 60, "z": 40},
    ...                         "Tom": {"x": 70, "y": 70, "z": 100}
    ... })
    >>> p = [200, 200, 200]
    >>> student_budgets = [110, 100, 90]
    >>> demand_function(p,instance, student_budgets)
    [0, 0, 0]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"x": 1, "y": 1, "z": 1},
    ...   valuations       = {"Alice": {"x": 55, "y": 55, "z": 100},
    ...                         "Bob": {"x": 40, "y": 60, "z": 40},
    ...                         "Tom": {"x": 70, "y": 70, "z": 100}
    ... })
    >>> p = [0, 0, 0]
    >>> student_budgets = [110, 100, 90]
    >>> demand_function(p,instance, student_budgets)
    [2, 3, 2]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 3, "Tom": 3},
    ...   item_capacities  = {"x": 1, "y": 1, "z": 1},
    ...   valuations       = {"Alice": {"x": 55, "y": 55, "z": 100},
    ...                         "Bob": {"x": 40, "y": 60, "z": 40},
    ...                         "Tom": {"x": 70, "y": 70, "z": 100}
    ... })
    >>> p = [70, 60, 110]
    >>> student_budgets = [110, 80, 90]
    >>> demand_function(p,instance, student_budgets)
    [1, 1, 1]
    """
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
