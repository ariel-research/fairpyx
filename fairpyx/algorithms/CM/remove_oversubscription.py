"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation
Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Naama Shiponi and Ben Dabush
1/6/2024
"""
from fairpyx.instances import Instance
from A_CEEI import course_demands
"""
Algorithm 2 : The algorithm makes sure that there are no courses that have more students registered than their capacity.
"""

def remove_oversubscription(price_vector: tuple, student_budgets: tuple, instance: Instance, epsilon: float, course_demands: callable):
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
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                         "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> price_vector = [1.2,0.9,1]
    >>> epsilon = 0.1
    >>> student_budgets = [2.2,2.1,2]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, course_demands)
    [1.26875,0.9, 1.24375]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 3, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 3, "c2": 3, "c3": 3}, 
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                         "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> price_vector = [1.2,0.9,1]
    >>> epsilon = 0.1
    >>> student_budgets = [2.2,2.1,2]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, course_demands)
    [1.2,0.9,1]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 3, "Bob": 3, "Tom": 3}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1}, 
    ...   valuations       = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                         "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                         "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> price_vector = [0,0,0]
    >>> epsilon = 0.1
    >>> student_budgets = [2.2,2.1,2]
    >>> remove_oversubscription(price_vector, student_budgets, instance, epsilon, course_demands)
    [2.0375,1.6875, 2.084375]
    """
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
