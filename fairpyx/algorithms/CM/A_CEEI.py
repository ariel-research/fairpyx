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
def A_CEEI(instance: Instance, budget : dict , time : int = 60):
    """
    Perform heuristic search to find the best price vector that matches student preferences and course capacities.

    :param instance (Instance): Instance object.
    :param budget (float): Initial budget.
    :param time (float): Time limit for the search.

    :return (tuple) Tuple containing the best price vector and the best error.
    """
    pass

def course_demands(price_vector: list ,instance: Instance,  budget : dict, preferred_schedule: dict):
    """
    :param price_vector: List of prices.
    :param instance: Instance object.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (dict) Dictionary of course demands.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = [1, 1, 1]
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice":  ["AC", "CB", "AB"] , "Bob": ["AB", "AC" "BC"], "Tom": ["AC", "AB", "BC"]}
    >>> course_demands(price_vector, instance, budget, preferred_schedule)
    {'c1': 2, 'c2': -1, 'c3': 0}

    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = [1.2,0.9,1]
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice":  ["AC", "CB", "AB"] , "Bob": ["AB", "AC" "BC"], "Tom": ["AC", "AB", "BC"]}
    >>> course_demands(price_vector, instance, budget, preferred_schedule)
    {'c1': 1, 'c2': 0, 'c3': 0}
    """
    pass

def alpha(course_demands: dict):
    """
    :param course_demands: Dictionary of course demands.

    :return (float) Alpha value.

    :example
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> alpha(course_demands) # sqrt(5)
    2.23606797749979
    
    >>> course_demands = {"c1": 1, "c2": 1, "c3": 1}
    >>> alpha(course_demands) # sqrt(3)
    1.7320508075688772

    >>> course_demands = {"c1": 0, "c2": 0, "c3": 0}
    >>> alpha(course_demands)
    0.0
    """
    return np.sqrt(sum([v**2 for v in course_demands.values()]))

def find_neighbors(price_vector: list ,instance: Instance,course_demands: dict, budget : dict, steps: list ):
    """
    :param price_vector: List of prices.
    :param instance: Instance object.
    :param course_demands: Dictionary of course demands.
    :param budget: Dictionary of budgets.
    :param steps: List of steps.

    :return (list of list) List of neighbors.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = [1, 1, 1]
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> steps = [0.1, 0.2]
    >>> find_neighbors(price_vector, instance, course_demands, budget, steps)
    [[1.2, 0.9, 1], [1.1, 1, 1], [1.4, 0.8, 1], [1, 0, 1]]

    """
    pass


def generate_individual_adjustment_neighbors(price_vector: list, instance: Instance, course_demands: dict, budget : dict ):
    """
    Generate individual adjustment neighbors.

    :param price_vector: List of prices.
    :param instance: Instance object.
    :param course_demands: Dictionary of course demands.
    :param budget: Dictionary of budgets.

    :return (list of list) List of individual adjustment neighbors.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = [1, 1, 1]
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> generate_individual_adjustment_neighbors(price_vector, instance, course_demands, budget)
    [[1.1, 1, 1], [1, 0, 1]]
    """
    pass
    
def generate_gradient_neighbors(price_vector: list, course_demands: dict, steps: list):
    """
    Generate gradient neighbors.

    :param price_vector: List of prices.
    :param course_demands: Dictionary of course demands.
    :param steps: List of steps.

    :return (list of list) List of gradient neighbors.

    :example
    >>> price_vector = [1, 1, 1]
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, course_demands, steps)
    [[1.2, 0.9, 1.0], [1.4, 0.8, 1.0]]

    >>> price_vector = [1, 1, 1]
    >>> course_demands = {"c1": 1, "c2": 1, "c3": 1}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, course_demands, steps)
    [[1.1, 1.1, 1.1], [1.2, 1.2, 1.2]]

    >>> price_vector = [0, 0, 0]
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, course_demands, steps)
    [[0.2, -0.1, 0.0], [0.4, -0.2, 0.0]]

    >>> price_vector = [1, 1, 1]
    >>> course_demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2, 0.3, 0.4, 0.5]
    >>> generate_gradient_neighbors(price_vector, course_demands, steps)
    [[1.2, 0.9, 1.0], [1.4, 0.8, 1.0], [1.6, 0.7, 1.0], [1.8, 0.6, 1.0], [2.0, 0.5, 1.0]]
    """
    return [[(price_vector[i] + (steps[j] * i)) for i in course_demands.values()] for j in range(len(steps))]
  

if __name__ == "__main__":
    import doctest
    doctest.testmod()



