"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

def course_match_algorithm(instance: Instance):
    """
    Perform the Course Match algorithm to find the best course allocations.
    
    :param instance: (Instance)

    :return: (dict) course allocations

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 1, "Bob": 1, "Tom": 1}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ...   valuations = {"Alice": {"c1": 100, "c2": 0, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ...                 "Tom": {"c1": 0, "c2": 0, "c3": 100}
    ... })
    >>> course_match_algorithm(instance)
    {'c1': ['Alice'], 'c2': ['Bob'], 'c3': ['Tom']}

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2}, 
    ...   item_capacities  = {"c1": 2, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 100, "c2": 100, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 100},
    ...                 "Tom": {"c1": 100, "c2": 0, "c3": 100}
    ... })
    >>> course_match_algorithm(instance)
    {'c1': ['Alice', 'Tom'], 'c2': ['Alice', 'Bob'], 'c3': ['Bob', 'Tom']}

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 1},
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 100, "c2": 60, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ... })
    >>> course_match_algorithm(instance)
    {'c1': ['Alice'], 'c2': ['Alice', 'Bob'], 'c3': []}
    """
    pass



if __name__ == "__main__":
    import doctest
    doctest.testmod()